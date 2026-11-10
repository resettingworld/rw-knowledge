use std::time::Duration;
use md5::{Md5, Digest};
use sha2::Sha256;

#[derive(Clone)]
pub struct TextEmbedder {
    pub ollama_url: String,
    pub model_name: String,
    pub base_dim: usize,
    client: reqwest::Client,
}

impl Default for TextEmbedder {
    fn default() -> Self {
        Self::new("http://localhost:11434", "embeddinggemma-2", 768)
    }
}

impl TextEmbedder {
    pub fn new(ollama_url: &str, model_name: &str, base_dim: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(400))
            .build()
            .unwrap_or_default();

        Self {
            ollama_url: ollama_url.trim_end_matches('/').to_string(),
            model_name: model_name.to_string(),
            base_dim,
            client,
        }
    }

    pub async fn embed_text(&self, text: &str, target_dim: usize) -> Vec<f32> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return vec![0.0; target_dim];
        }

        // Try Ollama endpoint asynchronously with tight timeout (400ms)
        if let Ok(vec) = self.try_ollama_embed(trimmed).await {
            if !vec.is_empty() {
                return self.apply_matryoshka(vec, target_dim);
            }
        }

        // Deterministic pseudo-semantic projector (Offline Edge / Bunker Fallback)
        let raw_vec = self.semantic_text_embedding(trimmed);
        self.apply_matryoshka(raw_vec, target_dim)
    }

    async fn try_ollama_embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let body = serde_json::json!({
            "model": self.model_name,
            "prompt": text
        });

        let res = self.client
            .post(format!("{}/api/embeddings", self.ollama_url))
            .json(&body)
            .send()
            .await?;

        if res.status().is_success() {
            let json: serde_json::Value = res.json().await?;
            if let Some(arr) = json.get("embedding").and_then(|v| v.as_array()) {
                let vec: Vec<f32> = arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
                if !vec.is_empty() {
                    return Ok(vec);
                }
            }
        }

        Err("Ollama embedding failed or unavailable".into())
    }

    pub fn semantic_text_embedding(&self, text: &str) -> Vec<f32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut vec = vec![0.0f32; self.base_dim];

        for (pos, word) in words.iter().enumerate() {
            let word_lower = word.to_lowercase();
            let mut hasher = Sha256::new();
            hasher.update(word_lower.as_bytes());
            let result = hasher.finalize();

            let weight = 1.0f32 / ((pos + 1) as f32).sqrt();

            for i in 0..self.base_dim {
                let byte_idx = i % result.len();
                let val = (result[byte_idx] as i8) as f32;
                vec[i] += val * weight;
            }
        }

        // 3-gram character hashing
        let chars: Vec<char> = text.chars().collect();
        if chars.len() >= 3 {
            for i in 0..chars.len() - 2 {
                let ngram: String = chars[i..i + 3].iter().collect();
                let mut md5_hasher = Md5::new();
                md5_hasher.update(ngram.as_bytes());
                let md5_res = md5_hasher.finalize();
                let hex_prefix = &md5_res[..4];
                let num = u32::from_be_bytes(hex_prefix.try_into().unwrap());
                let idx = (num as usize) % self.base_dim;
                vec[idx] += 1.0;
            }
        }

        self.normalize(vec)
    }

    pub fn apply_matryoshka(&self, mut vec: Vec<f32>, target_dim: usize) -> Vec<f32> {
        if target_dim < vec.len() {
            vec.truncate(target_dim);
        } else if target_dim > vec.len() {
            vec.resize(target_dim, 0.0);
        }
        self.normalize(vec)
    }

    fn normalize(&self, mut vec: Vec<f32>) -> Vec<f32> {
        let sum_sq: f32 = vec.iter().map(|v| v * v).sum();
        let norm = sum_sq.sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }
}
