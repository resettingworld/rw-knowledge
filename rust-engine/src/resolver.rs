use std::time::{Duration, Instant};
use crate::catalog::RagCatalog;
use crate::embedding::TextEmbedder;
use crate::models::{EngineResponse, QueryRequest, RetrievedChunk, TriageResult};
use crate::retriever::LowLevelRetriever;
use crate::triage::FastTriager;

pub struct DecisionResolver {
    pub catalog: RagCatalog,
    pub embedder: TextEmbedder,
    pub ollama_url: String,
    pub generator_model: String,
    client: reqwest::Client,
}

impl DecisionResolver {
    pub fn new(catalog: RagCatalog, ollama_url: &str, generator_model: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .unwrap_or_default();

        Self {
            catalog,
            embedder: TextEmbedder::new(ollama_url, "embeddinggemma-2", 768),
            ollama_url: ollama_url.trim_end_matches('/').to_string(),
            generator_model: generator_model.to_string(),
            client,
        }
    }

    pub async fn resolve(&self, req: &QueryRequest) -> EngineResponse {
        let start_time = Instant::now();

        // 1. Capa de Triatge Ràpid (Sistema 1 - sub-5ms)
        let triage = FastTriager::triage(&self.catalog, &req.query);

        // Determine which RAGs to query
        let target_rags: Vec<String> = if let Some(ref explicit_rags) = req.target_rags {
            explicit_rags.clone()
        } else {
            let mut top = Vec::new();
            for scored in triage.selected_rags.iter().take(2) {
                if scored.score >= 1.0 || top.is_empty() {
                    top.push(scored.rag_id.clone());
                }
            }
            top
        };

        // 2. Capa de Búsqueda Semántica Vectorial (EmbeddingGemma 2 + Matryoshka)
        let query_vec = self.embedder.embed_text(&req.query, req.dim).await;

        let mut all_retrieved: Vec<RetrievedChunk> = Vec::new();
        let mut extracted_images: Vec<String> = Vec::new();

        for rag_id in &target_rags {
            if let Some(db_path) = self.catalog.get_db_path(rag_id) {
                if let Ok(chunks) = LowLevelRetriever::search_rag(
                    &db_path,
                    rag_id,
                    &query_vec,
                    &req.query,
                    req.top_k,
                    req.dim,
                ) {
                    for chunk in chunks {
                        if let Some(ref asset) = chunk.asset_rel_path {
                            let img_path = format!("assets/{}", asset);
                            if !extracted_images.contains(&img_path) {
                                extracted_images.push(img_path);
                            }
                        }

                        for part in chunk.content.split("![") {
                            if let Some(end_idx) = part.find(')') {
                                let img_tag = &part[..end_idx];
                                if let Some(open_idx) = img_tag.find('(') {
                                    let img_file = &img_tag[open_idx + 1..];
                                    let lower = img_file.to_lowercase();
                                    if lower.ends_with(".png") || lower.ends_with(".jpg") || lower.ends_with(".jpeg") || lower.ends_with(".webp") {
                                        if !extracted_images.contains(&img_file.to_string()) {
                                            extracted_images.push(img_file.to_string());
                                        }
                                    }
                                }
                            }
                        }

                        all_retrieved.push(chunk);
                    }
                }
            }
        }

        all_retrieved.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_retrieved.truncate(req.top_k);

        // 3. Resolució: Extracció Exacta (Zero-LLM) vs Foto Finish (Conversacional)
        let resolution = if req.zero_llm {
            self.format_zero_llm_resolution(&req.query, &triage, &all_retrieved, &extracted_images)
        } else {
            self.format_conversational_resolution(&req.query, &triage, &all_retrieved, &extracted_images).await
        };

        let total_latency = start_time.elapsed().as_secs_f64() * 1000.0;

        EngineResponse {
            query: req.query.clone(),
            triage,
            retrieval: all_retrieved,
            images: extracted_images,
            resolution,
            zero_llm: req.zero_llm,
            total_latency_ms: total_latency,
        }
    }

    fn format_zero_llm_resolution(
        &self,
        _query: &str,
        triage: &TriageResult,
        chunks: &[RetrievedChunk],
        images: &[String],
    ) -> String {
        let mut out = String::new();

        out.push_str("### ⚡ PROTOCOL DE RESPOSTA IMMEDIATA (MODE ZERO-LLM / BÚNQUER)\n\n");
        out.push_str(&format!(
            "- **Categoria Triada**: `{}`\n- **Nivell d'Urgència**: `{}`\n- **Temps de Decisió**: `{:.2} ms`\n\n",
            triage.category, triage.urgency, triage.decision_latency_ms
        ));

        if chunks.is_empty() {
            out.push_str("⚠️ No s'han trobat fragments coincidents exactes a la biblioteca local.\n");
            return out;
        }

        out.push_str("#### 📋 Passos i Dades Literals Extretes dels Manuals:\n\n");
        for (i, c) in chunks.iter().enumerate() {
            out.push_str(&format!(
                "**{}. Font Oficial:** {}\n\n> {}\n\n",
                i + 1,
                c.citation,
                c.content.trim().replace('\n', "\n> ")
            ));
        }

        if !images.is_empty() {
            out.push_str("#### 🖼️ Esquemes i Diagrames Tècnics Extrets:\n\n");
            for img in images {
                out.push_str(&format!("![Esquema Oficial]({})\n\n", img));
            }
        }

        out.push_str("#### 📚 Referències Verificades:\n");
        for c in chunks {
            out.push_str(&format!("- {} (Puntuació semàntica: `{:.3}`)\n", c.citation, c.score));
        }

        out
    }

    async fn format_conversational_resolution(
        &self,
        query: &str,
        triage: &TriageResult,
        chunks: &[RetrievedChunk],
        images: &[String],
    ) -> String {
        let mut context_builder = String::new();
        for (i, c) in chunks.iter().enumerate() {
            context_builder.push_str(&format!(
                "--- FONT {} {} ---\n{}\n\n",
                i + 1, c.citation, c.content
            ));
        }

        let images_hint = if !images.is_empty() {
            format!(
                "Imatges disponibles per incorporar al text: {}\n",
                images.join(", ")
            )
        } else {
            "".to_string()
        };

        let system_prompt = format!(
            "Ets en Kev, l'assistent expert de camp i coneixement de la Biblioteca Digital SmartLocal.\n\
             Respon a la consulta de l'usuari amb to professional, fonamentat i clar en el mateix idioma que l'usuari.\n\
             NORMA SAGRADA: Cita SEMPRE el manual i número de pàgina al costat de cada instrucció crítica ex: [📖 Manual: ..., Pàg. X].\n\
             Si hi ha diagrames o imatges rellevants, inclou-les amb format Markdown ![Diagrama](nom_imatge).\n\
             Finalitza SEMPRE amb l'apartat '📚 Referències Consultades'.\n\n\
             CATEGORIA TRIATGE: {}\n\
             {}\n\
             === CONTEXT OFICIAL RECUPERAT ===\n\
             {}\n\
             ==================================",
            triage.category, images_hint, context_builder
        );

        if let Ok(res) = self.call_ollama_chat(&system_prompt, query).await {
            if !res.trim().is_empty() {
                return res;
            }
        }

        self.format_zero_llm_resolution(query, triage, chunks, images)
    }

    async fn call_ollama_chat(&self, system_prompt: &str, query: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let body = serde_json::json!({
            "model": self.generator_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": false,
            "options": {
                "temperature": 0.2,
                "num_predict": 3000
            }
        });

        let res = self.client
            .post(format!("{}/api/chat", self.ollama_url))
            .json(&body)
            .send()
            .await?;

        if res.status().is_success() {
            let json: serde_json::Value = res.json().await?;
            if let Some(msg) = json.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()) {
                return Ok(msg.to_string());
            }
        }

        Err("Ollama chat completion failed".into())
    }
}
