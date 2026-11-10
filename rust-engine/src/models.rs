use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagManifest {
    pub rag_id: String,
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub embedding_model: String,
    #[serde(default)]
    pub total_chunks: usize,
    #[serde(default)]
    pub total_documents: usize,
    #[serde(default)]
    pub total_images_extracted: usize,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredRag {
    pub rag_id: String,
    pub name: String,
    pub score: f32,
    pub matched_tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageResult {
    pub query: String,
    pub urgency: String,
    pub category: String,
    pub selected_rags: Vec<ScoredRag>,
    pub decision_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedChunk {
    pub chunk_id: String,
    pub rag_id: String,
    pub modality: String,
    pub content: String,
    pub page_number: Option<i64>,
    pub source_path: String,
    pub asset_rel_path: Option<String>,
    pub score: f32,
    pub citation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    pub target_rags: Option<Vec<String>>,
    #[serde(default)]
    pub zero_llm: bool,
    #[serde(default = "default_dim")]
    pub dim: usize,
}

fn default_top_k() -> usize {
    4
}

fn default_dim() -> usize {
    768
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineResponse {
    pub query: String,
    pub triage: TriageResult,
    pub retrieval: Vec<RetrievedChunk>,
    pub images: Vec<String>,
    pub resolution: String,
    pub zero_llm: bool,
    pub total_latency_ms: f64,
}
