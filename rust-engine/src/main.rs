mod catalog;
mod embedding;
mod models;
mod retriever;
mod resolver;
mod triage;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use clap::{Parser, Subcommand};
use axum::{
    routing::{get, post},
    extract::State,
    Json, Router,
};
use catalog::RagCatalog;
use models::{EngineResponse, QueryRequest, RagManifest, TriageResult};
use resolver::DecisionResolver;
use triage::FastTriager;

#[derive(Parser)]
#[command(name = "smart-engine")]
#[command(about = "Ultra-low-power, high-performance RAG & Emergency Decision Engine in Rust")]
struct Cli {
    #[arg(long, default_value = "data/rags")]
    data_dir: PathBuf,

    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    #[arg(long, default_value = "gemma4:latest")]
    generator_model: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Lists all detected RAG knowledge bases
    ListRags,

    /// Runs ultra-fast Sistema 1 triage (<5ms) without LLM
    Triage {
        query: String,
    },

    /// Executes complete hybrid decision flow
    Query {
        query: String,

        #[arg(long)]
        zero_llm: bool,

        #[arg(long, default_value_t = 4)]
        top_k: usize,

        #[arg(long, default_value_t = 768)]
        dim: usize,

        #[arg(long)]
        target_rags: Option<Vec<String>>,
    },

    /// Starts high-performance Axum REST API server
    Serve {
        #[arg(long, default_value_t = 8801)]
        port: u16,
    },
}

#[derive(Clone)]
struct AppState {
    resolver: Arc<DecisionResolver>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let catalog = RagCatalog::load_from_dir(&cli.data_dir);
    let resolver = Arc::new(DecisionResolver::new(
        catalog.clone(),
        &cli.ollama_url,
        &cli.generator_model,
    ));

    match cli.command {
        Commands::ListRags => {
            println!("📚 RAGs Detectats a {:?}:", cli.data_dir);
            for rag in catalog.list_rags() {
                println!(
                    " - [{}] {} (Fragments: {}, Tags: {:?})",
                    rag.rag_id, rag.name, rag.total_chunks, rag.tags
                );
            }
        }
        Commands::Triage { query } => {
            let res = FastTriager::triage(&catalog, &query);
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
        Commands::Query {
            query,
            zero_llm,
            top_k,
            dim,
            target_rags,
        } => {
            let req = QueryRequest {
                query,
                top_k,
                target_rags,
                zero_llm,
                dim,
            };
            let resp = resolver.resolve(&req).await;
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }
        Commands::Serve { port } => {
            let state = AppState {
                resolver: resolver.clone(),
            };

            let app = Router::new()
                .route("/v1/rags", get(api_list_rags))
                .route("/v1/triage", post(api_triage))
                .route("/v1/query", post(api_query))
                .with_state(state);

            let addr = SocketAddr::from(([0, 0, 0, 0], port));
            println!("🚀 SmartEngine Rust en marxa a http://0.0.0.0:{}", port);
            println!("⚡ Capacitat: Triatge Sistema 1 sub-5ms + Mode Zero-LLM + Foto Finish opcional");

            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app).await?;
        }
    }

    Ok(())
}

async fn api_list_rags(State(state): State<AppState>) -> Json<Vec<RagManifest>> {
    Json(state.resolver.catalog.list_rags())
}

async fn api_triage(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Json<TriageResult> {
    let query = payload
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let res = FastTriager::triage(&state.resolver.catalog, query);
    Json(res)
}

async fn api_query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Json<EngineResponse> {
    let resp = state.resolver.resolve(&req).await;
    Json(resp)
}
