from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smart_local.config import settings
from smart_local.manager.registry import KnowledgeManager
from smart_local.pipeline.workers import BackgroundPipelineWorker
from smart_local.models.generator import Gemma4E4B
from smart_local.models.embedding import EmbeddingGemma
from smart_local.knowledge.retriever import MultimodalRetriever
from smart_local.server.routes_openai import router as openai_router
from smart_local.server.routes_rag import router as rag_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize singletons and thread workers
    settings.ensure_directories()
    km = KnowledgeManager()
    worker = BackgroundPipelineWorker(knowledge_manager=km)
    generator = Gemma4E4B()
    embedder = EmbeddingGemma()
    retriever = MultimodalRetriever(embedder=embedder)

    app.state.knowledge_manager = km
    app.state.worker = worker
    app.state.generator = generator
    app.state.embedder = embedder
    app.state.retriever = retriever

    yield

    # Shutdown: Stop worker threads cleanly
    worker._running = False

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Multimodal Knowledge RAG system with Gemma 4 E4B, EmbeddingGemma 2, On-Demand Loading & OpenAI-compatible SDK",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router)
app.include_router(rag_router)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.version,
        "generator_model": settings.generator_model,
        "embedding_model": settings.embedding_model
    }
