import pytest
import numpy as np
from PIL import Image

from smart_local.models.embedding import EmbeddingGemma
from smart_local.models.generator import Gemma4E4B

def test_embedding_gemma_text():
    embedder = EmbeddingGemma()
    vec = embedder.embed_text("Aquest és un document sobre intel·ligència artificial i Gemma 4.")
    assert len(vec) == 768
    norm = np.linalg.norm(np.array(vec))
    assert abs(norm - 1.0) < 1e-4

def test_embedding_gemma_image():
    embedder = EmbeddingGemma()
    # Create simple 64x64 synthetic image
    img = Image.new("RGB", (64, 64), color="red")
    vec = embedder.embed_image(img)
    assert len(vec) == 768
    norm = np.linalg.norm(np.array(vec))
    assert abs(norm - 1.0) < 1e-4

def test_embedding_gemma_audio():
    embedder = EmbeddingGemma()
    dummy_audio_bytes = b"RIFF" + b"\x00" * 2000
    vec = embedder.embed_audio(dummy_audio_bytes)
    assert len(vec) == 768
    norm = np.linalg.norm(np.array(vec))
    assert abs(norm - 1.0) < 1e-4

@pytest.mark.asyncio
async def test_gemma_4_e4b_generator():
    generator = Gemma4E4B()
    messages = [{"role": "user", "content": "Explica'm el concepte de RAG"}]
    context_chunks = [
        {"rag_id": "test_rag", "source_path": "doc1.txt", "content": "RAG combina recuperació i generació.", "score": 0.88}
    ]

    resp = await generator.generate_response(messages, context_chunks=context_chunks)
    assert "content" in resp
    assert len(resp["content"]) > 0
    assert len(resp["citations"]) == 1

@pytest.mark.asyncio
async def test_gemma_4_e4b_stream():
    generator = Gemma4E4B()
    messages = [{"role": "user", "content": "Com funciona EmbeddingGemma?"}]
    context = [{"rag_id": "tech", "source_path": "embedding.txt", "content": "EmbeddingGemma és multimodal i fa servir 768 dimensions."}]

    tokens = []
    async for token in generator.stream_response(messages, context_chunks=context):
        tokens.append(token)

    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert len(full_text) > 0
