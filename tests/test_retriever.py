import tempfile
import shutil
from pathlib import Path
from PIL import Image

from smart_local.knowledge.store import KnowledgeStore
from smart_local.knowledge.retriever import MultimodalRetriever
from smart_local.knowledge.schema import Document, Chunk, ModalityType
from smart_local.models.embedding import EmbeddingGemma

def test_multimodal_retriever():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        rag_id = "multimodal_test"
        store = KnowledgeStore(temp_dir, rag_id)
        embedder = EmbeddingGemma()
        retriever = MultimodalRetriever(embedder=embedder)

        # 1. Text chunk
        text_content = "Manual d'arquitectura del sistema RAG intel·ligent."
        chunk_text = Chunk(
            id="c-txt",
            document_id="doc-txt",
            rag_id=rag_id,
            modality=ModalityType.TEXT,
            content=text_content,
            embedding=embedder.embed_text(text_content)
        )

        # 2. Image chunk
        img = Image.new("RGB", (32, 32), color="blue")
        img_path = store.assets_dir / "diagram.png"
        img.save(img_path)
        img_vec = embedder.embed_image(img_path)
        chunk_img = Chunk(
            id="c-img",
            document_id="doc-img",
            rag_id=rag_id,
            modality=ModalityType.IMAGE,
            content="[Diagrama d'arquitectura de memòria blava]",
            asset_rel_path="diagram.png",
            embedding=img_vec
        )

        store.add_chunks([chunk_text, chunk_img])

        # Retrieve via text query
        results = retriever.retrieve(store, query="Manual d'arquitectura", top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c-txt"

        # Retrieve via image query
        query_img = Image.new("RGB", (32, 32), color="blue")
        results_img = retriever.retrieve(store, query=query_img, top_k=2)
        assert len(results_img) == 2
        assert results_img[0].chunk_id == "c-img"
        assert results_img[0].asset_path is not None

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
