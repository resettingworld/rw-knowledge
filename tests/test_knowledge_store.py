import tempfile
import shutil
from pathlib import Path
from smart_local.knowledge.store import KnowledgeStore
from smart_local.knowledge.schema import Document, Chunk, ModalityType, KnowledgePackManifest
from smart_local.models.embedding import EmbeddingGemma

def test_knowledge_store_crud():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        rag_id = "test_store_rag"
        store = KnowledgeStore(temp_dir, rag_id)

        # 1. Add document
        doc = Document(
            id="doc-1",
            rag_id=rag_id,
            filename="sample.txt",
            file_type=".txt",
            file_size=1024,
            modality=ModalityType.TEXT,
            chunk_count=2
        )
        store.add_document(doc)

        # 2. Add chunks with embeddings
        embedder = EmbeddingGemma()
        c1_text = "Gemma 4 E4B és un model multimodal eficient per dispositius locals."
        c2_text = "EmbeddingGemma 2 ofereix un espai vectorial unificat de 768 dimensions."

        chunks = [
            Chunk(
                id="c-1",
                document_id="doc-1",
                rag_id=rag_id,
                modality=ModalityType.TEXT,
                content=c1_text,
                embedding=embedder.embed_text(c1_text)
            ),
            Chunk(
                id="c-2",
                document_id="doc-1",
                rag_id=rag_id,
                modality=ModalityType.TEXT,
                content=c2_text,
                embedding=embedder.embed_text(c2_text)
            )
        ]
        store.add_chunks(chunks)

        # 3. Check stats
        stats = store.get_stats()
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] == 2
        assert "text" in stats["modalities"]

        # 4. Search query
        query_vec = embedder.embed_text("informació sobre espai vectorial d'embeddings")
        results = store.search(query_vector=query_vec, top_k=2)
        assert len(results) == 2
        assert results[0].rag_id == rag_id
        assert isinstance(results[0].score, float)

        # 5. On-demand load & unload
        store.unload_from_memory()
        assert not store.is_loaded()
        assert store._vectors is None

        store.load_into_memory()
        assert store.is_loaded()
        assert store._vectors.shape == (2, 768)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
