import tempfile
import shutil
from pathlib import Path
from smart_local.knowledge.store import KnowledgeStore
from smart_local.knowledge.packager import KnowledgePackager
from smart_local.knowledge.schema import Document, Chunk, ModalityType
from smart_local.models.embedding import EmbeddingGemma

def test_package_export_and_import():
    temp_root = Path(tempfile.mkdtemp())
    try:
        rags_dir = temp_root / "rags"
        packs_dir = temp_root / "packs"
        packager = KnowledgePackager(rags_dir=rags_dir, packs_dir=packs_dir)

        # 1. Create a source RAG
        rag_id = "ai-science"
        store_dir = rags_dir / rag_id
        store = KnowledgeStore(store_dir, rag_id)

        embedder = EmbeddingGemma()
        content = "Gemma 4 suporta context d'almenys 128k tokens."
        doc = Document(
            id="doc-ai-1",
            rag_id=rag_id,
            filename="gemma4.txt",
            file_type=".txt",
            file_size=len(content),
            modality=ModalityType.TEXT,
            chunk_count=1
        )
        chunk = Chunk(
            id="chk-ai-1",
            document_id="doc-ai-1",
            rag_id=rag_id,
            modality=ModalityType.TEXT,
            content=content,
            embedding=embedder.embed_text(content)
        )
        store.add_document(doc)
        store.add_chunks([chunk])

        # 2. Package the RAG
        pkg_file = packager.package_rag(
            rag_id=rag_id,
            name="Ciència d'IA",
            version="1.2.0",
            description="Paquet de prova",
            tags=["ia", "gemma4"]
        )
        assert pkg_file.exists()
        assert pkg_file.name.endswith(".gpk")

        # 3. Import the package under a new RAG ID
        imported_id = packager.import_package(pkg_file, rag_id_override="ai-science-imported")
        assert imported_id == "ai-science-imported"

        imported_store = KnowledgeStore(rags_dir / imported_id, imported_id)
        stats = imported_store.get_stats()
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] == 1

        # 4. Search imported store
        results = imported_store.search(embedder.embed_text("context window tokens"), top_k=1)
        assert len(results) == 1
        assert "128k tokens" in results[0].content

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
