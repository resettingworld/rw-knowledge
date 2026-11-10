import tempfile
import shutil
from pathlib import Path
import pymupdf
from PIL import Image

from smart_local.knowledge.store import KnowledgeStore
from smart_local.pipeline.processor import MultimodalProcessor
from smart_local.models.embedding import EmbeddingGemma

def test_pdf_multimodal_ingestion_and_library():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        rag_id = "test_pdf_rag"
        store = KnowledgeStore(temp_dir, rag_id)
        embedder = EmbeddingGemma()
        processor = MultimodalProcessor(embedder=embedder)

        # 1. Create a dummy synthetic PDF with text and an embedded image
        pdf_path = temp_dir / "manual_supervivencia_test.pdf"
        doc = pymupdf.open()
        page1 = doc.new_page()
        page1.insert_text((50, 50), "Capítol 1: Com purificar aigua en cas d'emergència. Cal bullir-la durant 3 minuts.")

        # Create a diagram image with visual features and insert into page 1
        img_temp = temp_dir / "temp_fire.png"
        from PIL import ImageDraw
        im = Image.new("RGB", (200, 200), color="white")
        draw = ImageDraw.Draw(im)
        draw.rectangle([20, 20, 180, 180], fill="orange", outline="black", width=3)
        draw.line([(20, 20), (180, 180)], fill="red", width=4)
        im.save(img_temp)
        page1.insert_image(pymupdf.Rect(50, 100, 250, 300), filename=str(img_temp))
        page1.insert_text((50, 320), "Figura 1.1: Esquema tèrmic per bullir aigua.")

        # Page 2
        page2 = doc.new_page()
        page2.insert_text((50, 50), "Capítol 2: Construcció d'un refugi tèrmic amb branques i fulles seques.")

        doc.save(str(pdf_path))
        doc.close()

        # 2. Process PDF with processor
        d, lib_doc, chunks = processor.process_pdf_file(
            rag_id=rag_id,
            file_path=pdf_path,
            library_dir=store.library_dir,
            assets_dir=store.assets_dir
        )

        assert lib_doc is not None
        assert lib_doc.page_count == 2
        assert lib_doc.image_count >= 1
        assert (store.library_dir / lib_doc.stored_filename).exists()

        store.add_document(d)
        store.add_library_document(lib_doc)
        store.add_chunks(chunks)

        # 3. Verify digital library retrieval
        lib_docs = store.list_library_documents()
        assert len(lib_docs) == 1
        assert lib_docs[0]["original_filename"] == "manual_supervivencia_test.pdf"
        assert lib_docs[0]["page_count"] == 2

        # 4. Search and verify citation label with page number
        q_vec = embedder.embed_text("com purificar aigua bullir")
        results = store.search(q_vec, top_k=2)
        assert len(results) > 0
        top_res = results[0]
        assert "manual_supervivencia_test.pdf" in top_res.source_path
        assert top_res.page_number == 1
        assert "Pàg. 1" in top_res.citation_label

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
