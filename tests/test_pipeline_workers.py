import tempfile
import shutil
import time
from pathlib import Path
from PIL import Image

from smart_local.manager.registry import KnowledgeManager
from smart_local.pipeline.workers import BackgroundPipelineWorker

def test_pipeline_worker_thread():
    temp_root = Path(tempfile.mkdtemp())
    try:
        rags_dir = temp_root / "rags"
        packs_dir = temp_root / "packs"
        km = KnowledgeManager(rags_dir=rags_dir, packs_dir=packs_dir)
        worker = BackgroundPipelineWorker(knowledge_manager=km, packager=km.packager)

        rag_id = "async-worker-rag"
        km.create_rag(rag_id, name="Async RAG")

        # Create temporary sample files
        docs_dir = temp_root / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        txt_file = docs_dir / "knowledge.txt"
        txt_file.write_text("Aquest és un document de prova per al thread de fons.", encoding="utf-8")

        img_file = docs_dir / "photo.png"
        Image.new("RGB", (40, 40), color="green").save(img_file)

        # 1. Submit background ingestion
        task_id = worker.submit_ingest_files(rag_id, [txt_file, img_file])
        assert task_id is not None

        # Wait for worker thread to finish task (up to 10s)
        completed = False
        for _ in range(50):
            task_info = worker.get_task(task_id)
            if task_info and task_info["status"] == "completed":
                completed = True
                break
            time.sleep(0.1)

        assert completed, f"Task did not complete in time: {worker.get_task(task_id)}"
        assert worker.get_task(task_id)["progress"] == 1.0

        store = km.get_store(rag_id)
        stats = store.get_stats()
        assert stats["total_documents"] == 2
        assert stats["total_chunks"] >= 2

        # 2. Submit background packaging
        pkg_task_id = worker.submit_package(rag_id, name="Async Pack", version="1.0.0")
        pkg_completed = False
        for _ in range(50):
            pkg_info = worker.get_task(pkg_task_id)
            if pkg_info and pkg_info["status"] == "completed":
                pkg_completed = True
                break
            time.sleep(0.1)

        assert pkg_completed
        assert (packs_dir / f"{rag_id}-v1.0.0.gpk").exists()

        # Stop worker thread
        worker._running = False

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
