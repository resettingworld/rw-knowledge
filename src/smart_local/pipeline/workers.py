import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from smart_local.pipeline.processor import MultimodalProcessor
from smart_local.knowledge.packager import KnowledgePackager
from smart_local.knowledge.schema import KnowledgePackManifest

class TaskInfo:
    def __init__(self, task_id: str, task_type: str, rag_id: str):
        self.task_id = task_id
        self.task_type = task_type
        self.rag_id = rag_id
        self.status = "queued" # queued, running, completed, failed
        self.progress = 0.0
        self.message = "En cua d'espera"
        self.created_at = time.time()
        self.completed_at: Optional[float] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "rag_id": self.rag_id,
            "status": self.status,
            "progress": round(self.progress, 2),
            "message": self.message,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error
        }

class BackgroundPipelineWorker:
    """
    Dedicated background worker thread for multimodal knowledge ingestion and bundle packaging.
    """
    def __init__(self, knowledge_manager, packager: Optional[KnowledgePackager] = None):
        self.km = knowledge_manager
        self.packager = packager or KnowledgePackager()
        self.processor = MultimodalProcessor()

        self._queue: queue.Queue = queue.Queue()
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._running = True

        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="KnowledgeWorkerThread")
        self._worker_thread.start()

    def submit_ingest_files(self, rag_id: str, file_paths: List[Path]) -> str:
        task_id = str(uuid.uuid4())
        task = TaskInfo(task_id, "ingest_files", rag_id)
        with self._lock:
            self._tasks[task_id] = task

        self._queue.put(("ingest_files", task, file_paths))
        return task_id

    def submit_package(
        self,
        rag_id: str,
        name: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        task_id = str(uuid.uuid4())
        task = TaskInfo(task_id, "package_bundle", rag_id)
        with self._lock:
            self._tasks[task_id] = task

        self._queue.put(("package_bundle", task, {
            "name": name,
            "version": version,
            "description": description,
            "tags": tags
        }))
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None

    def list_tasks(self, rag_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            tasks = list(self._tasks.values())
            if rag_id:
                tasks = [t for t in tasks if t.rag_id == rag_id]
            return [t.to_dict() for t in sorted(tasks, key=lambda x: x.created_at, reverse=True)]

    def _run_loop(self):
        while self._running:
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            job_type, task, payload = job
            task.status = "running"
            task.message = f"Executant tasca {job_type}..."

            try:
                if job_type == "ingest_files":
                    self._execute_ingest(task, payload)
                elif job_type == "package_bundle":
                    self._execute_package(task, payload)
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                task.message = f"Error: {e}"
                task.completed_at = time.time()
            finally:
                self._queue.task_done()

    def _execute_ingest(self, task: TaskInfo, file_paths: List[Path]):
        store = self.km.get_or_create_store(task.rag_id)
        total = len(file_paths)
        total_chunks = 0
        total_docs = 0

        for i, fp in enumerate(file_paths):
            fp = Path(fp)
            if not fp.exists():
                continue

            ext = fp.suffix.lower()
            if ext == ".pdf":
                doc, lib_doc, chunks = self.processor.process_pdf_file(
                    rag_id=task.rag_id,
                    file_path=fp,
                    library_dir=store.library_dir,
                    assets_dir=store.assets_dir
                )
                if lib_doc:
                    store.add_library_document(lib_doc)
            elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                doc, lib_doc, chunks = self.processor.process_image_file(
                    rag_id=task.rag_id,
                    file_path=fp,
                    assets_dir=store.assets_dir,
                    library_dir=store.library_dir
                )
                if lib_doc:
                    store.add_library_document(lib_doc)
            elif ext in [".wav", ".mp3", ".ogg", ".m4a"]:
                doc, chunks = self.processor.process_audio_file(task.rag_id, fp, store.assets_dir)
            else:
                doc, lib_doc, chunks = self.processor.process_text_file(
                    rag_id=task.rag_id,
                    file_path=fp,
                    library_dir=store.library_dir
                )
                if lib_doc:
                    store.add_library_document(lib_doc)

            store.add_document(doc)
            store.add_chunks(chunks)

            total_docs += 1
            total_chunks += len(chunks)
            task.progress = (i + 1) / total
            task.message = f"Processat {i + 1}/{total}: {fp.name}"

        # Update manifest
        stats = store.get_stats()
        manifest = store.get_manifest() or KnowledgePackManifest(
            rag_id=task.rag_id,
            name=task.rag_id
        )
        manifest.total_chunks = stats["total_chunks"]
        manifest.total_documents = stats["total_documents"]
        manifest.modalities = stats["modalities"]
        store.save_manifest(manifest)

        task.status = "completed"
        task.progress = 1.0
        task.completed_at = time.time()
        task.message = f"Ingestió completada: {total_docs} documents, {total_chunks} chunks."
        task.result = {
            "processed_documents": total_docs,
            "processed_chunks": total_chunks,
            "stats": stats
        }

    def _execute_package(self, task: TaskInfo, params: Dict[str, Any]):
        task.message = "Generant paquet distributable .gpk..."
        task.progress = 0.3

        pkg_path = self.packager.package_rag(
            rag_id=task.rag_id,
            name=params.get("name"),
            version=params.get("version", "1.0.0"),
            description=params.get("description", ""),
            tags=params.get("tags")
        )

        task.progress = 1.0
        task.status = "completed"
        task.completed_at = time.time()
        task.message = f"Paquet generat correctament: {pkg_path.name}"
        task.result = {
            "package_file": pkg_path.name,
            "package_path": str(pkg_path),
            "size_bytes": pkg_path.stat().st_size
        }
