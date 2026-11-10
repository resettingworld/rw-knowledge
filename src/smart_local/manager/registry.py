import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Any

from smart_local.config import settings
from smart_local.knowledge.store import KnowledgeStore
from smart_local.knowledge.schema import KnowledgePackManifest
from smart_local.knowledge.packager import KnowledgePackager

class KnowledgeManager:
    """
    Dynamic on-demand RAG loader and lifecycle manager.
    Keeps an LRU in-memory cache of loaded vector stores, allowing dynamic loading/unloading
    of embeddings without restarting the server.
    """
    def __init__(
        self,
        rags_dir: Path = settings.rags_dir,
        packs_dir: Path = settings.packs_dir,
        max_loaded: int = settings.max_cached_rags
    ):
        self.rags_dir = Path(rags_dir)
        self.packs_dir = Path(packs_dir)
        self.max_loaded = max_loaded
        self.packager = KnowledgePackager(rags_dir, packs_dir)

        self._lock = threading.RLock()
        # Active stores on disk
        self._stores: Dict[str, KnowledgeStore] = {}
        # LRU tracker of currently loaded stores (rag_id -> True)
        self._loaded_lru: OrderedDict[str, bool] = OrderedDict()

        self._discover_existing_rags()

    def _discover_existing_rags(self) -> None:
        if not self.rags_dir.exists():
            return
        for item in self.rags_dir.iterdir():
            if item.is_dir() and (item / "index.db").exists():
                rag_id = item.name
                self._stores[rag_id] = KnowledgeStore(item, rag_id)

    def create_rag(
        self,
        rag_id: str,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> KnowledgeStore:
        """Creates a new specific RAG knowledge base."""
        with self._lock:
            rag_dir = self.rags_dir / rag_id
            rag_dir.mkdir(parents=True, exist_ok=True)
            store = KnowledgeStore(rag_dir, rag_id)
            manifest = KnowledgePackManifest(
                rag_id=rag_id,
                name=name or rag_id,
                description=description,
                tags=tags or [],
                embedding_model=settings.embedding_model,
                generator_model=settings.generator_model
            )
            store.save_manifest(manifest)
            self._stores[rag_id] = store
            return store

    def get_or_create_store(self, rag_id: str) -> KnowledgeStore:
        with self._lock:
            if rag_id not in self._stores:
                rag_dir = self.rags_dir / rag_id
                self._stores[rag_id] = KnowledgeStore(rag_dir, rag_id)
            return self._stores[rag_id]

    def get_store(self, rag_id: str, auto_load: bool = True) -> Optional[KnowledgeStore]:
        """
        Retrieves a store. If auto_load=True, dynamically mounts/loads embeddings into memory on-demand.
        """
        with self._lock:
            store = self._stores.get(rag_id)
            if not store:
                rag_dir = self.rags_dir / rag_id
                if rag_dir.exists() and (rag_dir / "index.db").exists():
                    store = KnowledgeStore(rag_dir, rag_id)
                    self._stores[rag_id] = store
                else:
                    return None

            if auto_load and not store.is_loaded():
                self._load_store_internal(rag_id, store)
            elif store.is_loaded():
                # Mark as recently used in LRU
                self._loaded_lru.move_to_end(rag_id)

            return store

    def load_rag(self, rag_id: str) -> bool:
        """Explicitly loads a RAG's embeddings into RAM on demand."""
        with self._lock:
            store = self._stores.get(rag_id)
            if not store:
                rag_dir = self.rags_dir / rag_id
                if rag_dir.exists() and (rag_dir / "index.db").exists():
                    store = KnowledgeStore(rag_dir, rag_id)
                    self._stores[rag_id] = store
                else:
                    return False

            self._load_store_internal(rag_id, store)
            return True

    def _load_store_internal(self, rag_id: str, store: KnowledgeStore) -> None:
        """Internal helper to load store and enforce LRU cache budget."""
        # Evict LRU if at max capacity
        while len(self._loaded_lru) >= self.max_loaded and rag_id not in self._loaded_lru:
            oldest_id, _ = self._loaded_lru.popitem(last=False)
            old_store = self._stores.get(oldest_id)
            if old_store:
                old_store.unload_from_memory()

        store.load_into_memory()
        self._loaded_lru[rag_id] = True
        self._loaded_lru.move_to_end(rag_id)

    def unload_rag(self, rag_id: str) -> bool:
        """Unloads a RAG's embeddings from RAM to free memory."""
        with self._lock:
            store = self._stores.get(rag_id)
            if not store:
                return False

            store.unload_from_memory()
            if rag_id in self._loaded_lru:
                del self._loaded_lru[rag_id]
            return True

    def delete_rag(self, rag_id: str) -> bool:
        """Deletes a RAG from disk and cache."""
        with self._lock:
            self.unload_rag(rag_id)
            if rag_id in self._stores:
                del self._stores[rag_id]

            rag_dir = self.rags_dir / rag_id
            if rag_dir.exists():
                shutil.rmtree(rag_dir)
            return True

    def list_rags(self) -> List[Dict[str, Any]]:
        """Lists all RAGs with their loaded status, document counts, and metadata."""
        with self._lock:
            results = []
            for item in self.rags_dir.iterdir():
                if item.is_dir() and (item / "index.db").exists():
                    rag_id = item.name
                    store = self._stores.get(rag_id)
                    if not store:
                        store = KnowledgeStore(item, rag_id)
                        self._stores[rag_id] = store

                    stats = store.get_stats()
                    manifest = store.get_manifest()
                    results.append({
                        "rag_id": rag_id,
                        "name": manifest.name if manifest else rag_id,
                        "description": manifest.description if manifest else "",
                        "version": manifest.version if manifest else "1.0.0",
                        "is_loaded": store.is_loaded(),
                        "total_documents": stats["total_documents"],
                        "total_chunks": stats["total_chunks"],
                        "modalities": stats["modalities"],
                        "tags": manifest.tags if manifest else []
                    })
            return sorted(results, key=lambda x: x["rag_id"])
