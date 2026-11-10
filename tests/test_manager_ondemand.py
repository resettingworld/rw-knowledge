import tempfile
import shutil
from pathlib import Path

from smart_local.manager.registry import KnowledgeManager
from smart_local.knowledge.schema import Chunk, ModalityType

def test_knowledge_manager_on_demand_and_lru():
    temp_root = Path(tempfile.mkdtemp())
    try:
        rags_dir = temp_root / "rags"
        packs_dir = temp_root / "packs"
        # Set max_loaded = 2 to test LRU eviction
        km = KnowledgeManager(rags_dir=rags_dir, packs_dir=packs_dir, max_loaded=2)

        # Create 3 RAGs
        s1 = km.create_rag("rag-1", name="RAG One")
        s2 = km.create_rag("rag-2", name="RAG Two")
        s3 = km.create_rag("rag-3", name="RAG Three")

        # Initially, none are loaded in memory
        assert not s1.is_loaded()
        assert not s2.is_loaded()
        assert not s3.is_loaded()

        # Load rag-1 and rag-2
        km.load_rag("rag-1")
        km.load_rag("rag-2")
        assert km.get_store("rag-1", auto_load=False).is_loaded()
        assert km.get_store("rag-2", auto_load=False).is_loaded()
        assert len(km._loaded_lru) == 2

        # Loading rag-3 should evict the oldest (rag-1)
        km.load_rag("rag-3")
        assert not km.get_store("rag-1", auto_load=False).is_loaded()
        assert km.get_store("rag-2", auto_load=False).is_loaded()
        assert km.get_store("rag-3", auto_load=False).is_loaded()

        # Querying rag-1 with auto_load=True should dynamically load it and evict rag-2
        store_1 = km.get_store("rag-1", auto_load=True)
        assert store_1.is_loaded()
        assert not km.get_store("rag-2", auto_load=False).is_loaded()
        assert km.get_store("rag-3", auto_load=False).is_loaded()

        # Explicit unload
        km.unload_rag("rag-1")
        assert not km.get_store("rag-1", auto_load=False).is_loaded()

        # Delete rag-3
        km.delete_rag("rag-3")
        assert km.get_store("rag-3") is None

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
