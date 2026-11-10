import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from smart_local.knowledge.schema import Chunk, Document, LibraryDocument, RetrievalResult, KnowledgePackManifest

class KnowledgeStore:
    """
    Self-contained SQLite and Vector index store for a specific RAG knowledge base.
    Optimized for dynamic on-demand loading, digital library preservation, and fast vectorized retrieval.
    """
    def __init__(self, rag_dir: Path, rag_id: str):
        self.rag_dir = Path(rag_dir)
        self.rag_id = rag_id
        self.db_path = self.rag_dir / "index.db"
        self.assets_dir = self.rag_dir / "assets"
        self.library_dir = self.rag_dir / "library"
        self.manifest_path = self.rag_dir / "manifest.json"

        self.rag_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.library_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._is_loaded = False
        self._vectors: Optional[np.ndarray] = None
        self._chunk_ids: List[str] = []
        self._chunk_map: Dict[str, Dict[str, Any]] = {}

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Digital Library table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS library_documents (
                id TEXT PRIMARY KEY,
                rag_id TEXT,
                original_filename TEXT,
                stored_filename TEXT,
                file_type TEXT,
                mime_type TEXT,
                file_size_bytes INTEGER,
                sha256 TEXT,
                page_count INTEGER,
                image_count INTEGER,
                chunk_count INTEGER,
                metadata TEXT,
                created_at REAL
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                rag_id TEXT,
                filename TEXT,
                file_type TEXT,
                file_size INTEGER,
                modality TEXT,
                chunk_count INTEGER,
                metadata TEXT,
                created_at REAL
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                rag_id TEXT,
                modality TEXT,
                content TEXT,
                page_number INTEGER,
                asset_rel_path TEXT,
                embedding_blob BLOB,
                metadata TEXT,
                created_at REAL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
            """)

            # Check if page_number column exists in chunks (for backwards compatibility)
            cursor.execute("PRAGMA table_info(chunks)")
            cols = [r["name"] for r in cursor.fetchall()]
            if "page_number" not in cols:
                cursor.execute("ALTER TABLE chunks ADD COLUMN page_number INTEGER")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_rag ON chunks(rag_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_modality ON chunks(modality)")
            conn.commit()
            conn.close()

    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_into_memory(self) -> None:
        """Loads all vector embeddings and chunk metadata into memory for low-latency search."""
        with self._lock:
            if self._is_loaded:
                return

            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            SELECT c.id, c.document_id, c.rag_id, c.modality, c.content, c.page_number, c.asset_rel_path,
                   c.embedding_blob, c.metadata, d.filename
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.embedding_blob IS NOT NULL
            """)
            rows = cursor.fetchall()
            conn.close()

            chunk_ids = []
            vec_list = []
            chunk_map = {}

            for r in rows:
                cid = r["id"]
                blob = r["embedding_blob"]
                if blob:
                    vec = np.frombuffer(blob, dtype=np.float32)
                    chunk_ids.append(cid)
                    vec_list.append(vec)
                    chunk_map[cid] = {
                        "document_id": r["document_id"],
                        "rag_id": r["rag_id"],
                        "modality": r["modality"],
                        "content": r["content"],
                        "page_number": r["page_number"],
                        "asset_rel_path": r["asset_rel_path"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                        "source_path": r["filename"] or "unknown"
                    }

            if vec_list:
                self._vectors = np.vstack(vec_list)
            else:
                self._vectors = np.empty((0, 768), dtype=np.float32)

            self._chunk_ids = chunk_ids
            self._chunk_map = chunk_map
            self._is_loaded = True

    def unload_from_memory(self) -> None:
        """Frees memory buffers when RAG is not currently active."""
        with self._lock:
            self._vectors = None
            self._chunk_ids = []
            self._chunk_map = {}
            self._is_loaded = False

    def add_library_document(self, lib_doc: LibraryDocument) -> None:
        """Persists a pristine digital library document entry."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO library_documents (
                id, rag_id, original_filename, stored_filename, file_type,
                mime_type, file_size_bytes, sha256, page_count, image_count,
                chunk_count, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lib_doc.id, lib_doc.rag_id, lib_doc.original_filename, lib_doc.stored_filename,
                lib_doc.file_type, lib_doc.mime_type, lib_doc.file_size_bytes, lib_doc.sha256,
                lib_doc.page_count, lib_doc.image_count, lib_doc.chunk_count,
                json.dumps(lib_doc.metadata), lib_doc.created_at
            ))
            conn.commit()
            conn.close()

    def list_library_documents(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM library_documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "rag_id": r["rag_id"],
                "original_filename": r["original_filename"],
                "stored_filename": r["stored_filename"],
                "file_type": r["file_type"],
                "mime_type": r["mime_type"],
                "file_size_bytes": r["file_size_bytes"],
                "sha256": r["sha256"],
                "page_count": r["page_count"],
                "image_count": r["image_count"],
                "chunk_count": r["chunk_count"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
                "download_url": f"/v1/rags/{r['rag_id']}/library/{r['id']}/download"
            })
        return results

    def get_library_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM library_documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row["id"],
            "rag_id": row["rag_id"],
            "original_filename": row["original_filename"],
            "stored_filename": row["stored_filename"],
            "file_type": row["file_type"],
            "mime_type": row["mime_type"],
            "file_size_bytes": row["file_size_bytes"],
            "sha256": row["sha256"],
            "page_count": row["page_count"],
            "image_count": row["image_count"],
            "chunk_count": row["chunk_count"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "created_at": row["created_at"],
            "file_path": str(self.library_dir / row["stored_filename"])
        }

    def delete_library_document(self, doc_id: str) -> bool:
        """
        Deletes a document from the digital library and from the RAG vector index.
        Removes chunks, assets, and the original file, then reloads in-memory index if active.
        """
        with self._lock:
            doc_info = self.get_library_document(doc_id)
            if not doc_info:
                return False

            conn = self._get_connection()
            cursor = conn.cursor()

            # Find all image chunks to clean up extracted image assets
            cursor.execute("SELECT asset_rel_path FROM chunks WHERE document_id = ? AND asset_rel_path IS NOT NULL", (doc_id,))
            asset_rows = cursor.fetchall()
            for ar in asset_rows:
                asset_rel = ar["asset_rel_path"]
                if asset_rel:
                    asset_file = self.assets_dir / asset_rel
                    if asset_file.exists():
                        try:
                            asset_file.unlink()
                        except Exception:
                            pass

            # Delete chunks, documents and library entry
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            cursor.execute("DELETE FROM library_documents WHERE id = ?", (doc_id,))
            conn.commit()
            conn.close()

            # Remove stored original file
            stored_file = self.library_dir / doc_info["stored_filename"]
            if stored_file.exists():
                try:
                    stored_file.unlink()
                except Exception:
                    pass

            # Update in-memory vectors if currently loaded
            if self._is_loaded:
                self._is_loaded = False
                self.load_into_memory()

            # Update manifest
            manifest = self.get_manifest()
            if manifest:
                stats = self.get_stats()
                manifest.total_chunks = stats["total_chunks"]
                manifest.total_documents = stats["total_documents"]
                manifest.total_library_files = stats["total_library_files"]
                manifest.total_images_extracted = stats["total_images_extracted"]
                manifest.modalities = stats["modalities"]
                self.save_manifest(manifest)

            return True

    def add_document(self, doc: Document) -> None:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO documents (id, rag_id, filename, file_type, file_size, modality, chunk_count, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, doc.rag_id, doc.filename, doc.file_type, doc.file_size,
                doc.modality.value if hasattr(doc.modality, "value") else str(doc.modality),
                doc.chunk_count, json.dumps(doc.metadata), doc.created_at
            ))
            conn.commit()
            conn.close()

    def add_chunks(self, chunks: List[Chunk]) -> None:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            rows = []
            for c in chunks:
                blob = None
                if c.embedding:
                    blob = np.array(c.embedding, dtype=np.float32).tobytes()
                rows.append((
                    c.id, c.document_id, c.rag_id,
                    c.modality.value if hasattr(c.modality, "value") else str(c.modality),
                    c.content, c.page_number, c.asset_rel_path, blob, json.dumps(c.metadata), c.created_at
                ))

            cursor.executemany("""
            INSERT OR REPLACE INTO chunks (id, document_id, rag_id, modality, content, page_number, asset_rel_path, embedding_blob, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            conn.close()

            # If already loaded, reload memory buffers
            if self._is_loaded:
                self._is_loaded = False
                self.load_into_memory()

    def search(
        self,
        query_vector: List[float],
        query_text: Optional[str] = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
        modality: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Performs hybrid vector cosine similarity and keyword search across chunks.
        Auto-loads into memory if not loaded.
        """
        if not self._is_loaded:
            self.load_into_memory()

        if self._vectors is None or len(self._chunk_ids) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(self._vectors, q_vec)

        # Parse query keywords if text is provided
        stop_words = {
            "el", "la", "los", "las", "un", "una", "de", "del", "en", "i", "y", "a",
            "com", "como", "que", "què", "es", "és", "tinc", "tengo", "puc", "puedo",
            "estic", "estoy", "per", "por", "si", "amb", "con"
        }
        keywords = []
        if query_text:
            cleaned = query_text.lower().replace("?", " ").replace("!", " ").replace(".", " ").replace(",", " ")
            keywords = [w for w in cleaned.split() if len(w) > 2 and w not in stop_words]

        valid_indices = range(len(self._chunk_ids))
        if modality:
            valid_indices = [
                i for i in valid_indices
                if self._chunk_map[self._chunk_ids[i]]["modality"] == modality
            ]

        if not valid_indices:
            return []

        scored_pairs = []
        for i in valid_indices:
            cid = self._chunk_ids[i]
            info = self._chunk_map[cid]
            v_score = float(scores[i])

            if keywords:
                content_lower = info["content"].lower()
                source_lower = info["source_path"].lower()

                # Calculate keyword match ratio
                kw_hits = sum(1 for kw in keywords if kw in content_lower or kw in source_lower)
                kw_score = min(1.0, kw_hits / max(1, len(keywords)))

                # Domain entity bonus: e.g. both selva and agua present
                if "selva" in keywords and "selva" in (content_lower + source_lower):
                    kw_score += 0.3
                if "agua" in keywords and "agua" in content_lower:
                    kw_score += 0.3
                if "fuego" in keywords and "fuego" in content_lower:
                    kw_score += 0.3

                final_score = 0.35 * v_score + 0.65 * kw_score
            else:
                final_score = v_score

            if min_score is None or final_score >= min_score:
                scored_pairs.append((i, final_score))

        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = scored_pairs[:top_k]

        results = []
        for idx, score in top_pairs:
            cid = self._chunk_ids[idx]
            info = self._chunk_map[cid]
            asset_full_path = None
            if info.get("asset_rel_path"):
                asset_full_path = str(self.assets_dir / info["asset_rel_path"])

            # Format standard citation label
            page_str = f", Pàg. {info['page_number']}" if info.get("page_number") else ""
            if info["modality"] == "image":
                citation = f"[🖼️ Imatge: {info.get('asset_rel_path', 'fig')} | 📖 Manual: {info['source_path']}{page_str}]"
            else:
                citation = f"[📖 Manual: {info['source_path']}{page_str}]"

            results.append(RetrievalResult(
                chunk_id=cid,
                document_id=info["document_id"],
                rag_id=info["rag_id"],
                modality=info["modality"],
                content=info["content"],
                score=round(score, 4),
                source_path=info["source_path"],
                page_number=info.get("page_number"),
                citation_label=citation,
                asset_path=asset_full_path,
                metadata=info["metadata"]
            ))

        return results

    def get_manifest(self) -> Optional[KnowledgePackManifest]:
        if not self.manifest_path.exists():
            return None
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return KnowledgePackManifest(**data)
        except Exception:
            return None

    def save_manifest(self, manifest: KnowledgePackManifest) -> None:
        self.manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM library_documents")
        total_lib_docs = cursor.fetchone()[0]
        cursor.execute("SELECT DISTINCT modality FROM chunks")
        modalities = [r[0] for r in cursor.fetchall() if r[0]]
        conn.close()

        # Count extracted images
        total_images = len(list(self.assets_dir.glob("*.png"))) + len(list(self.assets_dir.glob("*.jpg")))

        return {
            "rag_id": self.rag_id,
            "is_loaded": self._is_loaded,
            "total_documents": total_docs,
            "total_library_files": total_lib_docs,
            "total_chunks": total_chunks,
            "total_images_extracted": total_images,
            "modalities": modalities,
            "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
        }
