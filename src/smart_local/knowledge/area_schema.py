from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
from pathlib import Path

class PortalSource(BaseModel):
    name: str = "InfoLibros"
    url: str
    scraped_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class CategoryMeta(BaseModel):
    slug: str
    name: str
    url: str

class CoverImageMeta(BaseModel):
    rel_path: str = "images/banner.png"
    prompt: Optional[str] = None
    generator: str = "banana-pi"
    width: Optional[int] = 1024
    height: Optional[int] = 576

class BookMeta(BaseModel):
    id: str
    title: str
    author: Optional[str] = "Dominio Público / InfoLibros"
    filename: str
    source_url: Optional[str] = None
    file_size_bytes: int = 0
    sha256: str
    page_count: int = 1
    key_topics: List[str] = Field(default_factory=list)
    images_extracted: List[str] = Field(default_factory=list)

class RagSpecification(BaseModel):
    rag_id: str
    embedding_model: str = "embeddinggemma-2"
    generator_model: str = "gemma-4-e4b"
    chunk_size: int = 500
    chunk_overlap: int = 50
    index_db: str = "index.db"
    total_documents: int = 0
    total_chunks: int = 0
    total_images_extracted: int = 0

class AreaStats(BaseModel):
    total_books: int = 0
    total_pages: int = 0
    total_size_mb: float = 0.0

class KnowledgeAreaMeta(BaseModel):
    schema_version: str = "1.0.0"
    id: str
    name: str
    description: str
    category: CategoryMeta
    domain: str
    language: str = "es"
    portal_source: PortalSource
    tags: List[str] = Field(default_factory=list)
    subtopics_covered: List[str] = Field(default_factory=list)
    cover_image: Optional[CoverImageMeta] = None
    books: List[BookMeta] = Field(default_factory=list)
    rag_spec: RagSpecification
    stats: AreaStats = Field(default_factory=AreaStats)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: Path) -> "KnowledgeAreaMeta":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

# Backward compatibility alias
KnowledgeCapsuleMeta = KnowledgeAreaMeta
