from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time

class ModalityType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"

class LibraryDocument(BaseModel):
    """
    Standard digital library record preserving the original file.
    """
    id: str
    rag_id: str
    original_filename: str
    stored_filename: str
    file_type: str
    mime_type: str
    file_size_bytes: int
    sha256: str
    page_count: int = 1
    image_count: int = 0
    chunk_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

class Chunk(BaseModel):
    id: str
    document_id: str
    rag_id: str
    modality: ModalityType = ModalityType.TEXT
    content: str
    page_number: Optional[int] = None
    asset_rel_path: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

class Document(BaseModel):
    id: str
    rag_id: str
    filename: str
    file_type: str
    file_size: int
    modality: ModalityType
    chunk_count: int = 0
    page_count: int = 1
    image_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

class KnowledgePackManifest(BaseModel):
    rag_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = "smart-local"
    created_at: float = Field(default_factory=time.time)
    embedding_model: str = "embeddinggemma-2"
    generator_model: str = "gemma-4-e4b"
    total_chunks: int = 0
    total_documents: int = 0
    total_library_files: int = 0
    total_images_extracted: int = 0
    modalities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    rag_id: str
    modality: str
    content: str
    score: float
    source_path: str
    page_number: Optional[int] = None
    citation_label: Optional[str] = None
    asset_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
