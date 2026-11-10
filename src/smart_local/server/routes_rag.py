import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from smart_local.config import settings

router = APIRouter(prefix="/v1", tags=["RAG Management"])

class CreateRAGRequest(BaseModel):
    rag_id: str
    name: Optional[str] = None
    description: Optional[str] = ""
    tags: Optional[List[str]] = Field(default_factory=list)

class IngestTextRequest(BaseModel):
    title: str
    content: str

class PackageRAGRequest(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = "1.0.0"
    description: Optional[str] = ""
    tags: Optional[List[str]] = Field(default_factory=list)

@router.get("/rags")
async def list_rags(request: Request):
    """Lists all available RAG spaces and their on-demand loaded status."""
    km = request.app.state.knowledge_manager
    return {"rags": km.list_rags()}

@router.post("/rags")
async def create_rag(req: CreateRAGRequest, request: Request):
    """Creates a new specific RAG knowledge base."""
    km = request.app.state.knowledge_manager
    store = km.create_rag(
        rag_id=req.rag_id,
        name=req.name,
        description=req.description,
        tags=req.tags
    )
    return {"status": "created", "rag_id": req.rag_id, "stats": store.get_stats()}

@router.get("/rags/{rag_id}")
async def get_rag_info(rag_id: str, request: Request):
    """Returns details and statistics for a specific RAG."""
    km = request.app.state.knowledge_manager
    store = km.get_store(rag_id, auto_load=False)
    if not store:
        raise HTTPException(status_code=404, detail=f"RAG '{rag_id}' not found")
    stats = store.get_stats()
    manifest = store.get_manifest()
    return {
        "rag_id": rag_id,
        "manifest": manifest.model_dump() if manifest else None,
        "stats": stats
    }

@router.delete("/rags/{rag_id}")
async def delete_rag(rag_id: str, request: Request):
    """Deletes a RAG knowledge base."""
    km = request.app.state.knowledge_manager
    success = km.delete_rag(rag_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"RAG '{rag_id}' not found")
    return {"status": "deleted", "rag_id": rag_id}

@router.post("/rags/{rag_id}/load")
async def load_rag(rag_id: str, request: Request):
    """Loads a RAG's embeddings and index into RAM on-demand."""
    km = request.app.state.knowledge_manager
    success = km.load_rag(rag_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"RAG '{rag_id}' not found")
    return {"status": "loaded", "rag_id": rag_id}

@router.post("/rags/{rag_id}/unload")
async def unload_rag(rag_id: str, request: Request):
    """Unloads a RAG from RAM to free memory."""
    km = request.app.state.knowledge_manager
    success = km.unload_rag(rag_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"RAG '{rag_id}' not found")
    return {"status": "unloaded", "rag_id": rag_id}

@router.post("/rags/{rag_id}/ingest/files")
async def ingest_files(rag_id: str, request: Request, files: List[UploadFile] = File(...)):
    """
    Uploads and enqueues multimodal files (PDFs, docs, images, audio) for background ingestion.
    Preserves original files in the digital library and extracts images from PDFs.
    """
    km = request.app.state.knowledge_manager
    worker = request.app.state.worker

    store = km.get_or_create_store(rag_id)

    saved_paths = []
    upload_batch_dir = settings.uploads_dir / rag_id
    upload_batch_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        file_path = upload_batch_dir / upload.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
        saved_paths.append(file_path)

    task_id = worker.submit_ingest_files(rag_id, saved_paths)
    return {
        "status": "queued",
        "task_id": task_id,
        "rag_id": rag_id,
        "files_count": len(saved_paths)
    }

@router.post("/rags/{rag_id}/ingest/text")
async def ingest_text(rag_id: str, req: IngestTextRequest, request: Request):
    """Directly ingests a text segment into a RAG."""
    km = request.app.state.knowledge_manager
    worker = request.app.state.worker

    store = km.get_or_create_store(rag_id)
    doc, chunks = worker.processor.process_raw_text(
        rag_id=rag_id,
        title=req.title,
        text=req.content
    )
    store.add_document(doc)
    store.add_chunks(chunks)

    return {
        "status": "ingested",
        "rag_id": rag_id,
        "document_id": doc.id,
        "chunks_count": len(chunks)
    }

@router.get("/rags/{rag_id}/library")
async def list_library_documents(rag_id: str, request: Request):
    """Lists all preserved pristine original documents in this RAG's digital library."""
    km = request.app.state.knowledge_manager
    store = km.get_store(rag_id, auto_load=False)
    if not store:
        raise HTTPException(status_code=404, detail=f"RAG '{rag_id}' not found")
    docs = store.list_library_documents()
    return {
        "rag_id": rag_id,
        "total_documents": len(docs),
        "documents": docs
    }

@router.get("/rags/{rag_id}/library/{doc_id}/download")
async def download_library_document(rag_id: str, doc_id: str, request: Request):
    """Serves/downloads the original preserved file (e.g. PDF)."""
    km = request.app.state.knowledge_manager
    store = km.get_store(rag_id, auto_load=False)
    if not store:
        raise HTTPException(status_code=404, detail=f"RAG '{rag_id}' not found")
    doc_info = store.get_library_document(doc_id)
    if not doc_info:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    file_path = Path(doc_info["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(
        path=str(file_path),
        filename=doc_info["original_filename"],
        media_type=doc_info.get("mime_type", "application/octet-stream")
    )

@router.delete("/rags/{rag_id}/library/{doc_id}")
async def delete_library_document(rag_id: str, doc_id: str, request: Request):
    """Deletes a document from the digital library and RAG vector store."""
    km = request.app.state.knowledge_manager
    store = km.get_store(rag_id, auto_load=False)
    if not store:
        raise HTTPException(status_code=404, detail=f"RAG '{rag_id}' not found")
    success = store.delete_library_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in RAG '{rag_id}'")
    return {"status": "deleted", "rag_id": rag_id, "document_id": doc_id, "stats": store.get_stats()}

@router.post("/rags/{rag_id}/library/upload")
async def upload_library_file(rag_id: str, request: Request, file: UploadFile = File(...)):
    """
    Directly uploads and indexes a new book/document (PDF, text, image) into the RAG's digital library.
    Extracts text per page, embedded images, vectorizes with EmbeddingGemma 2, and updates dimensions.
    """
    km = request.app.state.knowledge_manager
    worker = request.app.state.worker
    store = km.get_or_create_store(rag_id)

    upload_batch_dir = settings.uploads_dir / rag_id
    upload_batch_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = upload_batch_dir / file.filename

    with temp_file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = temp_file_path.suffix.lower()
    if ext == ".pdf":
        doc, lib_doc, chunks = worker.processor.process_pdf_file(
            rag_id=rag_id,
            file_path=temp_file_path,
            library_dir=store.library_dir,
            assets_dir=store.assets_dir
        )
        if lib_doc:
            store.add_library_document(lib_doc)
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        doc, lib_doc, chunks = worker.processor.process_image_file(
            rag_id=rag_id,
            file_path=temp_file_path,
            assets_dir=store.assets_dir,
            library_dir=store.library_dir
        )
        if lib_doc:
            store.add_library_document(lib_doc)
    else:
        doc, lib_doc, chunks = worker.processor.process_text_file(
            rag_id=rag_id,
            file_path=temp_file_path,
            library_dir=store.library_dir
        )
        if lib_doc:
            store.add_library_document(lib_doc)

    store.add_document(doc)
    store.add_chunks(chunks)

    # Clean up temp upload
    try:
        temp_file_path.unlink()
    except Exception:
        pass

    return {
        "status": "uploaded_and_indexed",
        "rag_id": rag_id,
        "document": {
            "id": doc.id,
            "filename": doc.filename,
            "page_count": doc.page_count,
            "image_count": doc.image_count,
            "chunk_count": len(chunks)
        },
        "stats": store.get_stats()
    }

@router.get("/rags/{rag_id}/assets/{asset_name}")
async def get_asset_file(rag_id: str, asset_name: str, request: Request):
    """Serves an extracted image or media asset from the RAG."""
    km = request.app.state.knowledge_manager
    store = km.get_store(rag_id, auto_load=False)
    if not store:
        raise HTTPException(status_code=404, detail=f"RAG '{rag_id}' not found")
    asset_path = store.assets_dir / asset_name
    if not asset_path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = "image/png"
    if asset_path.suffix.lower() in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif asset_path.suffix.lower() == ".webp":
        media_type = "image/webp"
    return FileResponse(path=str(asset_path), media_type=media_type)

@router.get("/rags/{rag_id}/tasks")
async def list_rag_tasks(rag_id: str, request: Request):
    """Lists background tasks (ingestion or packaging) for this RAG."""
    worker = request.app.state.worker
    return {"tasks": worker.list_tasks(rag_id=rag_id)}

@router.get("/rags/{rag_id}/tasks/{task_id}")
async def get_task_status(rag_id: str, task_id: str, request: Request):
    """Queries status and progress of a background task."""
    worker = request.app.state.worker
    task = worker.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.post("/rags/{rag_id}/package")
async def trigger_package(rag_id: str, req: PackageRAGRequest, request: Request):
    """
    Triggers background packaging of the RAG into a distributable .gpk bundle.
    Includes manifest, vector index, digital library, and multimodal assets.
    """
    km = request.app.state.knowledge_manager
    worker = request.app.state.worker

    store = km.get_store(rag_id, auto_load=False)
    if not store:
        raise HTTPException(status_code=404, detail=f"RAG '{rag_id}' not found")

    task_id = worker.submit_package(
        rag_id=rag_id,
        name=req.name,
        version=req.version or "1.0.0",
        description=req.description or "",
        tags=req.tags
    )

    return {
        "status": "queued",
        "task_id": task_id,
        "rag_id": rag_id,
        "message": "Empaquetament iniciat en el thread de fons"
    }

@router.get("/rags/{rag_id}/download")
async def download_package(rag_id: str, version: Optional[str] = None):
    """Downloads the packaged .gpk bundle for distribution."""
    matches = list(settings.packs_dir.glob(f"{rag_id}*.gpk"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"No package found for RAG '{rag_id}'. Run package first.")

    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    target_file = matches[0]

    return FileResponse(
        path=str(target_file),
        filename=target_file.name,
        media_type="application/gzip"
    )

@router.post("/rags/import")
async def import_package(request: Request, package: UploadFile = File(...), rag_id_override: Optional[str] = Form(None)):
    """
    Imports a distributable .gpk bundle into the active RAG registry.
    """
    km = request.app.state.knowledge_manager
    temp_pkg_path = settings.packs_dir / f"incoming_{package.filename}"
    with temp_pkg_path.open("wb") as buffer:
        shutil.copyfileobj(package.file, buffer)

    resolved_rag_id = km.packager.import_package(temp_pkg_path, rag_id_override=rag_id_override)
    store = km.get_or_create_store(resolved_rag_id)

    return {
        "status": "imported",
        "rag_id": resolved_rag_id,
        "stats": store.get_stats(),
        "manifest": store.get_manifest().model_dump() if store.get_manifest() else None
    }

@router.get("/packs")
async def list_packages(request: Request):
    """Lists all packaged bundles available in the system."""
    km = request.app.state.knowledge_manager
    return km.packager.list_packages()
