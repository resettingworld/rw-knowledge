"""
Knowledge Capsule Packager
Builds standard, distributable Knowledge Package ZIPs from (Metadata + PDFs).
Includes:
- Versioned metadata.json (validated by JSON Schema)
- Pre-built RAG index.db (vectorized with EmbeddingGemma 2)
- Original digital library books in books/
- Extracted key figures and cover banner in images/ (ready for chat serving)
"""

import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from smart_local.config import settings
from smart_local.knowledge.area_schema import KnowledgeAreaMeta, BookMeta
from smart_local.knowledge.schema import KnowledgePackManifest, ModalityType
from smart_local.knowledge.store import KnowledgeStore
from smart_local.pipeline.processor import MultimodalProcessor

class KnowledgeCapsulePackager:
    def __init__(
        self,
        packs_dir: Path = settings.packs_dir,
        processor: Optional[MultimodalProcessor] = None
    ):
        self.packs_dir = Path(packs_dir)
        self.packs_dir.mkdir(parents=True, exist_ok=True)
        self.processor = processor or MultimodalProcessor()

    def build_package(
        self,
        metadata_source: Union[str, Path, KnowledgeAreaMeta, dict],
        pdfs_source_dir: Union[str, Path],
        output_zip: Optional[Union[str, Path]] = None,
        cover_image_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Creates a standardized .zip knowledge package from:
        Input: (metadata + pdfs)
        Output: <rag_id>-v<version>.zip
        """
        # 1. Parse & validate metadata
        if isinstance(metadata_source, KnowledgeAreaMeta):
            meta = metadata_source
        elif isinstance(metadata_source, dict):
            meta = KnowledgeAreaMeta(**metadata_source)
        else:
            meta = KnowledgeAreaMeta.load_json(Path(metadata_source))

        pdfs_dir = Path(pdfs_source_dir)
        if not pdfs_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdfs_dir}")

        pdf_files = sorted(list(pdfs_dir.glob("*.pdf")) + list(pdfs_dir.glob("*.PDF")))
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {pdfs_dir}")

        rag_id = meta.id or meta.rag_spec.rag_id
        version = meta.schema_version or "1.0.0"

        # Determine output zip path
        if output_zip:
            output_zip_path = Path(output_zip)
        else:
            output_zip_path = self.packs_dir / f"{rag_id}-v{version}.zip"

        output_zip_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Prepare staging area
        stage_dir = self.packs_dir / f"_staging_{rag_id}_{int(time.time() * 1000)}"
        books_stage = stage_dir / "books"
        images_stage = stage_dir / "images"
        books_stage.mkdir(parents=True, exist_ok=True)
        images_stage.mkdir(parents=True, exist_ok=True)

        try:
            # 3. Copy cover banner and covers directory if available
            if cover_image_path and Path(cover_image_path).exists():
                shutil.copy2(cover_image_path, images_stage / "banner.png")
            elif (pdfs_dir.parent / "images" / "banner.png").exists():
                shutil.copy2(pdfs_dir.parent / "images" / "banner.png", images_stage / "banner.png")

            # Copy catalog.json if available in pdfs_dir or parent dir
            if (pdfs_dir / "catalog.json").exists():
                shutil.copy2(pdfs_dir / "catalog.json", stage_dir / "catalog.json")
            elif (pdfs_dir.parent / "catalog.json").exists():
                shutil.copy2(pdfs_dir.parent / "catalog.json", stage_dir / "catalog.json")

            # Copy images/covers if available
            if (pdfs_dir / "images" / "covers").exists():
                shutil.copytree(pdfs_dir / "images" / "covers", images_stage / "covers", dirs_exist_ok=True)
            elif (pdfs_dir.parent / "images" / "covers").exists():
                shutil.copytree(pdfs_dir.parent / "images" / "covers", images_stage / "covers", dirs_exist_ok=True)
            elif (pdfs_dir / "covers").exists():
                shutil.copytree(pdfs_dir / "covers", images_stage / "covers", dirs_exist_ok=True)
            elif (pdfs_dir.parent / "covers").exists():
                shutil.copytree(pdfs_dir.parent / "covers", images_stage / "covers", dirs_exist_ok=True)

            # 4. Initialize SQLite knowledge store in stage
            store = KnowledgeStore(rag_dir=stage_dir, rag_id=rag_id)

            total_chunks = 0
            total_extracted_images = 0
            total_pages = 0
            processed_books_meta: List[BookMeta] = []

            # 5. Process each PDF file with MultimodalProcessor
            for idx, pdf_path in enumerate(pdf_files):
                doc, lib_doc, chunks = self.processor.process_pdf_file(
                    rag_id=rag_id,
                    file_path=pdf_path,
                    library_dir=books_stage,
                    assets_dir=images_stage,
                    chunk_size=meta.rag_spec.chunk_size,
                    chunk_overlap=meta.rag_spec.chunk_overlap,
                    extract_images=True,
                )

                # Persist into SQLite index
                store.add_document(doc)
                store.add_library_document(lib_doc)
                store.add_chunks(chunks)

                # Collect extracted images for this book
                book_extracted_images = [
                    c.asset_rel_path for c in chunks
                    if c.modality == ModalityType.IMAGE and c.asset_rel_path
                ]

                # Update stats
                total_chunks += len(chunks)
                total_extracted_images += len(book_extracted_images)
                total_pages += doc.page_count

                # Create or update BookMeta
                with open(pdf_path, "rb") as f_pdf:
                    sha256 = hashlib.sha256(f_pdf.read()).hexdigest()

                processed_books_meta.append(BookMeta(
                    id=doc.id,
                    title=pdf_path.stem.replace("-", " ").title(),
                    author="InfoLibros / Dominio Público",
                    filename=pdf_path.name,
                    source_url=None,
                    file_size_bytes=pdf_path.stat().st_size,
                    sha256=sha256,
                    page_count=doc.page_count,
                    key_topics=[meta.name],
                    images_extracted=book_extracted_images,
                ))

            # 6. Update metadata with actual extracted counts
            meta.books = processed_books_meta
            meta.rag_spec.total_documents = len(processed_books_meta)
            meta.rag_spec.total_chunks = total_chunks
            meta.rag_spec.total_images_extracted = total_extracted_images
            meta.stats.total_books = len(processed_books_meta)
            meta.stats.total_pages = total_pages
            total_mb = sum(b.file_size_bytes for b in processed_books_meta) / (1024 * 1024)
            meta.stats.total_size_mb = round(total_mb, 2)
            meta.updated_at = datetime.now(timezone.utc).isoformat()

            # Save updated metadata.json
            meta_json_path = stage_dir / "metadata.json"
            meta.save_json(meta_json_path)

            # 7. Write standard SmartLocal manifest.json for direct engine integration
            manifest = KnowledgePackManifest(
                rag_id=rag_id,
                name=meta.name,
                description=meta.description,
                version=version,
                author=f"SmartLocal / {meta.portal_source.name}",
                created_at=time.time(),
                embedding_model=meta.rag_spec.embedding_model,
                generator_model=meta.rag_spec.generator_model,
                total_chunks=total_chunks,
                total_documents=len(processed_books_meta),
                total_library_files=len(processed_books_meta),
                total_images_extracted=total_extracted_images,
                modalities=["text", "image"],
                tags=meta.tags,
            )
            store.save_manifest(manifest)

            # Clean temporary SQLite locks
            del store

            # 8. Compress staging folder into standard .zip package
            with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Add root files
                for f in ["metadata.json", "manifest.json", "index.db", "catalog.json"]:
                    p = stage_dir / f
                    if p.exists():
                        zf.write(p, arcname=f)

                # Add books/
                for p in books_stage.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(books_stage)
                        zf.write(p, arcname=f"books/{rel}")

                # Add images/ (including covers/)
                for p in images_stage.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(images_stage)
                        zf.write(p, arcname=f"images/{rel}")

            return output_zip_path

        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    def import_zip_package(
        self,
        zip_path: Union[str, Path],
        rags_dir: Path = settings.rags_dir,
        rag_id_override: Optional[str] = None
    ) -> str:
        """
        Unpacks and mounts a .zip knowledge package directly into the live SmartLocal RAGs directory.
        Returns the activated rag_id.
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Package zip not found: {zip_path}")

        temp_extract = self.packs_dir / f"_unzip_{int(time.time() * 1000)}"
        temp_extract.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(temp_extract)

            # Determine rag_id
            manifest_file = temp_extract / "manifest.json"
            meta_file = temp_extract / "metadata.json"

            target_rag_id = rag_id_override
            if not target_rag_id and manifest_file.exists():
                m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                target_rag_id = m_data.get("rag_id")
            if not target_rag_id and meta_file.exists():
                meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                target_rag_id = meta_data.get("id")
            if not target_rag_id:
                target_rag_id = zip_path.stem.split("-v")[0]

            rag_dest = rags_dir / target_rag_id
            if rag_dest.exists():
                shutil.rmtree(rag_dest)
            rag_dest.mkdir(parents=True, exist_ok=True)

            # Copy root files
            for fname in ["metadata.json", "manifest.json", "index.db"]:
                src_f = temp_extract / fname
                if src_f.exists():
                    shutil.copy2(src_f, rag_dest / fname)

            # Map books/ to library/ (and also preserve books/)
            books_src = temp_extract / "books"
            if books_src.exists():
                shutil.copytree(books_src, rag_dest / "books", dirs_exist_ok=True)
                shutil.copytree(books_src, rag_dest / "library", dirs_exist_ok=True)

            # Map images/ to assets/ (and also preserve images/)
            images_src = temp_extract / "images"
            if images_src.exists():
                shutil.copytree(images_src, rag_dest / "images", dirs_exist_ok=True)
                shutil.copytree(images_src, rag_dest / "assets", dirs_exist_ok=True)

            return target_rag_id

        finally:
            shutil.rmtree(temp_extract, ignore_errors=True)
