import hashlib
import io
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple, Optional
import pymupdf
from PIL import Image

from smart_local.knowledge.schema import Chunk, Document, LibraryDocument, ModalityType
from smart_local.models.embedding import EmbeddingGemma
from smart_local.pipeline.image_analyzer import ImageConceptAnalyzer

class MultimodalProcessor:
    """
    Parses and processes multimodal content into vectorized Chunks with EmbeddingGemma 2.
    Supports PDF document extraction with page tracking, image extraction, audio, and code.
    Maintains digital library preservation of pristine original files.
    """
    def __init__(self, embedder: Optional[EmbeddingGemma] = None, image_analyzer: Optional[ImageConceptAnalyzer] = None):
        self.embedder = embedder or EmbeddingGemma()
        self.image_analyzer = image_analyzer or ImageConceptAnalyzer()

    def _calc_sha256(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def process_pdf_file(
        self,
        rag_id: str,
        file_path: Path,
        library_dir: Path,
        assets_dir: Path,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        extract_images: bool = True
    ) -> Tuple[Document, LibraryDocument, List[Chunk]]:
        """
        Processes a PDF document:
        1. Archives the original PDF into library_dir.
        2. Extracts text page by page with explicit page numbering.
        3. Extracts embedded images and vectorizes them with EmbeddingGemma 2 into assets_dir.
        """
        file_path = Path(file_path)
        doc_id = str(uuid.uuid4())
        library_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        # 1. Archive pristine original into digital library
        stored_filename = f"{doc_id}_{file_path.name}"
        stored_lib_path = library_dir / stored_filename
        shutil.copy2(file_path, stored_lib_path)
        sha256_hash = self._calc_sha256(file_path)
        file_size = file_path.stat().st_size

        pdf_doc = pymupdf.open(str(file_path))
        page_count = len(pdf_doc)

        chunks: List[Chunk] = []
        extracted_images_count = 0

        for page_idx in range(page_count):
            page_num = page_idx + 1
            page = pdf_doc[page_idx]

            # Extract page text
            page_text = page.get_text() or ""
            page_text = page_text.strip()

            if page_text:
                page_chunks = self._chunk_text(page_text, chunk_size, chunk_overlap)
                for c_idx, seg in enumerate(page_chunks):
                    emb = self.embedder.embed_text(seg)
                    chunks.append(Chunk(
                        id=f"{doc_id}-p{page_num}-chk{c_idx}",
                        document_id=doc_id,
                        rag_id=rag_id,
                        modality=ModalityType.TEXT,
                        content=seg,
                        page_number=page_num,
                        embedding=emb,
                        metadata={
                            "source_file": file_path.name,
                            "page_number": page_num,
                            "chunk_index": c_idx,
                            "char_count": len(seg)
                        }
                    ))

            # Extract embedded images on this page
            if extract_images:
                try:
                    img_list = page.get_images(full=True)
                    for img_idx, img_info in enumerate(img_list):
                        xref = img_info[0]
                        base_img = pdf_doc.extract_image(xref)
                        w = base_img.get("width", 0)
                        h = base_img.get("height", 0)

                        # Filter out tiny icon/bullet graphics (less than 100x100)
                        if w < 100 or h < 100:
                            continue

                        img_bytes = base_img.get("image")
                        img_ext = base_img.get("ext", "png").lower()
                        if not img_bytes:
                            continue

                        img_filename = f"img_{doc_id}_p{page_num}_{img_idx}.{img_ext}"
                        img_path = assets_dir / img_filename
                        with open(img_path, "wb") as f_img:
                            f_img.write(img_bytes)

                        # Evaluate conceptual relevance with Multimodal/LLM analyzer
                        eval_res = self.image_analyzer.evaluate_image(
                            image_path=img_path,
                            page_text=page_text,
                            doc_title=file_path.stem.replace("-", " ").title(),
                            page_num=page_num,
                            width=w,
                            height=h,
                        )

                        if not eval_res.is_relevant:
                            img_path.unlink(missing_ok=True)
                            continue

                        # Generate multimodal image embedding
                        img_emb = self.embedder.embed_image(img_path)
                        extracted_images_count += 1

                        content_repr = (
                            f"{eval_res.markdown_tag}\n"
                            f"**Figura/Esquema Clau**: {eval_res.concept_name}\n"
                            f"**Descripció Conceptual**: {eval_res.caption}\n"
                            f"**Context del Llibre (Pàg. {page_num})**: {page_text[:350]}"
                        )

                        img_chunk = Chunk(
                            id=f"{doc_id}-p{page_num}-img{img_idx}",
                            document_id=doc_id,
                            rag_id=rag_id,
                            modality=ModalityType.IMAGE,
                            content=content_repr,
                            page_number=page_num,
                            asset_rel_path=img_filename,
                            embedding=img_emb,
                            metadata={
                                "source_file": file_path.name,
                                "page_number": page_num,
                                "width": w,
                                "height": h,
                                "image_format": img_ext,
                                "concept_name": eval_res.concept_name,
                                "caption": eval_res.caption,
                                "markdown_tag": eval_res.markdown_tag,
                                "is_conceptual": True,
                            }
                        )
                        chunks.append(img_chunk)
                except Exception:
                    pass

        pdf_doc.close()

        lib_doc = LibraryDocument(
            id=doc_id,
            rag_id=rag_id,
            original_filename=file_path.name,
            stored_filename=stored_filename,
            file_type=".pdf",
            mime_type="application/pdf",
            file_size_bytes=file_size,
            sha256=sha256_hash,
            page_count=page_count,
            image_count=extracted_images_count,
            chunk_count=len(chunks),
            metadata={"title": file_path.stem.replace("-", " "), "pages": page_count}
        )

        doc = Document(
            id=doc_id,
            rag_id=rag_id,
            filename=file_path.name,
            file_type=".pdf",
            file_size=file_size,
            modality=ModalityType.TEXT,
            chunk_count=len(chunks),
            page_count=page_count,
            image_count=extracted_images_count,
            metadata={"title": file_path.stem, "pages": page_count}
        )

        return doc, lib_doc, chunks

    def process_text_file(
        self,
        rag_id: str,
        file_path: Path,
        library_dir: Optional[Path] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> Tuple[Document, Optional[LibraryDocument], List[Chunk]]:
        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        doc_id = str(uuid.uuid4())

        is_code = file_path.suffix.lower() in [".py", ".js", ".ts", ".rs", ".go", ".cpp", ".c", ".json", ".yaml", ".sh"]
        modality = ModalityType.CODE if is_code else ModalityType.TEXT

        stored_filename = f"{doc_id}_{file_path.name}"
        lib_doc = None
        if library_dir:
            library_dir.mkdir(parents=True, exist_ok=True)
            stored_path = library_dir / stored_filename
            shutil.copy2(file_path, stored_path)
            sha256_hash = self._calc_sha256(file_path)
            lib_doc = LibraryDocument(
                id=doc_id,
                rag_id=rag_id,
                original_filename=file_path.name,
                stored_filename=stored_filename,
                file_type=file_path.suffix.lower(),
                mime_type="text/plain",
                file_size_bytes=file_path.stat().st_size,
                sha256=sha256_hash,
                page_count=1,
                chunk_count=0
            )

        raw_chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        chunks: List[Chunk] = []

        for i, text_segment in enumerate(raw_chunks):
            embedding = self.embedder.embed_text(text_segment)
            chunk = Chunk(
                id=f"{doc_id}-chk-{i}",
                document_id=doc_id,
                rag_id=rag_id,
                modality=modality,
                content=text_segment,
                page_number=1,
                embedding=embedding,
                metadata={
                    "source_file": file_path.name,
                    "chunk_index": i,
                    "char_count": len(text_segment)
                }
            )
            chunks.append(chunk)

        if lib_doc:
            lib_doc.chunk_count = len(chunks)

        doc = Document(
            id=doc_id,
            rag_id=rag_id,
            filename=file_path.name,
            file_type=file_path.suffix.lower(),
            file_size=file_path.stat().st_size,
            modality=modality,
            chunk_count=len(chunks),
            metadata={"lines": len(content.splitlines())}
        )

        return doc, lib_doc, chunks

    def process_image_file(
        self,
        rag_id: str,
        file_path: Path,
        assets_dir: Path,
        library_dir: Optional[Path] = None,
        caption: Optional[str] = None
    ) -> Tuple[Document, Optional[LibraryDocument], List[Chunk]]:
        file_path = Path(file_path)
        doc_id = str(uuid.uuid4())
        assets_dir.mkdir(parents=True, exist_ok=True)

        asset_name = f"{doc_id}_{file_path.name}"
        destination = assets_dir / asset_name
        shutil.copy2(file_path, destination)

        lib_doc = None
        if library_dir:
            library_dir.mkdir(parents=True, exist_ok=True)
            stored_lib = library_dir / asset_name
            shutil.copy2(file_path, stored_lib)
            sha256_hash = self._calc_sha256(file_path)
            lib_doc = LibraryDocument(
                id=doc_id,
                rag_id=rag_id,
                original_filename=file_path.name,
                stored_filename=asset_name,
                file_type=file_path.suffix.lower(),
                mime_type=f"image/{file_path.suffix.replace('.', '')}",
                file_size_bytes=file_path.stat().st_size,
                sha256=sha256_hash,
                page_count=1,
                image_count=1,
                chunk_count=1
            )

        with Image.open(destination) as img:
            width, height = img.size
            img_format = img.format or file_path.suffix.replace(".", "").upper()

        embedding = self.embedder.embed_image(destination)
        content_desc = caption or f"[🖼️ Imatge: {file_path.name} | Dimensions: {width}x{height} | Format: {img_format}]"

        chunk = Chunk(
            id=f"{doc_id}-chk-0",
            document_id=doc_id,
            rag_id=rag_id,
            modality=ModalityType.IMAGE,
            content=content_desc,
            page_number=1,
            asset_rel_path=asset_name,
            embedding=embedding,
            metadata={
                "source_file": file_path.name,
                "width": width,
                "height": height,
                "format": img_format
            }
        )

        doc = Document(
            id=doc_id,
            rag_id=rag_id,
            filename=file_path.name,
            file_type=file_path.suffix.lower(),
            file_size=file_path.stat().st_size,
            modality=ModalityType.IMAGE,
            chunk_count=1,
            metadata={"width": width, "height": height}
        )

        return doc, lib_doc, [chunk]

    def process_raw_text(
        self,
        rag_id: str,
        title: str,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> Tuple[Document, List[Chunk]]:
        doc_id = str(uuid.uuid4())
        raw_chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        chunks: List[Chunk] = []

        for i, text_segment in enumerate(raw_chunks):
            embedding = self.embedder.embed_text(text_segment)
            chunk = Chunk(
                id=f"{doc_id}-chk-{i}",
                document_id=doc_id,
                rag_id=rag_id,
                modality=ModalityType.TEXT,
                content=text_segment,
                page_number=1,
                embedding=embedding,
                metadata={"title": title, "chunk_index": i}
            )
            chunks.append(chunk)

        doc = Document(
            id=doc_id,
            rag_id=rag_id,
            filename=f"{title}.txt",
            file_type=".txt",
            file_size=len(text.encode("utf-8")),
            modality=ModalityType.TEXT,
            chunk_count=len(chunks),
            metadata={"title": title}
        )

        return doc, chunks

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_len + len(para) > chunk_size and current_chunk:
                joined = "\n\n".join(current_chunk)
                chunks.append(joined)
                overlap_text = joined[-chunk_overlap:] if len(joined) > chunk_overlap else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_len = sum(len(p) for p in current_chunk)
            else:
                current_chunk.append(para)
                current_len += len(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk).strip())

        return chunks if chunks else [text]
