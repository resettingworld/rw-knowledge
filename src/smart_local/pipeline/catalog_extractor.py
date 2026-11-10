"""
Front-Matter Extractor for PDF Books
Extracts the first 2-3 non-empty pages of PDF books (covers, title pages,
publisher info, legal deposit, credits, index) to enable AI agents to perform
deep academic, regional, and normative cataloging.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pymupdf

class FrontMatterExtractor:
    """
    Extracts front-matter text and metadata from PDF files.
    """
    def __init__(self, default_pages: int = 3):
        self.default_pages = default_pages

    def extract_from_pdf(self, pdf_path: Path, max_pages: Optional[int] = None) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        limit = max_pages or self.default_pages

        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        sha256 = h.hexdigest()
        file_size = pdf_path.stat().st_size

        doc = pymupdf.open(str(pdf_path))
        total_pages = len(doc)

        extracted_pages = []
        collected_text_parts = []

        # Find first non-empty pages up to limit
        for page_idx in range(min(total_pages, 8)):
            if len(extracted_pages) >= limit:
                break

            text = doc[page_idx].get_text() or ""
            stripped = text.strip()
            if len(stripped) > 20:  # Skip blank pages or tiny headers
                page_num = page_idx + 1
                extracted_pages.append({
                    "page_number": page_num,
                    "char_count": len(stripped),
                    "text": stripped
                })
                collected_text_parts.append(f"--- [PÀGINA {page_num}] ---\n{stripped}")

        # Extract embedded metadata if available
        meta = doc.metadata or {}
        doc.close()

        combined_front_matter = "\n\n".join(collected_text_parts)

        # Basic title heuristic
        title_guess = meta.get("title") or ""
        if not title_guess or len(title_guess) < 3 or title_guess.lower().endswith(".pdf"):
            title_guess = pdf_path.stem.replace("-", " ").title()

        author_guess = meta.get("author") or "Desconegut / Institucional"

        return {
            "filename": pdf_path.name,
            "path": str(pdf_path.resolve()),
            "file_size_bytes": file_size,
            "sha256": sha256,
            "total_pages": total_pages,
            "embedded_meta": {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creation_date": meta.get("creationDate", "")
            },
            "title_guess": title_guess,
            "author_guess": author_guess,
            "pages_analyzed_count": len(extracted_pages),
            "pages": extracted_pages,
            "front_matter_text": combined_front_matter
        }

    def scan_directory(self, dir_path: Path, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        pdf_files = sorted(list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.PDF")))
        results = []
        for pdf_path in pdf_files:
            try:
                item = self.extract_from_pdf(pdf_path, max_pages=max_pages)
                results.append(item)
            except Exception as e:
                results.append({
                    "filename": pdf_path.name,
                    "path": str(pdf_path.resolve()),
                    "error": str(e)
                })
        return results
