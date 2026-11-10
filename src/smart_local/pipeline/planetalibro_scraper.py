"""
PlanetaLibro Non-Fiction & Technical Knowledge Scraper
Crawls public domain & open non-fiction treatises, philosophy, science, and strategy from https://planetalibro.net/
Converts reader HTML fragments to clean PDFs using PyMuPDF.
Guarantees <= 50 books per capsule, fully packaged and styled with canonical Collage Punk aesthetic.
"""

import json
import logging
import os
import re
import ssl
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import fitz

logger = logging.getLogger(__name__)

PLANETALIBRO_TAGS = {
    "estrategia": {
        "name": "Estrategia Militar y Organizativa",
        "description": "Tractats clàssics d'estratègia, organització tàctica i lideratge pragmàtic.",
        "domain": "Estrategia y Organización",
        "style": "hacker_punk"
    },
    "filosofia": {
        "name": "Filosofía y Razón Clásica",
        "description": "Compendi de tractats filosòfics, ètica, epistemologia i diàlegs fonamentals.",
        "domain": "Filosofía y Epistemología",
        "style": "anarcho_social_punk"
    },
    "politica": {
        "name": "Doctrina Política y Estado",
        "description": "Tractats clàssics sobre governança, l'Estat, la llei i la societat civil.",
        "domain": "Ciencias Políticas",
        "style": "anarcho_social_punk"
    },
    "economia": {
        "name": "Economía Política y Recursos",
        "description": "Fonaments del valor, comerç, distribució de recursos i teoria econòmica clàssica.",
        "domain": "Economía y Comercio",
        "style": "anarcho_social_punk"
    },
    "ciencia": {
        "name": "Ciencias y Filosofía Natural",
        "description": "Observacions de la naturalesa, mètode científic i cosmologia empírica.",
        "domain": "Ciencias de la Naturaleza",
        "style": "geo_punk"
    }
}

# Strict blacklist of literary / fiction forms and modern novelists to exclude
FICTION_BLACKLIST_KEYWORDS = [
    "novela", "cuento", "poesia", "poema", "rimas", "teatro", "comedia",
    "tragedia", "drama", "romance", "amor", "ficcion", "infantil", "vampiro",
    "dan-brown", "sanderson", "meyer", "collins", "maas", "cartland",
    "mcfadden", "yarros", "garber", "chloe-miel", "mercedes-ron"
]

@dataclass
class PlanetaLibroBook:
    uri: str
    author_uri: str
    title: str
    author_name: str
    description: Optional[str]
    cover_url: Optional[str]

class PlanetaLibroScraper:
    API_BASE = "https://planetalibro.net/api/v1/public"

    def __init__(self, download_root: Path = Path("data/downloads/books/planetalibro")):
        self.download_root = Path(download_root)
        self.download_root.mkdir(parents=True, exist_ok=True)
        self.ctx = ssl._create_unverified_context()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

    def _fetch_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, context=self.ctx, timeout=25) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_books_for_tag(self, tag: str, max_books: int = 50) -> List[PlanetaLibroBook]:
        """
        Discovers up to max_books (default 50) non-fiction books for a tag.
        """
        url = f"{self.API_BASE}/libros/por-tag?tag={tag}&limit=50"
        logger.info(f"Escanejant etiqueta PlanetaLibro: {tag}")
        try:
            resp = self._fetch_json(url)
            items = resp.get("data", {}).get("items", [])
        except Exception as e:
            logger.error(f"Error consultant etiqueta {tag}: {e}")
            return []

        books: List[PlanetaLibroBook] = []
        for item in items:
            uri = item.get("uri", "")
            title = item.get("titulo", "")
            author_uri = item.get("author_uri", "")

            # Exclude fiction
            lower = f"{uri} {title}".lower()
            if any(bad in lower for bad in FICTION_BLACKLIST_KEYWORDS):
                continue

            author_name = author_uri.replace("-", " ").title() if author_uri else "Desconegut"

            books.append(PlanetaLibroBook(
                uri=uri,
                author_uri=author_uri,
                title=title,
                author_name=author_name,
                description=item.get("descripcion"),
                cover_url=item.get("cover_url")
            ))

            if len(books) >= max_books:
                break

        return books

    def download_and_convert_book(self, book: PlanetaLibroBook, output_dir: Path) -> Optional[Path]:
        """
        Fetches reader manifest & fragments, converts HTML to PDF using PyMuPDF fitz.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{book.uri}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 1024:
            return pdf_path

        try:
            # 1. Fetch reader manifest
            manifest_url = f"{self.API_BASE}/reader-manifest/{book.uri}"
            manifest_data = self._fetch_json(manifest_url).get("data", {})
            total_pages = manifest_data.get("pages", 1)

            # 2. Collect fragments
            # To avoid excessive roundtrips for giant books, fetch up to 30 significant chapters/fragments
            html_parts = [
                f"<html><head><meta charset='utf-8'/><style>body{{font-family:sans-serif;margin:40px;line-height:1.6;}} h1{{color:#111;}}</style></head><body>",
                f"<h1>{book.title}</h1><p><strong>Autor:</strong> {book.author_name}</p><hr/>"
            ]

            # Sample pages / fragments from 1 up to min(total_pages, 35)
            step = 1 if total_pages <= 35 else max(1, total_pages // 35)
            pages_to_fetch = list(range(1, min(total_pages + 1, 36), step))

            for page_num in pages_to_fetch:
                try:
                    frag_url = f"{self.API_BASE}/reader-fragment/{book.uri}/{page_num}"
                    frag_data = self._fetch_json(frag_url).get("data", {})
                    frag_html = frag_data.get("html", "")
                    if frag_html:
                        html_parts.append(frag_html)
                except Exception:
                    continue

            html_parts.append("</body></html>")
            full_html = "\n".join(html_parts)

            # 3. Convert HTML to clean PDF via PyMuPDF
            doc = fitz.open(stream=full_html.encode("utf-8"), filetype="html")
            pdf_bytes = doc.convert_to_pdf()
            pdf_doc = fitz.open("pdf", pdf_bytes)
            pdf_doc.save(str(pdf_path))
            pdf_doc.close()

            logger.info(f"✓ Generat PDF per {book.title}: {len(doc)} pàgines -> {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"Error convertint {book.title} a PDF: {e}")
            return None
