"""
Textos.info Technical & Non-Fiction Library Scraper
Crawls public domain essays, philosophy, treatises, and historical works from https://www.textos.info/
Generates modular RAG capsules partitioned to <= 50 books per capsule.
All packages are styled with canonical Collage Punk aesthetic.
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

logger = logging.getLogger(__name__)

# Non-fiction / technical knowledge tags mapped to canonical Collage Punk styles
TEXTOS_TAGS = {
    "filosofia": {
        "name": "Filosofía Clásica y Moderna",
        "description": "Tractats filosòfics, dialèctica, epistemologia i ètica clàssica i contemporània.",
        "domain": "Filosofía y Pensamiento Crítico",
        "style": "anarcho_social_punk",
        "max_pages": 1
    },
    "politica": {
        "name": "Teoría Política y Estado",
        "description": "Doctrina política clàssica, organització social, sobirania i tractats sobre l'Estat.",
        "domain": "Ciencias Políticas y Sociedad",
        "style": "anarcho_social_punk",
        "max_pages": 1
    },
    "historia": {
        "name": "Historia y Crónica Universal",
        "description": "Cròniques històriques, fonts primàries i anàlisi de transformacions civilitzatòries.",
        "domain": "Historia y Civilización",
        "style": "geo_punk",
        "max_pages": 1
    },
    "ensayo": {
        "name": "Ensayo Científico y Humanista",
        "description": "Tractats analítics, crítica d'art, ciència i reflexió humanista.",
        "domain": "Ensayo y Epistemología",
        "style": "anarcho_social_punk",
        "max_pages": 1
    },
    "tratado": {
        "name": "Tratados y Metodología",
        "description": "Compendis metodològics, regles d'art i ciència pràctica.",
        "domain": "Metodología y Tratados",
        "style": "hacker_punk",
        "max_pages": 1
    }
}

# Strict blacklist of literary / fiction forms to exclude
FICTION_BLACKLIST_KEYWORDS = [
    "novela", "cuento", "poesia", "poema", "rimas", "teatro", "comedia",
    "tragedia", "drama", "romance", "amor", "ficcion", "infantil"
]

@dataclass
class TextosBook:
    author_slug: str
    book_slug: str
    title: str
    author_name: str
    pdf_url: str
    referer_url: str
    detail_url: str

class TextosInfoScraper:
    BASE_URL = "https://www.textos.info"

    def __init__(self, download_root: Path = Path("data/downloads/books/textos_info")):
        self.download_root = Path(download_root)
        self.download_root.mkdir(parents=True, exist_ok=True)
        self.ctx = ssl._create_unverified_context()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def _fetch_html(self, url: str) -> str:
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, context=self.ctx, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def get_books_for_tag(self, tag: str, max_books: int = 50) -> List[TextosBook]:
        """
        Discovers up to max_books (default 50) non-fiction books for a tag.
        """
        url = f"{self.BASE_URL}/libros/etiqueta/{tag}"
        logger.info(f"Escanejant etiqueta textos.info: {url}")
        html = self._fetch_html(url)

        # Regex to capture links to books: ./<author-slug>/<book-slug>
        pattern = r'<a\s+href=[\"\']\./([a-z0-9\-]+)/([a-z0-9\-]+)[\"\'][^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html)

        discovered: List[TextosBook] = []
        seen = set()

        for author_slug, book_slug, raw_title in matches:
            if author_slug in ["etiqueta", "autor", "editor", "blog", "autores", "libros"]:
                continue
            
            clean_slug = f"{author_slug}/{book_slug}"
            if clean_slug in seen:
                continue
            seen.add(clean_slug)

            title = raw_title.strip()
            # Exclude titles matching fiction blacklist
            lower_title = title.lower()
            if any(bad in lower_title for bad in FICTION_BLACKLIST_KEYWORDS):
                continue

            author_name = author_slug.replace("-", " ").title()
            pdf_url = f"{self.BASE_URL}/{author_slug}/{book_slug}/pdf"
            referer_url = f"{self.BASE_URL}/{author_slug}/{book_slug}/descargar-pdf"
            detail_url = f"{self.BASE_URL}/{author_slug}/{book_slug}"

            discovered.append(TextosBook(
                author_slug=author_slug,
                book_slug=book_slug,
                title=title,
                author_name=author_name,
                pdf_url=pdf_url,
                referer_url=referer_url,
                detail_url=detail_url
            ))

            if len(discovered) >= max_books:
                break

        return discovered

    def download_book(self, book: TextosBook, output_dir: Path) -> Optional[Path]:
        """
        Downloads binary PDF from textos.info with required Referer.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{book.author_slug}-{book.book_slug}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 1024:
            return pdf_path

        headers = dict(self.headers)
        headers["Referer"] = book.referer_url

        try:
            req = urllib.request.Request(book.pdf_url, headers=headers)
            with urllib.request.urlopen(req, context=self.ctx, timeout=30) as resp:
                data = resp.read()
                if not data.startswith(b"%PDF"):
                    logger.warning(f"Resposta no és PDF vàlid per {book.title} ({len(data)} bytes)")
                    return None
                pdf_path.write_bytes(data)
                return pdf_path
        except Exception as e:
            logger.error(f"Error descarregant {book.title}: {e}")
            return None
