"""
Project Gutenberg Technical Knowledge Scraper & Capsule Generator
Discovers, filters, downloads, converts and packages technical and scientific knowledge from:
https://www.gutenberg.org/ebooks/categories.html

Strictly excludes fiction, literature, romance, novels, poetry, drama, and youth reading.
Focuses on:
- Engineering & Technology
- Mathematics
- Physics, Chemistry, Biology, Earth Sciences & Agriculture
- Research Methods, Statistics & Information Systems
- Health, Medicine & Pharmacology
- Economics, Business, Law & Architecture
- Philosophy, Reference & Practical Manuals
"""

import hashlib
import json
import os
import re
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import fitz  # PyMuPDF
import httpx

from smart_local.config import settings

GUTENBERG_BASE_URL = "https://www.gutenberg.org"
GUTENBERG_CATEGORIES_URL = f"{GUTENBERG_BASE_URL}/ebooks/categories.html"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Strictly excluded fiction / literature keywords
FICTION_KEYWORDS = [
    "literature", "novel", "poetry", "short stories", "adventure", "plays", "drama",
    "romance", "fiction", "fantasy", "mystery", "thriller", "crime", "humour",
    "children", "young adult", "erotica", "travel writing"
]

# Vetted Technical and Human Knowledge Bookshelves
TECHNICAL_BOOKSHELVES_MAP = {
    # 1. Engineering & Technology
    "671": {
        "slug": "engineering-technology",
        "name": "Engineering & Technology",
        "cluster": "engineering",
        "domain": "Ingeniería y Tecnología",
        "style_key": "cyber_mechanic_punk",
        "description": "Enginyeria mecànica, elèctrica, aeronàutica, processos industrials i tecnologia històrica."
    },
    "672": {
        "slug": "mathematics",
        "name": "Mathematics",
        "cluster": "engineering",
        "domain": "Ciencias Exactas",
        "style_key": "hacker_punk",
        "description": "Càlcul, àlgebra, geometria, teoria de nombres i fonaments matemàtics universals."
    },
    "673": {
        "slug": "research-methods-information-systems",
        "name": "Research Methods & Information Systems",
        "cluster": "engineering",
        "domain": "Metodología y Tecnología de la Información",
        "style_key": "hacker_punk",
        "description": "Mètodes d'investigació empírica, estadística analítica i sistemes d'informació."
    },
    "679": {
        "slug": "practical-manuals-how-to",
        "name": "Practical Manuals & How To",
        "cluster": "engineering",
        "domain": "Tecnología Aplicada y Oficios",
        "style_key": "cyber_mechanic_punk",
        "description": "Manuals pràctics d'oficis, procediments tècnics, fusteria, fosa, mecànica i construcció."
    },
    "674": {
        "slug": "architecture",
        "name": "Architecture & Structural Design",
        "cluster": "engineering",
        "domain": "Arquitectura y Estructuras",
        "style_key": "cyber_mechanic_punk",
        "description": "Disseny arquitectònic, càlcul estructural, història constructiva i urbanisme."
    },

    # 2. Fundamental & Natural Sciences
    "667": {
        "slug": "physics",
        "name": "Physics",
        "cluster": "sciences",
        "domain": "Ciencias Exactas y Naturales",
        "style_key": "geo_punk",
        "description": "Mecànica clàssica, termodinàmica, electromagnetisme, òptica i física teòrica."
    },
    "668": {
        "slug": "chemistry-biochemistry",
        "name": "Chemistry & Biochemistry",
        "cluster": "sciences",
        "domain": "Ciencias Exactas y Naturales",
        "style_key": "bio_punk",
        "description": "Química orgànica i inorgànica, anàlisi química, biofísica i processos moleculars."
    },
    "669": {
        "slug": "biology",
        "name": "Biology & Natural History",
        "cluster": "sciences",
        "domain": "Ciencias de la Vida",
        "style_key": "bio_punk",
        "description": "Zoologia, botànica, fisiologia, teoria de l'evolució i ciències biològiques."
    },
    "670": {
        "slug": "earth-agricultural-sciences",
        "name": "Earth & Agricultural Sciences",
        "cluster": "sciences",
        "domain": "Ciencias de la Tierra y Agronomía",
        "style_key": "geo_punk",
        "description": "Geologia, mineralogia, agronomia, edafologia i maneig de recursos naturals."
    },
    "685": {
        "slug": "environmental-issues",
        "name": "Environmental Issues & Conservation",
        "cluster": "sciences",
        "domain": "Ciencias Ambientales",
        "style_key": "geo_punk",
        "description": "Ecologia, conservació d'ecosistemes, silvicultura i gestió de recursos hídrics."
    },

    # 3. Medical Sciences & Health
    "681": {
        "slug": "health-medicine",
        "name": "Health & Medicine",
        "cluster": "medicine",
        "domain": "Salud y Medicina",
        "style_key": "bio_punk",
        "description": "Anatomia humana, cirurgia, patologia, fisiologia mèdica i epidemiologia."
    },
    "682": {
        "slug": "pharmacology-toxicology",
        "name": "Pharmacology & Toxicology",
        "cluster": "medicine",
        "domain": "Farmacología y Salud",
        "style_key": "bio_punk",
        "description": "Tractats de farmacopea, química farmacèutica, acció terapèutica i toxicologia."
    },
    "684": {
        "slug": "nutrition-dietetics",
        "name": "Nutrition & Dietetics",
        "cluster": "medicine",
        "domain": "Nutrición y Salud",
        "style_key": "bio_punk",
        "description": "Bases de la nutrició humana, dietètica mèdica i química dels aliments."
    },
    "688": {
        "slug": "psychiatry-psychology",
        "name": "Psychiatry & Psychology",
        "cluster": "medicine",
        "domain": "Ciencias de la Conducta",
        "style_key": "anarcho_social_punk",
        "description": "Psicologia experimental, processos cognitius, psiquiatria clínica i psicoanàlisi."
    },

    # 4. Economics, Business, Law & Social Sciences
    "696": {
        "slug": "economics",
        "name": "Economics",
        "cluster": "social_sciences",
        "domain": "Economía y Finanzas",
        "style_key": "anarcho_social_punk",
        "description": "Economia política, comerç internacional, teoria monetària i desenvolupament."
    },
    "695": {
        "slug": "business-management",
        "name": "Business & Industrial Management",
        "cluster": "social_sciences",
        "domain": "Administración y Negocios",
        "style_key": "cyber_mechanic_punk",
        "description": "Organització científica del treball, comptabilitat, logística i gestió empresarial."
    },
    "689": {
        "slug": "law-criminology",
        "name": "Law & Criminology",
        "cluster": "social_sciences",
        "domain": "Ciencias Jurídicas",
        "style_key": "anarcho_social_punk",
        "description": "Jurisprudència, dret constitucional, dret internacional, tractats legals i criminologia."
    },
    "693": {
        "slug": "sociology",
        "name": "Sociology",
        "cluster": "social_sciences",
        "domain": "Ciencias Sociales",
        "style_key": "anarcho_social_punk",
        "description": "Estructura social, demografia, institucions cíviques i dinàmiques de grup."
    },
    "686": {
        "slug": "archaeology-anthropology",
        "name": "Archaeology & Anthropology",
        "cluster": "social_sciences",
        "domain": "Antropología y Arqueología",
        "style_key": "anarcho_social_punk",
        "description": "Evolució humana, mètodes d'excavació arqueològica, cultures materials i etnografia."
    },

    # 5. Philosophy, Reference & Epistemology
    "691": {
        "slug": "philosophy-ethics",
        "name": "Philosophy, Logic & Ethics",
        "cluster": "philosophy",
        "domain": "Filosofía y Lógica",
        "style_key": "anarcho_social_punk",
        "description": "Lògica formal, teoria del coneixement (epistemologia), ètica i filosofia de la ciència."
    },
    "697": {
        "slug": "encyclopedias-reference",
        "name": "Encyclopedias & Reference",
        "cluster": "philosophy",
        "domain": "Obras de Referencia y Enciclopedias",
        "style_key": "general_punk",
        "description": "Compendis del saber universal, diccionaris tècnics i sumes enciclopèdiques."
    },
    "704": {
        "slug": "teaching-education",
        "name": "Teaching & Educational Methods",
        "cluster": "philosophy",
        "domain": "Pedagogía y Educación",
        "style_key": "general_punk",
        "description": "Mètodes d'instrucció docent, teoria de l'aprenentatge i pedagogia científica."
    }
}

class GutenbergBookshelfParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.books = []
        self.current_href = None
        self.current_title = []
        self.current_author = []
        self.in_title = False
        self.in_author = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a" and "/ebooks/" in attrs_dict.get("href", ""):
            self.current_href = attrs_dict["href"]
        if tag == "span" and "title" in attrs_dict.get("class", ""):
            self.in_title = True
            self.current_title = []
        if tag == "span" and "subtitle" in attrs_dict.get("class", ""):
            self.in_author = True
            self.current_author = []

    def handle_endtag(self, tag):
        if tag == "span" and self.in_title:
            self.in_title = False
        if tag == "span" and self.in_author:
            self.in_author = False
        if tag == "li":
            if self.current_href and self.current_title:
                title = "".join(self.current_title).strip()
                author = "".join(self.current_author).strip() or "Various / Project Gutenberg"
                # Extract ebook ID
                m = re.search(r'/ebooks/(\d+)', self.current_href)
                ebook_id = m.group(1) if m else None
                if ebook_id:
                    self.books.append({
                        "ebook_id": ebook_id,
                        "title": title,
                        "author": author,
                        "url": f"{GUTENBERG_BASE_URL}/ebooks/{ebook_id}",
                    })
                self.current_href = None
                self.current_title = []
                self.current_author = []

    def handle_data(self, data):
        if self.in_title:
            self.current_title.append(data)
        if self.in_author:
            self.current_author.append(data)

class ProjectGutenbergScraper:
    """
    Crawls and downloads technical / scientific human knowledge from Project Gutenberg,
    converting texts into clean, high-precision PDFs ready for modular RAG packaging.
    """
    def __init__(
        self,
        downloads_base: Path = Path("data/downloads/books/gutenberg"),
        client: Optional[httpx.Client] = None,
    ):
        self.downloads_base = Path(downloads_base)
        self.downloads_base.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=30.0,
        )

    def list_vetted_topics(self) -> List[Dict[str, Any]]:
        """Returns the vetted list of non-fiction, scientific and technical bookshelves."""
        topics = []
        for shelf_id, meta in sorted(TECHNICAL_BOOKSHELVES_MAP.items(), key=lambda x: x[1]["name"]):
            topics.append({
                "bookshelf_id": shelf_id,
                "url": f"{GUTENBERG_BASE_URL}/ebooks/bookshelf/{shelf_id}",
                **meta
            })
        return topics

    def get_bookshelf_books(self, bookshelf_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrapes books from a specific Gutenberg technical bookshelf."""
        shelf_url = f"{GUTENBERG_BASE_URL}/ebooks/bookshelf/{bookshelf_id}"
        res = self.client.get(shelf_url)
        if res.status_code != 200:
            raise RuntimeError(f"Failed to fetch bookshelf {shelf_url}: {res.status_code}")

        parser = GutenbergBookshelfParser()
        parser.feed(res.text)

        books = parser.books
        if limit:
            books = books[:limit]
        return books

    def download_and_convert_to_pdf(
        self,
        ebook_id: str,
        dest_pdf_path: Path,
        preferred_title: Optional[str] = None,
        author: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Downloads a Gutenberg ebook (trying UTF-8 text or EPUB) and converts it into a high-quality PDF.
        """
        dest_pdf_path = Path(dest_pdf_path)
        dest_pdf_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_pdf_path.exists() and dest_pdf_path.stat().st_size > 1000:
            # Already downloaded and converted
            h = hashlib.sha256()
            with open(dest_pdf_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return {
                "path": dest_pdf_path,
                "size_bytes": dest_pdf_path.stat().st_size,
                "sha256": h.hexdigest(),
                "pages": len(fitz.open(dest_pdf_path))
            }

        # Try download sources: 1. txt.utf-8, 2. epub3.images, 3. epub.images
        txt_url = f"{GUTENBERG_BASE_URL}/ebooks/{ebook_id}.txt.utf-8"
        epub_url = f"{GUTENBERG_BASE_URL}/ebooks/{ebook_id}.epub.images"

        pdf_doc = None
        # Try UTF-8 plain text first for maximum speed and clean typography
        try:
            res = self.client.get(txt_url)
            if res.status_code == 200 and len(res.content) > 1000:
                raw_text = res.text
                # Strip excessive Gutenberg boilerplate headers if present
                clean_text = self._clean_gutenberg_text(raw_text, preferred_title, author)
                doc = fitz.open(stream=clean_text.encode("utf-8"), filetype="txt")
                pdf_bytes = doc.convert_to_pdf()
                pdf_doc = fitz.open("pdf", pdf_bytes)
        except Exception:
            pdf_doc = None

        # Fallback to EPUB if txt wasn't available
        if not pdf_doc:
            try:
                res = self.client.get(epub_url)
                if res.status_code == 200 and len(res.content) > 1000:
                    doc = fitz.open(stream=res.content, filetype="epub")
                    pdf_bytes = doc.convert_to_pdf()
                    pdf_doc = fitz.open("pdf", pdf_bytes)
            except Exception:
                pdf_doc = None

        if not pdf_doc:
            return None

        pdf_doc.save(dest_pdf_path)
        file_size = dest_pdf_path.stat().st_size
        page_count = len(pdf_doc)

        h = hashlib.sha256()
        with open(dest_pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)

        return {
            "path": dest_pdf_path,
            "size_bytes": file_size,
            "sha256": h.hexdigest(),
            "pages": page_count
        }

    def _clean_gutenberg_text(self, text: str, title: Optional[str] = None, author: Optional[str] = None) -> str:
        """Removes the standard Gutenberg legal headers and prepends a clear academic title header."""
        header_end = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
        if header_end != -1:
            line_break = text.find("\n", header_end)
            text = text[line_break + 1:] if line_break != -1 else text[header_end:]

        footer_start = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        if footer_start != -1:
            text = text[:footer_start]

        title_line = title or "Project Gutenberg Monograph"
        author_line = author or "Historical Scholar / Project Gutenberg"
        
        academic_header = (
            f"======================================================================\n"
            f"{title_line.upper()}\n"
            f"Author: {author_line}\n"
            f"Source: Project Gutenberg Technical Collection\n"
            f"Academic Archive Edition\n"
            f"======================================================================\n\n"
        )
        return academic_header + text.strip()
