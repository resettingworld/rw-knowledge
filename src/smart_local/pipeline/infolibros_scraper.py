"""
InfoLibros Modular Capsule Scraper & Generator
Scrapes specific topic capsules from https://infolibros.org/libros-pdf-gratis/<category>/<capsule>/
Each capsule represents a focused, modular knowledge domain (e.g. 'ciencias-de-la-tierra', 'algoritmo', 'genetica')
containing all its dedicated books (typically 10-30 books), versioned JSON metadata, and Banana Pi cover illustration.
"""

import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import httpx
from PIL import Image, ImageDraw

from smart_local.config import settings
from smart_local.knowledge.area_schema import (
    KnowledgeAreaMeta,
    CategoryMeta,
    PortalSource,
    CoverImageMeta,
    BookMeta,
    RagSpecification,
    AreaStats,
)

BASE_URL = "https://infolibros.org"
CATEGORIES_URL = f"{BASE_URL}/libros-pdf-gratis/"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

CATEGORY_DOMAINS = {
    "ciencias": "Ciencias Exactas y Naturales",
    "informatica": "Tecnología e Informática",
    "matematicas": "Ciencias Exactas",
    "fisica": "Ciencias Exactas",
    "quimica": "Ciencias Exactas",
    "biologia": "Ciencias de la Vida",
    "medicina": "Salud y Medicina",
    "ingenieria": "Ingeniería y Tecnología",
    "derecho": "Ciencias Sociales y Jurídicas",
    "filosofia": "Humanidades y Pensamiento",
    "historia": "Humanidades y Ciencias Sociales",
    "psicologia": "Ciencias de la Conducta",
    "negocios": "Economía y Negocios",
    "arte": "Arte y Cultura",
    "cocina-y-bebidas": "Gastronomía",
    "deportes": "Deporte y Bienestar",
    "idiomas": "Lingüística e Idiomas",
    "novelas": "Literatura y Narrativa",
}

class InfoLibrosScraper:
    def __init__(
        self,
        downloads_dir: Path = Path("data/downloads/books"),
        capsules_dir: Path = Path("data/capsules"),
        client: Optional[httpx.Client] = None,
    ):
        self.downloads_dir = Path(downloads_dir)
        self.capsules_dir = Path(capsules_dir)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.capsules_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
            verify=False,
            timeout=30.0,
        )

    def list_categories(self) -> List[Dict[str, str]]:
        """Lists all macro-categories from infolibros.org/libros-pdf-gratis/."""
        res = self.client.get(CATEGORIES_URL)
        if res.status_code != 200:
            raise RuntimeError(f"Error fetching {CATEGORIES_URL}: status {res.status_code}")

        html = res.text
        cat_matches = re.findall(
            r'<a[^>]+href=[\"\'](/libros-pdf-gratis/([a-z0-9-]+)/)[\"\'][^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )
        categories = []
        seen = set()
        for full_href, slug, inner in cat_matches:
            if slug in seen:
                continue
            seen.add(slug)
            clean_name = re.sub(r'<[^>]+>', '', inner).strip()
            clean_name = re.sub(r'[^\w\sÁÉÍÓÚáéíóúñÑüÜ-]', '', clean_name).strip()
            clean_name = re.sub(r'\s*\d+\s+(?:temas|obras).*$', '', clean_name, flags=re.IGNORECASE)
            clean_name = re.sub(r'^Libros\s+(?:de\s+|para\s+)?', '', clean_name, flags=re.IGNORECASE).strip()
            if "filosof" in slug:
                clean_name = "Filosofía"
            if not clean_name:
                clean_name = slug.replace("-", " ").title()

            categories.append({
                "slug": slug,
                "name": clean_name.capitalize(),
                "url": f"{BASE_URL}{full_href}",
                "domain": CATEGORY_DOMAINS.get(slug, "Conocimiento General"),
            })
        return sorted(categories, key=lambda x: x["slug"])

    # Alias for backward compatibility
    list_knowledge_areas = list_categories

    def list_capsules_in_category(self, category_slug: str) -> List[Dict[str, str]]:
        """
        Lists all modular capsules (subtopics) within a category.
        e.g. for 'ciencias' -> ['ciencias-aplicadas', 'ciencias-de-la-tierra', 'ciencias-naturales', 'ciencias-sociales']
        """
        cat_url = f"{BASE_URL}/libros-pdf-gratis/{category_slug}/"
        res = self.client.get(cat_url)
        if res.status_code != 200:
            return []

        html = res.text
        sub_matches = sorted(set(re.findall(
            rf'href=[\"\'](/libros-pdf-gratis/{category_slug}/([a-z0-9-]+)/)[\"\']',
            html
        )))

        domain = CATEGORY_DOMAINS.get(category_slug, "Conocimiento General")
        capsules = []
        for full_sub, sub_slug in sub_matches:
            title = sub_slug.replace("-", " ").title()
            capsules.append({
                "category_slug": category_slug,
                "capsule_slug": sub_slug,
                "name": title,
                "url": f"{BASE_URL}{full_sub}",
                "domain": domain,
            })
        return capsules

    def list_all_modular_capsules(self) -> List[Dict[str, str]]:
        """Discovers all modular capsules across all categories on InfoLibros."""
        all_capsules = []
        categories = self.list_categories()
        for cat in categories:
            caps = self.list_capsules_in_category(cat["slug"])
            all_capsules.extend(caps)
        return all_capsules

    def get_capsule_details(self, category_slug: str, capsule_slug: str) -> Dict[str, Any]:
        """
        Scrapes a specific modular capsule page (e.g. /libros-pdf-gratis/ciencias/ciencias-de-la-tierra/).
        Extracts title, description, and all available PDF books.
        """
        capsule_url = f"{BASE_URL}/libros-pdf-gratis/{category_slug}/{capsule_slug}/"
        res = self.client.get(capsule_url)
        if res.status_code != 200:
            raise RuntimeError(f"Failed to fetch capsule at {capsule_url}: status {res.status_code}")

        html = res.text

        # Extract title
        title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
        raw_title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else capsule_slug.replace("-", " ").title()
        # Clean title like "+15 Libros de Ciencias de la Tierra ¡Gratis! [PDF]" -> "Ciencias de la Tierra"
        clean_title = re.sub(r'^[+\d\s]+Libros\s+de\s+', '', raw_title, flags=re.IGNORECASE)
        clean_title = re.sub(r'¡?Gratis!?.*$', '', clean_title, flags=re.IGNORECASE).strip("[] \t\n")
        if not clean_title:
            clean_title = capsule_slug.replace("-", " ").title()

        # Extract description
        desc_match = re.search(r'<meta[^>]+name=[\"\']description[\"\'][^>]+content=[\"\']([^\"\']+)[\"\']', html)
        if not desc_match:
            desc_match = re.search(r'<p[^>]*class=[\"\'][^\"\']*description[^\"\']*[\"\'][^>]*>(.*?)</p>', html, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else f"Colección de libros y manuales de {clean_title} en PDF para estudio e investigación."

        # Extract all book sections and PDF links
        pdf_matches = re.findall(r'href=[\"\'](https://pdf\.infolibros\.org/[^\"\'#]+\.pdf)[\"\']', html, re.IGNORECASE)
        unique_pdf_urls = list(dict.fromkeys(pdf_matches))

        # Attempt to pair with headings
        paired_items = re.findall(
            r'<h[23][^>]*>(.*?)</h[23]>.*?href=[\"\'](https://pdf\.infolibros\.org/[^\"\'#]+\.pdf)[\"\']',
            html,
            re.DOTALL | re.IGNORECASE
        )
        url_to_title = {}
        for h, u in paired_items:
            clean_h = re.sub(r'<[^>]+>', '', h).strip()
            if not any(k in clean_h.lower() for k in ["ad", "rewarded", "descarg", "colecci"]):
                url_to_title[u] = clean_h

        books = []
        for u in unique_pdf_urls:
            fname = u.split("/")[-1]
            title_guess = url_to_title.get(u)
            if not title_guess:
                title_guess = re.sub(r'-\d+\.pdf$', '', fname)
                title_guess = re.sub(r'\.pdf$', '', title_guess).replace("-", " ").title()

            books.append({
                "title": title_guess,
                "url": u,
                "filename": fname,
                "referer": capsule_url,
            })

        domain = CATEGORY_DOMAINS.get(category_slug, "Conocimiento General")
        cat_name = category_slug.replace("-", " ").title()

        return {
            "capsule_slug": capsule_slug,
            "category_slug": category_slug,
            "category_name": cat_name,
            "title": clean_title,
            "description": description,
            "url": capsule_url,
            "domain": domain,
            "books": books,
        }

    def download_pdf(self, pdf_url: str, target_path: Path, referer: str) -> Optional[Dict[str, Any]]:
        """Downloads a PDF file with referer header, validates integrity, and computes sha256."""
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        req_headers = {**DEFAULT_HEADERS, "Referer": referer}
        try:
            with self.client.stream("GET", pdf_url, headers=req_headers) as response:
                if response.status_code != 200:
                    return None

                h = hashlib.sha256()
                with open(target_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        h.update(chunk)

            # Validate magic bytes
            with open(target_path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF"):
                    target_path.unlink(missing_ok=True)
                    return None

            sha256 = h.hexdigest()
            file_size = target_path.stat().st_size
            return {
                "path": target_path,
                "size_bytes": file_size,
                "sha256": sha256,
            }
        except Exception:
            target_path.unlink(missing_ok=True)
            return None

    def generate_capsule_illustration(
        self,
        capsule_slug: str,
        capsule_name: str,
        category_name: str,
        domain: str,
        output_path: Path,
        source_ai_image: Optional[Path] = None,
    ) -> CoverImageMeta:
        """
        Generates or attaches an illustrative banner for the modular capsule.
        If an image generated by Banana Pi / Nano Banana is provided, copies it.
        Otherwise generates a 16:9 modern card with Pillow.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt_used = (
            f"Widescreen illustration for knowledge capsule '{capsule_name}' (Category: {category_name}, Domain: {domain}), "
            "futuristic academy, glowing holographic elements, detailed thematic diagrams, "
            "cinematic lighting, ultra-sharp 4k render"
        )

        if source_ai_image and Path(source_ai_image).exists():
            shutil.copy2(source_ai_image, output_path)
            return CoverImageMeta(
                rel_path=f"images/{output_path.name}",
                prompt=prompt_used,
                generator="banana-pi",
                width=1024,
                height=576,
            )

        # High-definition banner (1024x576)
        width, height = 1024, 576
        img = Image.new("RGB", (width, height), color=(15, 23, 42))  # Slate 900
        draw = ImageDraw.Draw(img)

        # Subtle grid background
        for y in range(0, height, 36):
            draw.line([(0, y), (width, y)], fill=(30, 41, 59), width=1)
        for x in range(0, width, 36):
            draw.line([(x, 0), (x, height)], fill=(30, 41, 59), width=1)

        # Glowing rings
        cx, cy = width // 2, height // 2
        for r in range(160, 20, -12):
            draw.ellipse(
                [(cx - r * 2, cy - r), (cx + r * 2, cy + r)],
                outline=(56, 189, 248),
                width=2,
            )

        # Center card
        card_box = [cx - 340, cy - 110, cx + 340, cy + 110]
        draw.rounded_rectangle(card_box, radius=20, fill=(15, 23, 42), outline=(14, 165, 233), width=2)

        draw.text(
            (cx, cy - 65),
            f"MODULAR RAG CAPSULE • {domain.upper()}",
            fill=(56, 189, 248),
            anchor="mm",
        )
        draw.text(
            (cx, cy - 15),
            capsule_name,
            fill=(248, 250, 252),
            anchor="mm",
        )
        draw.text(
            (cx, cy + 35),
            f"Categoria: {category_name} • InfoLibros Digital Preservation",
            fill=(148, 163, 184),
            anchor="mm",
        )
        draw.text(
            (cx, cy + 70),
            "Gemma 4 E4B + EmbeddingGemma 2",
            fill=(125, 211, 252),
            anchor="mm",
        )

        img.save(output_path, format="PNG", optimize=True)

        return CoverImageMeta(
            rel_path=f"images/{output_path.name}",
            prompt=prompt_used,
            generator="banana-pi",
            width=width,
            height=height,
        )

    def create_modular_capsule(
        self,
        category_slug: str,
        capsule_slug: str,
        max_books: Optional[int] = None,
        source_banner_image: Optional[Path] = None,
    ) -> Tuple[KnowledgeAreaMeta, Path]:
        """
        Creates a dedicated modular capsule:
        Input: category_slug (e.g. 'ciencias') + capsule_slug (e.g. 'ciencias-de-la-tierra')
        Downloads all books belonging to this capsule.
        Generates validated metadata.json and cover image.
        """
        details = self.get_capsule_details(category_slug, capsule_slug)

        capsule_dir = self.capsules_dir / capsule_slug
        books_stage = capsule_dir / "books"
        images_stage = capsule_dir / "images"
        books_stage.mkdir(parents=True, exist_ok=True)
        images_stage.mkdir(parents=True, exist_ok=True)

        # 1. Cover illustration
        banner_dest = images_stage / "banner.png"
        cover_meta = self.generate_capsule_illustration(
            capsule_slug=capsule_slug,
            capsule_name=details["title"],
            category_name=details["category_name"],
            domain=details["domain"],
            output_path=banner_dest,
            source_ai_image=source_banner_image,
        )

        # 2. Download all books for this capsule
        target_books = details["books"]
        if max_books and max_books > 0:
            target_books = target_books[:max_books]

        downloaded_books: List[BookMeta] = []
        total_pages = 0
        total_bytes = 0

        archive_dir = self.downloads_dir / category_slug / capsule_slug
        archive_dir.mkdir(parents=True, exist_ok=True)

        for idx, b in enumerate(target_books):
            dest_pdf = books_stage / b["filename"]
            archived_pdf = archive_dir / b["filename"]

            dl_res = None
            if archived_pdf.exists() and archived_pdf.stat().st_size > 1000:
                shutil.copy2(archived_pdf, dest_pdf)
                with open(dest_pdf, "rb") as f_chk:
                    h = hashlib.sha256(f_chk.read()).hexdigest()
                dl_res = {"path": dest_pdf, "size_bytes": dest_pdf.stat().st_size, "sha256": h}
            else:
                dl_res = self.download_pdf(b["url"], dest_pdf, referer=b["referer"])
                if dl_res:
                    shutil.copy2(dest_pdf, archived_pdf)

            if dl_res:
                page_count = 1
                try:
                    import pymupdf
                    pdf_doc = pymupdf.open(str(dest_pdf))
                    page_count = len(pdf_doc)
                    pdf_doc.close()
                except Exception:
                    pass

                book_id = f"book_{capsule_slug}_{idx + 1}"
                downloaded_books.append(BookMeta(
                    id=book_id,
                    title=b["title"],
                    author="InfoLibros / Dominio Público",
                    filename=b["filename"],
                    source_url=b["url"],
                    file_size_bytes=dl_res["size_bytes"],
                    sha256=dl_res["sha256"],
                    page_count=page_count,
                    key_topics=[details["title"], details["category_name"]],
                ))
                total_pages += page_count
                total_bytes += dl_res["size_bytes"]

        # 3. Assemble validated metadata
        meta = KnowledgeAreaMeta(
            schema_version="1.0.0",
            id=capsule_slug,
            name=details["title"],
            description=details["description"],
            category=CategoryMeta(
                slug=category_slug,
                name=details["category_name"],
                url=f"{BASE_URL}/libros-pdf-gratis/{category_slug}/",
            ),
            domain=details["domain"],
            language="es",
            portal_source=PortalSource(
                name="InfoLibros",
                url=details["url"],
                scraped_at=datetime.now(timezone.utc).isoformat(),
            ),
            tags=[
                capsule_slug,
                category_slug,
                details["domain"].lower(),
                "infolibros",
                "modular-rag",
            ],
            subtopics_covered=[b.title for b in downloaded_books[:10]],
            cover_image=cover_meta,
            books=downloaded_books,
            rag_spec=RagSpecification(
                rag_id=capsule_slug,
                embedding_model=settings.embedding_model,
                generator_model=settings.generator_model,
                chunk_size=500,
                chunk_overlap=50,
                index_db="index.db",
                total_documents=len(downloaded_books),
            ),
            stats=AreaStats(
                total_books=len(downloaded_books),
                total_pages=total_pages,
                total_size_mb=round(total_bytes / (1024 * 1024), 2),
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        meta_path = capsule_dir / "metadata.json"
        meta.save_json(meta_path)

        return meta, capsule_dir
