"""
Build Dedicated Agriculture, Pest Control & Cultivation Knowledge Capsules
Sources: InfoLibros (temas-varios, oficios)
Enforces:
- <= 50 books per capsule
- Authentic Collage Punk visual identity (bio_punk & geo_punk styles)
- Immediate staging cleanup to safeguard disk space
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from smart_local.knowledge.area_schema import (
    KnowledgeAreaMeta,
    CategoryMeta,
    PortalSource,
    RagSpecification
)
from smart_local.manager.registry import KnowledgeManager
from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
from smart_local.pipeline.catalog_extractor import FrontMatterExtractor
from smart_local.pipeline.catalog_enricher import CatalogEnricher
from smart_local.pipeline.infolibros_scraper import InfoLibrosScraper

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgriculturePackager")

AGRICULTURE_MODULES = [
    {
        "category_slug": "temas-varios",
        "capsule_slug": "control-de-plagas",
        "slug": "control-de-plagas",
        "name": "Control de Plagues i Malalties Agrícoles",
        "domain": "Agronomía y Sanidad Vegetal",
        "style": "bio_punk",
        "description": "Manuals tècnics de Maneig Integrat de Plagues (MIP), control biològic, sanitat vegetal i protecció fitosanitària per a horts i cultius."
    },
    {
        "category_slug": "temas-varios",
        "capsule_slug": "hidroponia",
        "slug": "cultiu-interior-hidroponia",
        "name": "Cultiu en Interior i Hidroponia",
        "domain": "Horticultura Protegida y Tecnología Agrícola",
        "style": "hacker_punk",
        "description": "Tractats de cultiu hidropònic (NFT, arrel flotant, substrat inert), formulació de solucions nutritives i entorns controlats d'interior."
    },
    {
        "category_slug": "oficios",
        "capsule_slug": "jardineria",
        "slug": "jardineria-horticultura-urbana",
        "name": "Jardineria i Horticultura Urbana",
        "domain": "Horticultura y Jardinería",
        "style": "bio_punk",
        "description": "Tècniques de cultiu tant en interior com en exterior, horts urbans en recipients, reproducció vegetal i manteniment de plantes."
    },
    {
        "category_slug": "temas-varios",
        "capsule_slug": "composta",
        "slug": "compostatge-fertilitat-sol",
        "name": "Compostatge i Fertilitat del Sòl",
        "domain": "Ciencias del Suelo y Agroecología",
        "style": "geo_punk",
        "description": "Elaboració de compost, microbiologia del sòl, esmenes orgàniques, biofertilitzants i maneig nutricional de la terra."
    },
    {
        "category_slug": "temas-varios",
        "capsule_slug": "permacultura",
        "slug": "permacultura-agroecologia",
        "name": "Permacultura i Agroecologia",
        "domain": "Diseño Agroecológico",
        "style": "geo_punk",
        "description": "Mètodes de disseny resilient, agricultura regenerativa (Fukuoka, Mollison), horts de baix manteniment i cicles naturals."
    },
    {
        "category_slug": "temas-varios",
        "capsule_slug": "agricultura",
        "slug": "agricultura-general-i-sostenible",
        "name": "Agricultura General i Bones Pràctiques",
        "domain": "Ciencias Agrícolas",
        "style": "geo_punk",
        "description": "Manuals agronòmics generals, bones pràctiques agrícoles (FAO, IICA), agricultura de precisió i sistemes de producció vegetal."
    }
]

def build_agriculture_capsules(target_slugs=None, max_books=50):
    console.print("\n[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]")
    console.print("[bold yellow]🌾 Iniciant Generació de Càpsules d'Agricultura, Plagues i Cultiu[/bold yellow]")
    console.print("[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]\n")

    scraper = InfoLibrosScraper()
    extractor = FrontMatterExtractor()
    enricher = CatalogEnricher()
    packager = KnowledgeCapsulePackager()
    km = KnowledgeManager()

    results = []

    for mod in AGRICULTURE_MODULES:
        if target_slugs and mod["slug"] not in target_slugs and mod["capsule_slug"] not in target_slugs:
            continue

        slug = mod["slug"]
        name = mod["name"]
        cat = mod["category_slug"]
        cap = mod["capsule_slug"]
        domain = mod["domain"]
        style = mod["style"]

        console.print(f"\n[bold green]▶ CÀPSULA: {name} ({slug})[/bold green]")
        console.print(f"  Domini: [cyan]{domain}[/cyan] | Estil Collage Punk: [magenta]{style}[/magenta]")

        # 1. Obtenir llista de llibres de la càpsula
        details = scraper.get_capsule_details(cat, cap)
        books = details.get("books", [])
        if not books:
            console.print("  [red]Cap llibre trobat a la web.[/red]")
            continue

        capped_books = books[:min(len(books), max_books, 50)]
        console.print(f"  Trobats [bold yellow]{len(capped_books)}[/bold yellow] llibres tècnics (màxim permès <= 50).")

        # 2. Descarregar PDFs a disc temporal
        capsule_dir = Path("data/downloads/books") / cat / cap
        capsule_dir.mkdir(parents=True, exist_ok=True)
        downloaded_pdfs = []

        for b_idx, b in enumerate(capped_books, 1):
            title = b["title"]
            pdf_url = b["url"]
            filename = b["filename"]
            target_pdf = capsule_dir / filename

            console.print(f"    ({b_idx}/{len(capped_books)}) Descarregant: [white]{title[:45]}...[/white]")
            if not target_pdf.exists() or target_pdf.stat().st_size < 1024:
                referer = b.get("referer", details["url"])
                res_info = scraper.download_pdf(pdf_url, target_pdf, referer=referer)
                if res_info and target_pdf.exists() and target_pdf.stat().st_size > 1024:
                    downloaded_pdfs.append(target_pdf)
                else:
                    console.print(f"      [dim yellow]Avís: no s'ha pogut descarregar {filename}[/dim yellow]")
            else:
                downloaded_pdfs.append(target_pdf)

        console.print(f"  ✓ Total PDFs preparats a disc: [bold green]{len(downloaded_pdfs)}[/bold green]")
        if not downloaded_pdfs:
            continue

        # 3. Extreure Front-Matter i Generar Catàleg Collage Punk
        console.print(f"  Extraient front-matter de les primeres pàgines de {len(downloaded_pdfs)} volums...")
        covers_dir = capsule_dir / "covers"
        covers_dir.mkdir(parents=True, exist_ok=True)
        images_dir = capsule_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        front_matters = extractor.scan_directory(capsule_dir, max_pages=3)
        enriched_records = []
        total_pages = 0
        total_bytes = 0

        for fm in front_matters:
            if "error" in fm:
                continue
            rec = enricher.enrich_book_front_matter(
                front_matter_data=fm,
                topic_name=name,
                domain=domain,
                covers_dir=covers_dir
            )
            enriched_records.append(rec.model_dump())
            total_pages += rec.total_pages
            total_bytes += rec.file_size_bytes

        active_style = enricher._select_style_for_topic(name, domain)
        banner_file = images_dir / "banner.png"
        if not banner_file.exists():
            banner_file = images_dir / "banner.jpg"
        banner_rel = f"images/{banner_file.name}" if banner_file.exists() else "images/banner.png"

        catalog_data = {
            "topic": name,
            "domain": domain,
            "source_portal": "InfoLibros",
            "category_slug": cat,
            "capsule_slug": cap,
            "canonical_style_id": active_style["id"],
            "style_name": active_style["name"],
            "banner_cover": banner_rel,
            "books": enriched_records
        }
        catalog_path = capsule_dir / "catalog.json"
        catalog_path.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"  ✓ Catàleg Collage Punk generat a: {catalog_path}")

        # 4. Preparar Staging
        stage_dir = Path("data/capsules") / slug
        stage_images = stage_dir / "images"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_images.mkdir(parents=True, exist_ok=True)

        meta = KnowledgeAreaMeta(
            id=slug,
            name=name,
            description=mod["description"],
            category=CategoryMeta(
                slug=cat,
                name=cat.capitalize(),
                url=f"https://infolibros.org/libros-pdf-gratis/{cat}/"
            ),
            domain=domain,
            portal_source=PortalSource(
                name="InfoLibros",
                url=f"https://infolibros.org/libros-pdf-gratis/{cat}/{cap}/"
            ),
            tags=[cap, "agricultura", "cultivo", "plagas", "collage-punk", "smartlocal"],
            rag_spec=RagSpecification(rag_id=slug)
        )
        meta.save_json(stage_dir / "metadata.json")
        if catalog_path.exists():
            shutil.copy2(catalog_path, stage_dir / "catalog.json")
        if covers_dir.exists():
            shutil.copytree(covers_dir, stage_images / "covers", dirs_exist_ok=True)
        if banner_file.exists():
            shutil.copy2(banner_file, stage_images / banner_file.name)

        # 5. Empaquetar a ZIP amb Index i Chunks Vectorials
        console.print(f"  [dim]Vectoritzant i empaquetant {slug}-v1.0.0.zip ({len(downloaded_pdfs)} llibres)...[/dim]")
        zip_output_path = Path("data/packs") / f"{slug}-v1.0.0.zip"
        master_cover = banner_file if banner_file.exists() else None

        final_zip = packager.build_package(
            metadata_source=stage_dir / "metadata.json",
            pdfs_source_dir=capsule_dir,
            output_zip=zip_output_path,
            cover_image_path=master_cover
        )
        console.print(f"  ✓ Paquet empaquetat: [bold green]{final_zip}[/bold green] ({round(final_zip.stat().st_size / (1024*1024), 2)} MB)")

        # 6. Importar al RAG
        console.print(f"  [dim]Important i indexant al motor RAG local...[/dim]")
        resolved_id = km.packager.import_package(final_zip)
        km.load_rag(resolved_id)
        store = km.get_store(resolved_id)
        manifest = store.get_manifest()

        results.append({
            "slug": slug,
            "name": name,
            "books": len(enriched_records),
            "pages": total_pages,
            "chunks": manifest.total_chunks if manifest else 0,
            "zip_size_mb": round(final_zip.stat().st_size / (1024*1024), 2)
        })

        # Neteja de seguretat de disc
        shutil.rmtree(capsule_dir, ignore_errors=True)
        shutil.rmtree(stage_dir, ignore_errors=True)

    table = Table(title="🌾 Càpsules d'Agricultura, Plagues i Cultiu Creades i Indexades")
    table.add_column("Càpsula", style="cyan bold")
    table.add_column("Nom", style="white")
    table.add_column("Llibres", style="yellow")
    table.add_column("Pàgines", style="magenta")
    table.add_column("Chunks", style="green")
    table.add_column("Mida ZIP", style="bright_yellow")

    for r in results:
        table.add_row(r["slug"], r["name"], str(r["books"]), str(r["pages"]), str(r["chunks"]), f"{r['zip_size_mb']} MB")
    console.print(table)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsules", nargs="+", default=None)
    parser.add_argument("--books", type=int, default=50)
    args = parser.parse_args()

    build_agriculture_capsules(target_slugs=args.capsules, max_books=args.books)
