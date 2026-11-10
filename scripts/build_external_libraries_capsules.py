"""
Master Pipeline: Build Technical Knowledge Capsules from textos.info and planetalibro.net
Strictly enforces:
1. Maximum 50 books per RAG capsule (<= 50).
2. Only non-fiction, technical, scientific, philosophical, historical & strategic knowledge.
3. Fully canonical Collage Punk covers and manifest styling.
4. Immediate staging cleanup to safeguard local disk space.
"""

import argparse
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
from smart_local.pipeline.textos_info_scraper import TextosInfoScraper, TEXTOS_TAGS
from smart_local.pipeline.planetalibro_scraper import PlanetaLibroScraper, PLANETALIBRO_TAGS

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExternalLibrariesPackager")

def run_textos_info_pipeline(tags=None, max_books_per_capsule=50):
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print("[bold yellow]🚀 Iniciant Pipeline de Coneixement: textos.info (Max 50 llibres/càpsula)[/bold yellow]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n")

    scraper = TextosInfoScraper()
    extractor = FrontMatterExtractor()
    enricher = CatalogEnricher()
    packager = KnowledgeCapsulePackager()
    km = KnowledgeManager()

    target_tags = tags or list(TEXTOS_TAGS.keys())
    results = []

    for idx, tag in enumerate(target_tags, 1):
        config = TEXTOS_TAGS.get(tag, {
            "name": tag.capitalize(),
            "description": f"Compendio de obras técnicas y ensayos sobre {tag}",
            "domain": "Conocimiento General",
            "style": "hacker_punk"
        })

        slug = f"textos-info-{tag}"
        console.print(f"\n[bold green]▶ [{idx}/{len(target_tags)}] TEXTOS.INFO: {config['name']} ({tag})[/bold green]")
        console.print(f"  Domini: [cyan]{config['domain']}[/cyan] | Estil Collage Punk: [magenta]{config['style']}[/magenta]")

        # 1. Discover books (capped strictly <= 50)
        books = scraper.get_books_for_tag(tag, max_books=min(max_books_per_capsule, 50))
        console.print(f"  Trobats [bold yellow]{len(books)}[/bold yellow] llibres tècnics/assagístics disponibles.")

        if not books:
            console.print("  [dim]Cap llibre trobat, ometent...[/dim]")
            continue

        # 2. Download PDFs
        capsule_dir = Path("data/downloads/books/textos_info") / tag
        capsule_dir.mkdir(parents=True, exist_ok=True)
        downloaded_pdfs = []

        for b_idx, b in enumerate(books, 1):
            console.print(f"    ({b_idx}/{len(books)}) Descarregant: [white]{b.title[:45]}...[/white]")
            p = scraper.download_book(b, capsule_dir)
            if p and p.exists() and p.stat().st_size > 1024:
                downloaded_pdfs.append(p)

        console.print(f"  ✓ Total PDFs preparats a disc: [bold green]{len(downloaded_pdfs)}[/bold green]")
        if not downloaded_pdfs:
            continue

        # 3. Extract front-matter and generate Collage Punk catalog
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
                topic_name=config["name"],
                domain=config["domain"],
                covers_dir=covers_dir
            )
            enriched_records.append(rec.model_dump())
            total_pages += rec.total_pages
            total_bytes += rec.file_size_bytes

        active_style = enricher._select_style_for_topic(config["name"], config["domain"])
        banner_file = images_dir / "banner.png"
        if not banner_file.exists():
            banner_file = images_dir / "banner.jpg"
        banner_rel = f"images/{banner_file.name}" if banner_file.exists() else "images/banner.png"

        catalog_data = {
            "topic": config["name"],
            "domain": config["domain"],
            "source_portal": "textos.info",
            "tag": tag,
            "canonical_style_id": active_style["id"],
            "style_name": active_style["name"],
            "banner_cover": banner_rel,
            "books": enriched_records
        }
        catalog_path = capsule_dir / "catalog.json"
        catalog_path.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"  ✓ Catàleg Collage Punk generat a: {catalog_path}")

        # 4. Prepare Staging Area
        stage_dir = Path("data/capsules") / slug
        stage_images = stage_dir / "images"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_images.mkdir(parents=True, exist_ok=True)

        meta = KnowledgeAreaMeta(
            id=slug,
            name=config["name"],
            description=config["description"],
            category=CategoryMeta(
                slug=tag,
                name=config["name"],
                url=f"https://www.textos.info/libros/etiqueta/{tag}"
            ),
            domain=config["domain"],
            portal_source=PortalSource(
                name="textos.info",
                url=f"https://www.textos.info/libros/etiqueta/{tag}"
            ),
            tags=[tag, "textos-info", "collage-punk", "non-fiction", "smartlocal"],
            rag_spec=RagSpecification(rag_id=slug)
        )
        meta.save_json(stage_dir / "metadata.json")
        if catalog_path.exists():
            shutil.copy2(catalog_path, stage_dir / "catalog.json")
        if covers_dir.exists():
            shutil.copytree(covers_dir, stage_images / "covers", dirs_exist_ok=True)
        if banner_file.exists():
            shutil.copy2(banner_file, stage_images / banner_file.name)

        # 5. Pack ZIP & Multimodal Vector Store
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

        # 6. Import to RAG
        console.print(f"  [dim]Important i indexant al motor RAG local...[/dim]")
        resolved_id = km.packager.import_package(final_zip)
        km.load_rag(resolved_id)
        store = km.get_store(resolved_id)
        manifest = store.get_manifest()

        results.append({
            "capsule": slug,
            "name": config["name"],
            "books": len(enriched_records),
            "chunks": manifest.total_chunks if manifest else 0,
            "zip_size_mb": round(final_zip.stat().st_size / (1024*1024), 2)
        })

        # Safeguard disk space
        shutil.rmtree(capsule_dir, ignore_errors=True)
        shutil.rmtree(stage_dir, ignore_errors=True)

    return results

def run_planetalibro_pipeline(tags=None, max_books_per_capsule=50):
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print("[bold yellow]🚀 Iniciant Pipeline de Coneixement: PlanetaLibro (Max 50 llibres/càpsula)[/bold yellow]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n")

    scraper = PlanetaLibroScraper()
    extractor = FrontMatterExtractor()
    enricher = CatalogEnricher()
    packager = KnowledgeCapsulePackager()
    km = KnowledgeManager()

    target_tags = tags or list(PLANETALIBRO_TAGS.keys())
    results = []

    for idx, tag in enumerate(target_tags, 1):
        config = PLANETALIBRO_TAGS.get(tag, {
            "name": tag.capitalize(),
            "description": f"Compendio de obras fundamentales sobre {tag}",
            "domain": "Conocimiento General",
            "style": "anarcho_social_punk"
        })

        slug = f"planetalibro-{tag}"
        console.print(f"\n[bold green]▶ [{idx}/{len(target_tags)}] PLANETALIBRO: {config['name']} ({tag})[/bold green]")
        console.print(f"  Domini: [cyan]{config['domain']}[/cyan] | Estil Collage Punk: [magenta]{config['style']}[/magenta]")

        # 1. Discover books (capped strictly <= 50)
        books = scraper.get_books_for_tag(tag, max_books=min(max_books_per_capsule, 50))
        console.print(f"  Trobats [bold yellow]{len(books)}[/bold yellow] llibres no-ficció/tècnics disponibles.")

        if not books:
            console.print("  [dim]Cap llibre trobat, ometent...[/dim]")
            continue

        # 2. Download and convert books to PDF
        capsule_dir = Path("data/downloads/books/planetalibro") / tag
        capsule_dir.mkdir(parents=True, exist_ok=True)
        downloaded_pdfs = []

        for b_idx, b in enumerate(books, 1):
            console.print(f"    ({b_idx}/{len(books)}) Convertint: [white]{b.title[:45]}...[/white]")
            p = scraper.download_and_convert_book(b, capsule_dir)
            if p and p.exists() and p.stat().st_size > 1024:
                downloaded_pdfs.append(p)

        console.print(f"  ✓ Total PDFs tècnics preparats: [bold green]{len(downloaded_pdfs)}[/bold green]")
        if not downloaded_pdfs:
            continue

        # 3. Extract front-matter and generate Collage Punk catalog
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
                topic_name=config["name"],
                domain=config["domain"],
                covers_dir=covers_dir
            )
            enriched_records.append(rec.model_dump())
            total_pages += rec.total_pages
            total_bytes += rec.file_size_bytes

        active_style = enricher._select_style_for_topic(config["name"], config["domain"])
        banner_file = images_dir / "banner.png"
        if not banner_file.exists():
            banner_file = images_dir / "banner.jpg"
        banner_rel = f"images/{banner_file.name}" if banner_file.exists() else "images/banner.png"

        catalog_data = {
            "topic": config["name"],
            "domain": config["domain"],
            "source_portal": "PlanetaLibro",
            "tag": tag,
            "canonical_style_id": active_style["id"],
            "style_name": active_style["name"],
            "banner_cover": banner_rel,
            "books": enriched_records
        }
        catalog_path = capsule_dir / "catalog.json"
        catalog_path.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"  ✓ Catàleg Collage Punk generat a: {catalog_path}")

        # 4. Prepare Staging Area
        stage_dir = Path("data/capsules") / slug
        stage_images = stage_dir / "images"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_images.mkdir(parents=True, exist_ok=True)

        meta = KnowledgeAreaMeta(
            id=slug,
            name=config["name"],
            description=config["description"],
            category=CategoryMeta(
                slug=tag,
                name=config["name"],
                url=f"https://planetalibro.net/tema/{tag}"
            ),
            domain=config["domain"],
            portal_source=PortalSource(
                name="PlanetaLibro",
                url=f"https://planetalibro.net/tema/{tag}"
            ),
            tags=[tag, "planetalibro", "collage-punk", "non-fiction", "smartlocal"],
            rag_spec=RagSpecification(rag_id=slug)
        )
        meta.save_json(stage_dir / "metadata.json")
        if catalog_path.exists():
            shutil.copy2(catalog_path, stage_dir / "catalog.json")
        if covers_dir.exists():
            shutil.copytree(covers_dir, stage_images / "covers", dirs_exist_ok=True)
        if banner_file.exists():
            shutil.copy2(banner_file, stage_images / banner_file.name)

        # 5. Pack ZIP & Multimodal Vector Store
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

        # 6. Import to RAG
        console.print(f"  [dim]Important i indexant al motor RAG local...[/dim]")
        resolved_id = km.packager.import_package(final_zip)
        km.load_rag(resolved_id)
        store = km.get_store(resolved_id)
        manifest = store.get_manifest()

        results.append({
            "capsule": slug,
            "name": config["name"],
            "books": len(enriched_records),
            "chunks": manifest.total_chunks if manifest else 0,
            "zip_size_mb": round(final_zip.stat().st_size / (1024*1024), 2)
        })

        # Safeguard disk space
        shutil.rmtree(capsule_dir, ignore_errors=True)
        shutil.rmtree(stage_dir, ignore_errors=True)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["textos_info", "planetalibro", "all"], default="all")
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--books", type=int, default=50)
    args = parser.parse_args()

    all_results = []
    if args.source in ["textos_info", "all"]:
        all_results.extend(run_textos_info_pipeline(tags=args.tags, max_books_per_capsule=args.books))
    if args.source in ["planetalibro", "all"]:
        all_results.extend(run_planetalibro_pipeline(tags=args.tags, max_books_per_capsule=args.books))

    table = Table(title="🎉 Paquets de Coneixement Externs Creads (<= 50 llibres/càpsula)")
    table.add_column("Càpsula", style="cyan bold")
    table.add_column("Nom", style="white")
    table.add_column("Llibres", style="yellow")
    table.add_column("Chunks", style="green")
    table.add_column("Mida ZIP", style="bright_yellow")

    for r in all_results:
        table.add_row(r["capsule"], r["name"], str(r["books"]), str(r["chunks"]), f"{r['zip_size_mb']} MB")
    console.print(table)
