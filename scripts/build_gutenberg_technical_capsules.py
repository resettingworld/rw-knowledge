"""
Project Gutenberg Technical Knowledge Indexer & Capsule Packager
Downloads non-fiction technical & scientific books from Project Gutenberg,
extracts academic front-matter, enriches catalog with Collage Punk art direction,
extracts diagrams, vectorizes with EmbeddingGemma, and packages into installable RAG capsules.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from smart_local.pipeline.gutenberg_scraper import ProjectGutenbergScraper, TECHNICAL_BOOKSHELVES_MAP
from smart_local.pipeline.catalog_extractor import FrontMatterExtractor
from smart_local.pipeline.catalog_enricher import CatalogEnricher
from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
from smart_local.manager.registry import KnowledgeManager
from smart_local.knowledge.area_schema import KnowledgeAreaMeta, CategoryMeta, PortalSource, RagSpecification

console = Console()

def run_gutenberg_pipeline(
    shelf_ids: Optional[List[str]] = None,
    books_per_topic: int = 5,
    download_all_in_shelf: bool = False
):
    console.print("\n[bold magenta]══════════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold yellow]⚡ PROJECT GUTENBERG: INDEXACIÓ DE CONEIXAMENT TÈCNIC I CIENTÍFIC[/bold yellow]")
    console.print("[bold magenta]══════════════════════════════════════════════════════════════════[/bold magenta]\n")

    scraper = ProjectGutenbergScraper()
    extractor = FrontMatterExtractor()
    enricher = CatalogEnricher()
    packager = KnowledgeCapsulePackager()
    km = KnowledgeManager()

    target_shelves = shelf_ids or ["671", "672", "667", "681", "673"]

    results = []

    for idx, s_id in enumerate(target_shelves, start=1):
        if s_id not in TECHNICAL_BOOKSHELVES_MAP:
            console.print(f"[yellow]Avís: Bookshelf {s_id} no està al mapa de coneixement tècnic. Ometent.[/yellow]")
            continue

        meta = TECHNICAL_BOOKSHELVES_MAP[s_id]
        slug = f"gutenberg-{meta['slug']}"
        name = f"Gutenberg: {meta['name']}"
        domain = meta["domain"]
        style_key = meta["style_key"]

        console.print(f"\n[bold cyan]▶ [{idx}/{len(target_shelves)}] BOOKSHELF {s_id}: {name}[/bold cyan]")
        console.print(f"  [dim]Domini: {domain} | Estil: {style_key}[/dim]")

        # 1. Obtenir llista de llibres del bookshelf
        try:
            available_books = scraper.get_bookshelf_books(s_id)
        except Exception as e:
            console.print(f"  [red]Error obtenint llibres de Gutenberg: {e}[/red]")
            continue

        console.print(f"  Trobats [bold green]{len(available_books)}[/bold green] tractats tècnics disponibles a Gutenberg.")
        target_books = available_books if download_all_in_shelf else available_books[:books_per_topic]

        # 2. Descarregar i convertir a PDF
        capsule_dir = Path("data/downloads/books/gutenberg") / meta["slug"]
        capsule_dir.mkdir(parents=True, exist_ok=True)

        downloaded_pdfs = []
        for b_idx, b in enumerate(target_books, start=1):
            clean_fname = re.sub(r'[^a-zA-Z0-9_-]', '_', f"gutenberg_{b['ebook_id']}_{b['title'][:30]}") + ".pdf"
            dest_pdf = capsule_dir / clean_fname
            
            console.print(f"    [dim]({b_idx}/{len(target_books)}) Obtenint: {b['title'][:45]}...[/dim]")
            res = scraper.download_and_convert_to_pdf(
                ebook_id=b["ebook_id"],
                dest_pdf_path=dest_pdf,
                preferred_title=b["title"],
                author=b["author"]
            )
            if res and dest_pdf.exists():
                downloaded_pdfs.append(dest_pdf)
            else:
                console.print(f"    [yellow]Avís: no s'ha pogut descarregar/convertir ebook {b['ebook_id']}[/yellow]")

        console.print(f"  ✓ Total PDFs tècnics preparats: [bold green]{len(downloaded_pdfs)}[/bold green]")
        if not downloaded_pdfs:
            continue

        # 3. Preparar directoris d'imatges
        images_dir = capsule_dir / "images"
        covers_dir = images_dir / "covers"
        covers_dir.mkdir(parents=True, exist_ok=True)

        # 4. Extracció de Front-Matter
        console.print(f"  [dim]Extraient front-matter de les primeres pàgines de {len(downloaded_pdfs)} volums...[/dim]")
        front_matters = [extractor.extract_from_pdf(p, max_pages=3) for p in downloaded_pdfs]

        # 5. Generació del Catàleg amb Estil Collage Punk
        console.print("  [dim]Generant metadades i portades Collage Punk...[/dim]")
        enriched_records = []
        total_pages = 0
        total_bytes = 0

        for fm in front_matters:
            if "error" in fm:
                continue
            rec = enricher.enrich_book_front_matter(
                front_matter_data=fm,
                topic_name=meta["name"],
                domain=domain,
                covers_dir=covers_dir
            )
            enriched_records.append(rec.model_dump())
            total_pages += rec.total_pages
            total_bytes += rec.file_size_bytes

        active_style = enricher._select_style_for_topic(meta["name"], domain)
        banner_file = images_dir / "banner.jpg"
        if not banner_file.exists():
            banner_file = images_dir / "banner.png"
        banner_rel = f"images/{banner_file.name}" if banner_file.exists() else "images/banner.jpg"

        catalog_data = {
            "topic": name,
            "domain": domain,
            "source_portal": "Project Gutenberg",
            "bookshelf_id": s_id,
            "canonical_style_id": active_style["id"],
            "canonical_style_name": active_style["name"],
            "banner_image": {
                "rel_path": banner_rel,
                "style_id": active_style["id"],
                "style_name": active_style["name"],
                "prompt_template": active_style["template"]
            },
            "total_books": len(enriched_records),
            "total_pages": total_pages,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "books": enriched_records
        }

        catalog_path = capsule_dir / "catalog.json"
        catalog_path.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"  ✓ Catàleg Collage Punk generat a: {catalog_path}")

        # 6. Preparar Stage per a Empaquetament
        stage_dir = Path("data/capsules") / slug
        stage_images = stage_dir / "images"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_images.mkdir(parents=True, exist_ok=True)

        meta_obj = KnowledgeAreaMeta(
            id=slug,
            name=name,
            description=meta["description"],
            category=CategoryMeta(
                slug="gutenberg-technical",
                name="Project Gutenberg Technical Knowledge",
                url=f"https://www.gutenberg.org/ebooks/bookshelf/{s_id}"
            ),
            domain=domain,
            portal_source=PortalSource(
                name="Project Gutenberg",
                url=f"https://www.gutenberg.org/ebooks/bookshelf/{s_id}"
            ),
            tags=[slug, "gutenberg", "technical", "non-fiction", "collage-punk", "modular-rag"],
            rag_spec=RagSpecification(rag_id=slug)
        )
        meta_obj.save_json(stage_dir / "metadata.json")
        shutil.copy2(catalog_path, stage_dir / "catalog.json")
        shutil.copytree(covers_dir, stage_images / "covers", dirs_exist_ok=True)
        if banner_file.exists():
            shutil.copy2(banner_file, stage_images / banner_file.name)

        # 7. Empaquetar a ZIP
        console.print(f"  [dim]Vectoritzant i empaquetant {slug}-v1.0.0.zip...[/dim]")
        zip_output_path = Path("data/packs") / f"{slug}-v1.0.0.zip"
        master_cover_for_pack = banner_file if banner_file.exists() else (covers_dir / f"{enriched_records[0]['id']}.png" if enriched_records else None)

        final_zip = packager.build_package(
            metadata_source=stage_dir / "metadata.json",
            pdfs_source_dir=capsule_dir,
            output_zip=zip_output_path,
            cover_image_path=master_cover_for_pack
        )
        console.print(f"  ✓ Paquet empaquetat: [bold green]{final_zip}[/bold green] ({round(final_zip.stat().st_size / (1024*1024), 2)} MB)")

        # 8. Instal·lar al motor RAG
        console.print(f"  [dim]Important i indexant al motor RAG local...[/dim]")
        resolved_id = km.packager.import_package(final_zip)
        km.load_rag(resolved_id)
        store = km.get_store(resolved_id)
        manifest = store.get_manifest()

        results.append({
            "shelf_id": s_id,
            "slug": slug,
            "name": meta["name"],
            "books": len(enriched_records),
            "pages": total_pages,
            "chunks": manifest.total_chunks if manifest else 0,
            "figures": manifest.total_images_extracted if manifest else 0,
            "zip_size_mb": round(final_zip.stat().st_size / (1024*1024), 2)
        })

    # Resum final
    table = Table(title="🏛️ Project Gutenberg: Paquets de Coneixement Tècnic Indexats")
    table.add_column("Bookshelf ID", style="dim")
    table.add_column("Càpsula", style="cyan bold")
    table.add_column("Disciplina", style="white")
    table.add_column("Llibres", style="yellow")
    table.add_column("Pàgines", style="magenta")
    table.add_column("Chunks", style="green")
    table.add_column("Figures IA", style="blue")
    table.add_column("Mida ZIP", style="bright_yellow")

    for r in results:
        table.add_row(
            r["shelf_id"],
            r["slug"],
            r["name"],
            str(r["books"]),
            str(r["pages"]),
            str(r["chunks"]),
            str(r["figures"]),
            f"{r['zip_size_mb']} MB"
        )
    console.print(table)
    console.print("\n[bold green]✓ Indexació de Project Gutenberg completada amb èxit![/bold green]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Gutenberg Technical Knowledge Indexer")
    parser.add_argument("--shelves", nargs="+", default=["671", "672", "667", "681", "673"], help="Bookshelf IDs a processar")
    parser.add_argument("--books", type=int, default=5, help="Llibres per bookshelf")
    parser.add_argument("--all", action="store_true", help="Descarregar tots els llibres del bookshelf")
    args = parser.parse_args()

    run_gutenberg_pipeline(
        shelf_ids=args.shelves,
        books_per_topic=args.books,
        download_all_in_shelf=args.all
    )
