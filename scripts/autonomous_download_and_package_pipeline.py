"""
Autonomous Download and Knowledge Capsule Builder Pipeline
Collage Punk Aesthetic
Scrapes InfoLibros subtopics, downloads books with referer validation,
extracts front-matters, generates Collage Punk covers and enriched catalogs,
extracts diagrams, vectorizes with EmbeddingGemma, packages into .zip, and registers in RAG engine.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from smart_local.config import settings
from smart_local.pipeline.infolibros_scraper import InfoLibrosScraper
from smart_local.pipeline.catalog_extractor import FrontMatterExtractor
from smart_local.pipeline.catalog_enricher import CatalogEnricher
from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
from smart_local.pipeline.processor import MultimodalProcessor
from smart_local.pipeline.image_analyzer import ImageConceptAnalyzer
from smart_local.manager.registry import KnowledgeManager
from smart_local.knowledge.area_schema import KnowledgeAreaMeta, CategoryMeta, PortalSource, RagSpecification
from smart_local.models.embedding import EmbeddingGemma

console = Console()

def run_pipeline(
    max_books_per_capsule: Optional[int] = None,
    download_all: bool = True,
    categories: Optional[List[str]] = None,
    target_capsule_slugs: Optional[List[str]] = None
):
    console.print("\n[bold magenta]══════════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold yellow]⚡ INICIANT PIPELINE: DESCÀRREGA DE TOTS ELS LLIBRES I CREACIÓ DE PAQUETS COLLAGE PUNK[/bold yellow]")
    console.print("[bold magenta]══════════════════════════════════════════════════════════════════[/bold magenta]\n")

    scraper = InfoLibrosScraper()
    extractor = FrontMatterExtractor()
    enricher = CatalogEnricher()
    packager = KnowledgeCapsulePackager()
    km = KnowledgeManager()

    # Discover capsules
    cats_to_scan = categories or ["ciencias", "informatica"]
    capsules_to_process = []

    # Priority order for initial capsules
    priority_slugs = [
        "ciencias-de-la-tierra",
        "ciencias-aplicadas",
        "ciencias-naturales",
        "ciencias-sociales",
        "algoritmo",
        "inteligencia-artificial",
        "machine-learning",
        "seguridad-informatica",
        "bases-de-datos",
        "redes-informaticas"
    ]

    for cat in cats_to_scan:
        cat_caps = scraper.list_capsules_in_category(cat)
        domain = "Ciencias Exactas y Naturales" if cat == "ciencias" else "Tecnología e Informática"
        for c in cat_caps:
            slug = c["capsule_slug"]
            if target_capsule_slugs and slug not in target_capsule_slugs:
                continue

            # Assign style key
            if "tierra" in slug:
                style_key = "geo_punk"
            elif "aplicad" in slug or "robot" in slug or "automatiz" in slug:
                style_key = "cyber_mechanic_punk"
            elif "natural" in slug or "bio" in slug or "ecolog" in slug:
                style_key = "bio_punk"
            elif "social" in slug:
                style_key = "anarcho_social_punk"
            elif cat == "informatica":
                style_key = "hacker_punk"
            else:
                style_key = "general_punk"

            capsules_to_process.append({
                "category_slug": cat,
                "capsule_slug": slug,
                "name": c["name"],
                "domain": domain,
                "description": f"Col·lecció completa de llibres i manuals de {c['name']} (Estil Collage Punk DIY).",
                "style_key": style_key
            })

    # Sort priority capsules first
    capsules_to_process.sort(
        key=lambda x: priority_slugs.index(x["capsule_slug"]) if x["capsule_slug"] in priority_slugs else 999
    )

    console.print(f"Càpsules a processar: [bold green]{len(capsules_to_process)}[/bold green] (Categories: {', '.join(cats_to_scan)})")
    if download_all:
        console.print("[bold yellow]Modus: DESCÀRREGA TOTAL DE TOTS ELS LLIBRES DISPONIBLES[/bold yellow]\n")

    results = []

    for idx, item in enumerate(capsules_to_process, start=1):
        cat = item["category_slug"]
        slug = item["capsule_slug"]
        name = item["name"]
        domain = item["domain"]
        style_key = item["style_key"]

        console.print(f"\n[bold cyan]▶ [{idx}/{len(capsules_to_process)}] CÀPSULA: {name} ({cat}/{slug})[/bold cyan]")

        # 1. Obtenir informació i enllaços de descàrrega des de la web
        capsule_dir = Path("data/downloads/books") / cat / slug
        capsule_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"  [dim]Obtenint índex de la web d'InfoLibros per a {slug}...[/dim]")
        capsule_info = scraper.get_capsule_details(category_slug=cat, capsule_slug=slug)
        available_books = capsule_info.get("books", [])
        console.print(f"  Trobats [bold green]{len(available_books)}[/bold green] llibres disponibles a la web.")

        target_books = available_books if (download_all or max_books_per_capsule is None) else available_books[:max_books_per_capsule]
        console.print(f"  [bold]Objectiu: Descarregar i catalogar TOTS els {len(target_books)} llibres.[/bold]")

        # 2. Descarregar PDFs amb gestió de capçaleres i Referer
        downloaded_pdfs = []
        for b_idx, b in enumerate(target_books, start=1):
            target_file = capsule_dir / b["filename"]
            if not target_file.exists():
                console.print(f"    [dim]({b_idx}/{len(target_books)}) Descarregant: {b['title'][:40]}...[/dim]")
                dl_res = scraper.download_pdf(pdf_url=b["url"], target_path=target_file, referer=b["referer"])
                if dl_res:
                    downloaded_pdfs.append(target_file)
                else:
                    console.print(f"    [yellow]Avís: no s'ha pogut descarregar {b['filename']}[/yellow]")
            else:
                downloaded_pdfs.append(target_file)

        console.print(f"  ✓ Total PDFs preparats a disc: [bold green]{len(downloaded_pdfs)}[/bold green]")

        # 3. Comprovar o generar imatges Collage Punk
        images_dir = capsule_dir / "images"
        covers_dir = images_dir / "covers"
        covers_dir.mkdir(parents=True, exist_ok=True)

        # 4. Extracció de Front-Matter
        console.print(f"  [dim]Analitzant les primeres pàgines de crèdits de {len(downloaded_pdfs)} PDFs...[/dim]")
        front_matters = [extractor.extract_from_pdf(p, max_pages=3) for p in downloaded_pdfs]

        # 5. Enriquiment del Catàleg amb Estil Collage Punk
        console.print("  [dim]Generant metadades acadèmiques i portades Collage Punk...[/dim]")
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
        banner_file = images_dir / "banner.jpg"
        if not banner_file.exists():
            banner_file = images_dir / "banner.png"
        banner_rel = f"images/{banner_file.name}" if banner_file.exists() else "images/banner.jpg"

        catalog_data = {
            "topic": name,
            "domain": domain,
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
        console.print(f"  ✓ Catàleg Collage Punk generat amb {len(enriched_records)} llibres a: {catalog_path}")

        # 6. Preparar Stage per a Empaquetament
        stage_dir = Path("data/capsules") / slug
        stage_images = stage_dir / "images"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_images.mkdir(parents=True, exist_ok=True)

        meta = KnowledgeAreaMeta(
            id=slug,
            name=name,
            description=item["description"],
            category=CategoryMeta(
                slug=cat,
                name=cat.capitalize(),
                url=f"https://infolibros.org/libros-pdf-gratis/{cat}/"
            ),
            domain=domain,
            portal_source=PortalSource(
                name="InfoLibros",
                url=f"https://infolibros.org/libros-pdf-gratis/{cat}/{slug}/"
            ),
            tags=[slug, cat, "collage-punk", "diy-zine", "modular-rag", "smartlocal"],
            rag_spec=RagSpecification(rag_id=slug)
        )
        meta.save_json(stage_dir / "metadata.json")
        shutil.copy2(catalog_path, stage_dir / "catalog.json")
        shutil.copytree(covers_dir, stage_images / "covers", dirs_exist_ok=True)
        if banner_file.exists():
            shutil.copy2(banner_file, stage_images / banner_file.name)

        # 7. Empaquetar a ZIP amb MultimodalProcessor & Concept Image Analyzer
        console.print(f"  [dim]Vectoritzant i empaquetant {slug}-v1.0.0.zip ({len(downloaded_pdfs)} llibres)...[/dim]")
        zip_output_path = Path("data/packs") / f"{slug}-v1.0.0.zip"
        
        master_cover_for_pack = banner_file if banner_file.exists() else (covers_dir / f"{enriched_records[0]['id']}.png" if enriched_records else None)

        final_zip = packager.build_package(
            metadata_source=stage_dir / "metadata.json",
            pdfs_source_dir=capsule_dir,
            output_zip=zip_output_path,
            cover_image_path=master_cover_for_pack
        )
        console.print(f"  ✓ Paquet empaquetat: [bold green]{final_zip}[/bold green] ({round(final_zip.stat().st_size / (1024*1024), 2)} MB)")

        # 8. Instal·lar i carregar al motor RAG de SmartLocal
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
            "figures": manifest.total_images_extracted if manifest else 0,
            "zip_size_mb": round(final_zip.stat().st_size / (1024*1024), 2)
        })

    # Resum final
    table = Table(title="🎉 Paquets de Coneixement Collage Punk Creats i Instal·lats")
    table.add_column("Càpsula", style="cyan bold")
    table.add_column("Nom", style="white")
    table.add_column("Llibres", style="yellow")
    table.add_column("Pàgines", style="magenta")
    table.add_column("Chunks", style="green")
    table.add_column("Figures IA", style="blue")
    table.add_column("Mida ZIP", style="bright_yellow")

    for r in results:
        table.add_row(
            r["slug"],
            r["name"],
            str(r["books"]),
            str(r["pages"]),
            str(r["chunks"]),
            str(r["figures"]),
            f"{r['zip_size_mb']} MB"
        )
    console.print(table)
    console.print("\n[bold green]✓ Pipeline completat amb èxit![/bold green]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Download and Knowledge Capsule Builder")
    parser.add_argument("--books", type=int, default=None, help="Llibres per càpsula (default: tots)")
    parser.add_argument("--all", action="store_true", default=True, help="Descarregar tots els llibres disponibles a cada càpsula")
    parser.add_argument("--categories", nargs="+", default=["ciencias", "informatica"], help="Categories a processar")
    parser.add_argument("--capsules", nargs="+", default=None, help="Càpsules específiques (ex: ciencias-de-la-tierra algoritmo)")
    args = parser.parse_args()

    run_pipeline(
        max_books_per_capsule=args.books,
        download_all=(args.books is None or args.all),
        categories=args.categories,
        target_capsule_slugs=args.capsules
    )
