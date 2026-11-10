"""
Build and Load all 5 Modular Knowledge Blocks into SmartLocal RAG Engine
Executes:
1. Verification of PDFs
2. AI Front-matter extraction
3. AI Catalog enrichment (short/long desc, regional scope, normatives, canonical covers)
4. Conceptual image analysis & multimodal vectorization
5. Package into standard ZIP (<capsule>-v1.0.0.zip)
6. Register into SmartLocal RAG engine and load into memory on-demand
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from smart_local.config import settings
from smart_local.pipeline.catalog_extractor import FrontMatterExtractor
from smart_local.pipeline.catalog_enricher import CatalogEnricher
from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
from smart_local.pipeline.image_analyzer import ImageConceptAnalyzer
from smart_local.pipeline.processor import MultimodalProcessor
from smart_local.manager.registry import KnowledgeManager
from smart_local.knowledge.area_schema import KnowledgeAreaMeta, CategoryMeta, PortalSource, RagSpecification
from smart_local.models.embedding import EmbeddingGemma

console = Console()

MODULAR_BLOCKS = [
    {
        "category_slug": "ciencias",
        "capsule_slug": "ciencias-de-la-tierra",
        "name": "Ciencias de la Tierra",
        "domain": "Ciencias Exactas y Naturales",
        "description": "Geologia, tectonica de plaques, sismologia, hidrologia i dinamica terrestre."
    },
    {
        "category_slug": "ciencias",
        "capsule_slug": "ciencias-aplicadas",
        "name": "Ciencias Aplicadas",
        "domain": "Ciencias Exactas y Naturales",
        "description": "Metodologia cientifica, tecnologia aplicada, administracio i enginyeria de processos."
    },
    {
        "category_slug": "ciencias",
        "capsule_slug": "ciencias-naturales",
        "name": "Ciencias Naturales",
        "domain": "Ciencias Exactas y Naturales",
        "description": "Ecosistemes, meteorologia, medi ambient, biodiversitat i ciencies planetaries."
    },
    {
        "category_slug": "ciencias",
        "capsule_slug": "ciencias-sociales",
        "name": "Ciencias Sociales",
        "domain": "Ciencias Sociales y Humanidades",
        "description": "Sociologia, relacions internacionals, comunicacio, metode social i politiques publiques."
    },
    {
        "category_slug": "informatica",
        "capsule_slug": "algoritmo",
        "name": "Algoritmos y Programación",
        "domain": "Tecnología e Informática",
        "description": "Algoritmia, estructures de dades, pseudocodi, complexitat computacional i programacio."
    }
]

def process_and_load_block(
    block: Dict[str, str],
    max_books: int = 5,
    extractor: FrontMatterExtractor = None,
    enricher: CatalogEnricher = None,
    packager: KnowledgeCapsulePackager = None,
    km: KnowledgeManager = None
) -> Dict[str, Any]:
    cat = block["category_slug"]
    slug = block["capsule_slug"]
    name = block["name"]
    domain = block["domain"]

    input_dir = Path("data/downloads/books") / cat / slug
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # 1. Verified PDFs
    all_pdfs = sorted(list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF")))
    target_pdfs = all_pdfs[:max_books] if max_books else all_pdfs

    # 2. Extract Front-Matter
    front_matters = [extractor.extract_from_pdf(p, max_pages=2) for p in target_pdfs]

    # 3. Enrich Catalog & Render Covers
    covers_dir = input_dir / "images" / "covers"
    covers_dir.mkdir(parents=True, exist_ok=True)

    enriched_records = []
    total_pages = 0
    total_bytes = 0

    for fm in front_matters:
        rec = enricher.enrich_book_front_matter(
            front_matter_data=fm,
            topic_name=name,
            domain=domain,
            covers_dir=covers_dir
        )
        enriched_records.append(rec.model_dump())
        total_pages += rec.total_pages
        total_bytes += rec.file_size_bytes

    catalog_data = {
        "topic": name,
        "domain": domain,
        "canonical_style_id": "smartlocal-academic-minimalist-v1",
        "total_books": len(enriched_records),
        "total_pages": total_pages,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "books": enriched_records
    }
    catalog_path = input_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4. Stage & Vectorize
    stage_dir = Path("data/capsules") / slug
    stage_books = stage_dir / "books"
    stage_images = stage_dir / "images"
    stage_books.mkdir(parents=True, exist_ok=True)
    stage_images.mkdir(parents=True, exist_ok=True)

    for p in target_pdfs:
        shutil.copy2(p, stage_books / p.name)

    # Metadata
    meta = KnowledgeAreaMeta(
        id=slug,
        name=name,
        description=block["description"],
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
        tags=[slug, cat, domain.lower(), "modular-rag", "smartlocal"],
        rag_spec=RagSpecification(rag_id=slug)
    )
    meta.save_json(stage_dir / "metadata.json")
    shutil.copy2(catalog_path, stage_dir / "catalog.json")
    if covers_dir.exists():
        shutil.copytree(covers_dir, stage_images / "covers", dirs_exist_ok=True)

    # 5. Package ZIP
    zip_output_path = Path("data/packs") / f"{slug}-v1.0.0.zip"
    final_zip = packager.build_package(
        metadata_source=stage_dir / "metadata.json",
        pdfs_source_dir=stage_books,
        output_zip=zip_output_path,
        cover_image_path=stage_images / "covers" / f"{enriched_records[0]['id']}.png" if enriched_records else None
    )

    # 6. Install into SmartLocal RAG Engine & Load
    resolved_id = km.packager.import_package(final_zip)
    km.load_rag(resolved_id)

    store = km.get_store(resolved_id)
    manifest = store.get_manifest()

    return {
        "slug": slug,
        "name": name,
        "books_count": len(enriched_records),
        "total_pages": total_pages,
        "total_chunks": manifest.total_chunks if manifest else 0,
        "images_extracted": manifest.total_images_extracted if manifest else 0,
        "zip_size_mb": round(final_zip.stat().st_size / (1024 * 1024), 2),
        "is_loaded_in_ram": store.is_loaded()
    }

def main():
    parser = argparse.ArgumentParser(description="Carregar els 5 blocs modulars al motor RAG de SmartLocal.")
    parser.add_argument("--max-books", type=int, default=5, help="Llibres per bloc (per defecte: 5)")
    args = parser.parse_args()

    console.print("[bold green]=======================================================[/bold green]")
    console.print("[bold green]  Carregant els 5 Blocs Modulars al Motor RAG          [/bold green]")
    console.print("[bold green]  EmbeddingGemma 2 + Gemma 4 E4B + Anàlisi Conceptual  [/bold green]")
    console.print("[bold green]=======================================================[/bold green]\n")

    embedder = EmbeddingGemma()
    image_analyzer = ImageConceptAnalyzer()
    processor = MultimodalProcessor(embedder=embedder, image_analyzer=image_analyzer)
    packager = KnowledgeCapsulePackager(processor=processor)
    extractor = FrontMatterExtractor(default_pages=2)
    enricher = CatalogEnricher()
    km = KnowledgeManager()

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[bold cyan]Processant i carregant blocs...", total=len(MODULAR_BLOCKS))

        for block in MODULAR_BLOCKS:
            slug = block["capsule_slug"]
            name = block["name"]
            progress.update(task, description=f"[bold cyan]Processant {name} ({slug})...")

            try:
                res = process_and_load_block(
                    block=block,
                    max_books=args.max_books,
                    extractor=extractor,
                    enricher=enricher,
                    packager=packager,
                    km=km
                )
                results.append(res)
                console.print(f"  [bold green]✓ RAG '{slug}' carregat:[/bold green] {res['books_count']} llibres | {res['total_chunks']} chunks | {res['images_extracted']} figures | {res['zip_size_mb']} MB")
            except Exception as e:
                console.print(f"  [bold red]✗ Error en '{slug}': {e}[/bold red]")

            progress.advance(task)

    # Final summary table
    table = Table(title="Bases de Coneixement Modulars Carregades al Motor RAG")
    table.add_column("RAG ID", style="bold cyan")
    table.add_column("Nom de la Càpsula", style="white")
    table.add_column("Llibres", style="green")
    table.add_column("Pàgines", style="cyan")
    table.add_column("Chunks", style="yellow")
    table.add_column("Figures Clau", style="magenta")
    table.add_column("Estat a RAM", style="bold green")
    table.add_column("Mida ZIP", style="blue")

    for r in results:
        loaded_str = "🟢 Carregat (RAM)" if r["is_loaded_in_ram"] else "⚪ A demanda"
        table.add_row(
            r["slug"],
            r["name"],
            str(r["books_count"]),
            str(r["total_pages"]),
            str(r["total_chunks"]),
            str(r["images_extracted"]),
            loaded_str,
            f"{r['zip_size_mb']} MB"
        )

    console.print("\n")
    console.print(table)
    console.print("\n[bold green]✓ Tots els blocs de coneixement estan instal·lats i llestos per consultar![/bold green]")

if __name__ == "__main__":
    main()
