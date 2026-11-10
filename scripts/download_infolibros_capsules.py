"""
Download and Build InfoLibros Modular Knowledge Capsules
Discovers and builds focused, modular RAG capsules from InfoLibros topics.
Level 1: Category (e.g. ciencias, informatica, biologia)
Level 2: Capsule (e.g. ciencias-de-la-tierra, algoritmo, genetica) with all its books.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from smart_local.pipeline.infolibros_scraper import InfoLibrosScraper
from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
from smart_local.manager.registry import KnowledgeManager

console = Console()

def run_capsules_pipeline(
    category_filter: str = None,
    capsules_filter: list = None,
    max_books_per_capsule: int = None,
    auto_package: bool = True,
    auto_import: bool = False,
    delay_between_requests: float = 1.0
):
    scraper = InfoLibrosScraper()
    packager = KnowledgeCapsulePackager()
    km = KnowledgeManager() if auto_import else None

    console.print("[bold green]=======================================================[/bold green]")
    console.print("[bold green]  Generador de Càpsules RAG Modulars d'InfoLibros       [/bold green]")
    console.print("[bold green]  Nivell: Tema Específic (ex: ciencias-de-la-tierra)   [/bold green]")
    console.print("[bold green]=======================================================[/bold green]\n")

    # Discover target capsules
    target_capsules = []
    if category_filter and category_filter != "all":
        caps = scraper.list_capsules_in_category(category_filter)
        target_capsules.extend(caps)
    else:
        # If specific capsule slugs given
        if capsules_filter and "all" not in capsules_filter:
            all_caps = scraper.list_all_modular_capsules()
            target_capsules = [c for c in all_caps if c["capsule_slug"] in capsules_filter]
        else:
            # All capsules across all categories
            target_capsules = scraper.list_all_modular_capsules()

    if not target_capsules:
        console.print("[bold red]No s'ha trobat cap càpsula que coincideixi amb els filtres.[/bold red]")
        return

    console.print(f"[cyan]S'han seleccionat {len(target_capsules)} càpsules modulars per processar.[/cyan]")
    limit_str = str(max_books_per_capsule) if max_books_per_capsule else "TOTS els disponibles"
    console.print(f"[cyan]Llibres per càpsula: {limit_str}[/cyan]\n")

    summary_table = Table(title="Càpsules Modulars Processades")
    summary_table.add_column("Categoria", style="magenta")
    summary_table.add_column("Slug Càpsula", style="bold cyan")
    summary_table.add_column("Nom del Tema", style="white")
    summary_table.add_column("Llibres", style="green")
    summary_table.add_column("Paquet ZIP", style="yellow")
    summary_table.add_column("Mida", style="blue")

    successful_capsules = 0
    total_books_downloaded = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        overall_task = progress.add_task("[bold cyan]Processant càpsules modulars...", total=len(target_capsules))

        for cap in target_capsules:
            cat_slug = cap["category_slug"]
            cap_slug = cap["capsule_slug"]
            name = cap["name"]
            progress.update(overall_task, description=f"[bold cyan]Càpsula: {name} ({cat_slug}/{cap_slug})...")

            try:
                # 1. Scrape specific topic and download books
                meta, capsule_dir = scraper.create_modular_capsule(
                    category_slug=cat_slug,
                    capsule_slug=cap_slug,
                    max_books=max_books_per_capsule
                )

                book_count = len(meta.books)
                total_books_downloaded += book_count

                zip_status = "N/A"
                zip_size_str = "N/A"

                # 2. Package into standard ZIP
                if auto_package and book_count > 0:
                    zip_path = packager.build_package(
                        metadata_source=capsule_dir / "metadata.json",
                        pdfs_source_dir=capsule_dir / "books",
                        cover_image_path=capsule_dir / "images" / "banner.png"
                    )
                    zip_status = zip_path.name
                    zip_size_str = f"{zip_path.stat().st_size / (1024 * 1024):.2f} MB"

                    if auto_import and km:
                        km.packager.import_package(zip_path)

                summary_table.add_row(cat_slug, cap_slug, name, str(book_count), zip_status, zip_size_str)
                successful_capsules += 1

            except Exception as e:
                console.print(f"[bold red]✗ Error en càpsula '{cap_slug}': {e}[/bold red]")
                summary_table.add_row(cat_slug, cap_slug, name, "0 (Error)", "Fallat", "0 MB")

            time.sleep(delay_between_requests)
            progress.advance(overall_task)

    console.print("\n")
    console.print(summary_table)
    console.print(f"\n[bold green]✓ Procés completat![/bold green]")
    console.print(f"Càpsules generades amb èxit: [bold cyan]{successful_capsules}/{len(target_capsules)}[/bold cyan]")
    console.print(f"Total llibres descarregats: [bold cyan]{total_books_downloaded}[/bold cyan]")

def main():
    parser = argparse.ArgumentParser(description="Descarregar i crear càpsules de coneixement modulars d'InfoLibros.")
    parser.add_argument("--category", type=str, default=None, help="Filtrar per macro-categoria (ex: ciencias, informatica, biologia)")
    parser.add_argument("--capsules", nargs="*", default=None, help="Slugs de càpsules específiques (ex: ciencias-de-la-tierra algoritmo)")
    parser.add_argument("--max-books", type=int, default=None, help="Límit opcional de llibres per càpsula (per defecte: tots els del tema)")
    parser.add_argument("--no-package", action="store_true", help="Només descarregar sense vectoritzar ni crear ZIP")
    parser.add_argument("--import-rag", action="store_true", help="Importar automàticament els RAGs al registre local")
    parser.add_argument("--delay", type=float, default=1.0, help="Pausa entre peticions en segons")

    args = parser.parse_args()

    run_capsules_pipeline(
        category_filter=args.category,
        capsules_filter=args.capsules,
        max_books_per_capsule=args.max_books,
        auto_package=not args.no_package,
        auto_import=args.import_rag,
        delay_between_requests=args.delay
    )

if __name__ == "__main__":
    main()
