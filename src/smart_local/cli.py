import os
import sys
import time
from pathlib import Path
from typing import Optional, List
import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from smart_local.config import settings
from smart_local.manager.registry import KnowledgeManager
from smart_local.sdk.client import SmartLocalSDK

app = typer.Typer(help="SmartLocal: Gemma 4 E4B + EmbeddingGemma Multimodal RAG CLI")
console = Console()

@app.command()
def serve(
    host: str = typer.Option(settings.host, "--host", "-h", help="Server host"),
    port: int = typer.Option(settings.port, "--port", "-p", help="Server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development")
):
    """Start the OpenAI-compatible FastAPI server."""
    console.print(f"[bold green]Iniciant el servidor SmartLocal RAG a http://{host}:{port}[/bold green]")
    console.print(f"Model generador: [cyan]{settings.generator_model}[/cyan]")
    console.print(f"Model embeddings: [cyan]{settings.embedding_model}[/cyan] (dim: {settings.embedding_dim})")
    uvicorn.run("smart_local.server.app:app", host=host, port=port, reload=reload)

@app.command("list")
def list_rags():
    """List all available RAG knowledge spaces and their status."""
    km = KnowledgeManager()
    rags = km.list_rags()
    table = Table(title="Bases de Coneixement RAG Disponibles")
    table.add_column("RAG ID", style="bold cyan")
    table.add_column("Nom", style="white")
    table.add_column("Estat a RAM", style="magenta")
    table.add_column("Documents", style="green")
    table.add_column("Chunks", style="yellow")
    table.add_column("Modalitats", style="blue")

    for r in rags:
        loaded_str = "🟢 Carregat (RAM)" if r["is_loaded"] else "⚪ A demanda (Disk)"
        table.add_row(
            r["rag_id"],
            r["name"],
            loaded_str,
            str(r["total_documents"]),
            str(r["total_chunks"]),
            ", ".join(r["modalities"]) or "cap"
        )

    console.print(table)

@app.command()
def create(
    rag_id: str = typer.Argument(..., help="Identificador únic del RAG"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Nom descriptiu"),
    description: str = typer.Option("", "--desc", "-d", help="Descripció del coneixement")
):
    """Create a new RAG knowledge space."""
    km = KnowledgeManager()
    km.create_rag(rag_id=rag_id, name=name, description=description)
    console.print(f"[bold green]✓ RAG '{rag_id}' creat correctament a {settings.rags_dir / rag_id}[/bold green]")

@app.command()
def load(rag_id: str = typer.Argument(..., help="Identificador del RAG a carregar a RAM")):
    """Load a RAG's embeddings into memory on demand."""
    km = KnowledgeManager()
    success = km.load_rag(rag_id)
    if success:
        console.print(f"[bold green]✓ Embeddings del RAG '{rag_id}' carregats a memòria RAM.[/bold green]")
    else:
        console.print(f"[bold red]✗ No s'ha trobat el RAG '{rag_id}'.[/bold red]")

@app.command()
def unload(rag_id: str = typer.Argument(..., help="Identificador del RAG a descarregar de RAM")):
    """Unload a RAG from memory to free RAM."""
    km = KnowledgeManager()
    success = km.unload_rag(rag_id)
    if success:
        console.print(f"[bold yellow]✓ RAG '{rag_id}' descarregat de memòria RAM.[/bold yellow]")
    else:
        console.print(f"[bold red]✗ No s'ha trobat el RAG '{rag_id}'.[/bold red]")

@app.command()
def package(
    rag_id: str = typer.Argument(..., help="Identificador del RAG a empaquetar"),
    version: str = typer.Option("1.0.0", "--version", "-v", help="Versió del paquet"),
    desc: str = typer.Option("", "--desc", help="Descripció del paquet")
):
    """Package a RAG into a distributable .gpk bundle."""
    km = KnowledgeManager()
    console.print(f"[cyan]Empaquetant RAG '{rag_id}' en paquet distributable...[/cyan]")
    pkg_path = km.packager.package_rag(rag_id=rag_id, version=version, description=desc)
    console.print(f"[bold green]✓ Paquet creat correctament a:[/bold green] {pkg_path}")
    console.print(f"Mida: {pkg_path.stat().st_size / 1024:.2f} KB")

@app.command()
def import_pack(
    package_path: str = typer.Argument(..., help="Ruta al fitxer .gpk o .zip a importar"),
    rag_id: Optional[str] = typer.Option(None, "--rag-id", help="Reanomenar RAG a la importació")
):
    """Import a distributable .gpk or .zip package into local RAG registry."""
    km = KnowledgeManager()
    resolved_id = km.packager.import_package(Path(package_path), rag_id_override=rag_id)
    console.print(f"[bold green]✓ Paquet importat satisfactòriament com a RAG: '{resolved_id}'[/bold green]")

@app.command("list-areas")
def list_areas():
    """List all knowledge areas available on infolibros.org."""
    from smart_local.pipeline.infolibros_scraper import InfoLibrosScraper
    scraper = InfoLibrosScraper()
    areas = scraper.list_knowledge_areas()
    table = Table(title="Àrees de Coneixement a InfoLibros (https://infolibros.org/)")
    table.add_column("Slug", style="bold cyan")
    table.add_column("Nom de l'Àrea", style="white")
    table.add_column("Macro-Domini", style="magenta")
    table.add_column("URL", style="blue")

    for a in areas:
        table.add_row(a["slug"], a["name"], a["domain"], a["url"])

    console.print(table)
    console.print(f"[green]Total àrees trobades: {len(areas)}[/green]")

@app.command("list-capsules")
def list_capsules(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filtrar per categoria (ex: ciencias, informatica, biologia)")
):
    """List all modular knowledge capsules available on InfoLibros."""
    from smart_local.pipeline.infolibros_scraper import InfoLibrosScraper
    scraper = InfoLibrosScraper()
    table = Table(title="Càpsules Modulars de Coneixement a InfoLibros")
    table.add_column("Categoria", style="magenta")
    table.add_column("Slug Càpsula", style="bold cyan")
    table.add_column("Nom del Tema", style="white")
    table.add_column("URL", style="blue")

    if category:
        caps = scraper.list_capsules_in_category(category)
    else:
        caps = scraper.list_all_modular_capsules()

    for c in caps:
        table.add_row(c["category_slug"], c["capsule_slug"], c["name"], c["url"])

    console.print(table)
    console.print(f"[green]Total càpsules modulars trobades: {len(caps)}[/green]")

@app.command("build-capsule")
def build_capsule(
    category_slug: str = typer.Argument(..., help="Slug de la categoria pare (ex: ciencias, informatica)"),
    capsule_slug: str = typer.Argument(..., help="Slug del tema específic (ex: ciencias-de-la-tierra, algoritmo)"),
    max_books: Optional[int] = typer.Option(None, "--max-books", "-m", help="Límit opcional de llibres (per defecte: tots els de la pàgina)"),
    cover_image: Optional[str] = typer.Option(None, "--cover-image", help="Ruta a una imatge de portada pròpia"),
    auto_import: bool = typer.Option(True, "--import", help="Importar automàticament al registre local SmartLocal")
):
    """Build a complete modular RAG capsule: Scrape topic -> Download all PDFs -> Vectorize -> Package ZIP -> Register."""
    from smart_local.pipeline.infolibros_scraper import InfoLibrosScraper
    from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
    scraper = InfoLibrosScraper()
    console.print(f"[bold cyan]1. Descarregant llibres i generant metadades per a '{category_slug}/{capsule_slug}'...[/bold cyan]")
    source_img = Path(cover_image) if cover_image else None
    meta, capsule_dir = scraper.create_modular_capsule(
        category_slug=category_slug,
        capsule_slug=capsule_slug,
        max_books=max_books,
        source_banner_image=source_img
    )
    console.print(f"[green]✓ Metadades i {len(meta.books)} llibres preparats a {capsule_dir}[/green]")

    console.print(f"[bold cyan]2. Vectoritzant i construint paquet modular ZIP...[/bold cyan]")
    packager = KnowledgeCapsulePackager()
    zip_path = packager.build_package(
        metadata_source=capsule_dir / "metadata.json",
        pdfs_source_dir=capsule_dir / "books",
        cover_image_path=capsule_dir / "images" / "banner.png"
    )
    console.print(f"[bold green]✓ Paquet ZIP creat:[/bold green] {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.2f} MB)")

    if auto_import:
        console.print(f"[bold cyan]3. Registrant RAG modular al sistema SmartLocal...[/bold cyan]")
        km = KnowledgeManager()
        km.packager.import_package(zip_path)
        console.print(f"[bold green]✓ RAG modular '{meta.id}' registrat i disponible per a consultes directes![/bold green]")



@app.command("extract-front-matter")
def extract_front_matter(
    dir_path: str = typer.Argument(..., help="Carpeta amb fitxers PDF a escanejar"),
    pages: int = typer.Option(2, "--pages", "-p", help="Nombre de pàgines no buides inicials a extreure"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Ruta de sortida en JSON per a l'agent")
):
    """Extract front-matter text (first 2-3 pages) from all PDFs in a folder for AI agents."""
    from smart_local.pipeline.catalog_extractor import FrontMatterExtractor
    extractor = FrontMatterExtractor(default_pages=pages)
    console.print(f"[cyan]Escanejant les primeres {pages} pàgines dels PDFs a: {dir_path}...[/cyan]")
    results = extractor.scan_directory(Path(dir_path), max_pages=pages)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[bold green]✓ S'han extret {len(results)} documents a:[/bold green] {out_path}")
    else:
        table = Table(title=f"Front-Matter Extret ({len(results)} PDFs)")
        table.add_column("Fitxer", style="bold cyan")
        table.add_column("Pàgines", style="white")
        table.add_column("Pàg. Analitzades", style="yellow")
        table.add_column("Títol Detectat", style="green")
        table.add_column("Fragment de Text Inicial", style="magenta")

        for r in results:
            snippet = r.get("front_matter_text", "").replace("\n", " ")[:90] + "..."
            table.add_row(
                r["filename"],
                str(r.get("total_pages", 0)),
                str(r.get("pages_analyzed_count", 0)),
                r.get("title_guess", ""),
                snippet
            )
        console.print(table)

@app.command("enrich-catalog")
def enrich_catalog(
    dir_path: str = typer.Argument(..., help="Carpeta amb fitxers PDF descarregats"),
    topic: str = typer.Option("Conocimiento General", "--topic", "-t", help="Nom del tema o càpsula"),
    domain: str = typer.Option("Ciencias Exactas y Naturales", "--domain", "-d", help="Domini disciplinar"),
    pages: int = typer.Option(2, "--pages", "-p", help="Pàgines inicials a analitzar"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Fitxer catalog.json destí"),
    render_covers: bool = typer.Option(True, "--render-covers/--no-covers", help="Generar portades visuals canòniques a images/covers/")
):
    """Enrich book catalog with AI: short/long descriptions, regional scope, normatives, and canonical cover prompts."""
    from smart_local.pipeline.catalog_extractor import FrontMatterExtractor
    from smart_local.pipeline.catalog_enricher import CatalogEnricher
    import json

    input_dir = Path(dir_path)
    extractor = FrontMatterExtractor(default_pages=pages)
    enricher = CatalogEnricher()

    console.print(f"[cyan]1. Extraient front-matter de {input_dir}...[/cyan]")
    raw_front_matters = extractor.scan_directory(input_dir, max_pages=pages)

    console.print(f"[cyan]2. Catalogant i analitzant amb IA ({len(raw_front_matters)} llibres)...[/cyan]")
    covers_dir = input_dir / "images" / "covers" if render_covers else None

    enriched_records = []
    total_pages = 0
    total_bytes = 0

    for item in raw_front_matters:
        if "error" in item:
            continue
        rec = enricher.enrich_book_front_matter(
            front_matter_data=item,
            topic_name=topic,
            domain=domain,
            covers_dir=covers_dir
        )
        enriched_records.append(rec.model_dump())
        total_pages += rec.total_pages
        total_bytes += rec.file_size_bytes

    active_style = enricher._select_style_for_topic(topic, domain)
    banner_file = input_dir / "images" / "banner.jpg"
    if not banner_file.exists():
        banner_file = input_dir / "images" / "banner.png"
    banner_rel = f"images/{banner_file.name}" if banner_file.exists() else "images/banner.jpg"

    out_file = Path(output) if output else (input_dir / "catalog.json")
    catalog_data = {
        "topic": topic,
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
    out_file.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green]✓ Catàleg enriquit generat amb èxit a:[/bold green] {out_file}")
    console.print(f"Total llibres catalogats: {len(enriched_records)} | Mida: {catalog_data['total_size_mb']} MB")

@app.command("show-catalog")
def show_catalog(
    catalog_file: str = typer.Argument(..., help="Ruta al fitxer catalog.json")
):
    """Display an enriched catalog.json in a formatted summary table."""
    import json
    p = Path(catalog_file)
    if not p.exists():
        console.print(f"[bold red]Fitxer no trobat: {p}[/bold red]")
        return

    data = json.loads(p.read_text(encoding="utf-8"))
    books = data.get("books", [])

    table = Table(title=f"Catàleg Enriquit: {data.get('topic', 'Llibres')} ({len(books)} títols)")
    table.add_column("Títol", style="bold cyan")
    table.add_column("Àmbit Regional", style="magenta")
    table.add_column("Normatives / Estàndards", style="yellow")
    table.add_column("Nivell Acadèmic", style="blue")
    table.add_column("Descripció Curta", style="white")

    for b in books:
        norms = ", ".join(b.get("normatives_and_standards", [])) or "Estàndards generals"
        table.add_row(
            b.get("title", "")[:35],
            b.get("regional_scope", "")[:30],
            norms[:35],
            b.get("academic_level", "")[:25],
            b.get("short_description", "")[:60] + "..."
        )
    console.print(table)

@app.command()
def chat(
    rag_id: Optional[str] = typer.Option(None, "--rag", "-r", help="RAG opcional a consultar"),
    api_url: str = typer.Option("http://localhost:8000/v1", "--url", help="URL base de la API")
):
    """Interactive chat with Gemma 4 E4B using optional RAG knowledge."""
    client = SmartLocalSDK(base_url=api_url)
    model_name = f"gemma-4-e4b:{rag_id}" if rag_id else "gemma-4-e4b"
    console.print(f"[bold green]Iniciant xat amb {model_name}... (Escriu 'exit' per sortir)[/bold green]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]Tu > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "sortir"]:
                break

            console.print(f"\n[bold magenta]Gemma 4 E4B > [/bold magenta]", end="")
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_input}],
                stream=True,
                extra_body={"rag_id": rag_id} if rag_id else {}
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                console.print(content, end="")
            console.print("\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]\n")

def main():
    app()

if __name__ == "__main__":
    main()
