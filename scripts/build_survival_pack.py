import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from smart_local.config import settings
from smart_local.manager.registry import KnowledgeManager
from smart_local.pipeline.processor import MultimodalProcessor
from smart_local.knowledge.schema import KnowledgePackManifest

console = Console()

def main():
    console.print("[bold green]=======================================================[/bold green]")
    console.print("[bold green]  Creació del Paquet de Supervivència en Espanyol      [/bold green]")
    console.print("[bold green]  Gemma 4 E4B + EmbeddingGemma 2 Multimodal RAG        [/bold green]")
    console.print("[bold green]=======================================================[/bold green]\n")

    downloads_dir = Path(os.path.expanduser("~/Downloads"))
    rag_id = "supervivencia-es"
    rag_name = "Biblioteca de Supervivencia y Primeros Auxilios"
    rag_desc = "Manuales completos de supervivencia urbana, natural, desastres, primeros auxilios, refugio y rescate en español."
    tags = ["supervivencia", "primeros-auxilios", "desastres", "bushcraft", "preparacionismo", "español"]

    km = KnowledgeManager()
    store = km.create_rag(
        rag_id=rag_id,
        name=rag_name,
        description=rag_desc,
        tags=tags
    )

    processor = MultimodalProcessor()

    # Identify survival PDFs
    survival_keywords = ['supervivencia', 'desastres', 'primeros-auxilios', 'emergencia', 'sismo', 'preparacionista', 'bushcraft', 'alimentos']
    pdf_files = []
    for f in downloads_dir.glob("*.pdf"):
        fl = f.name.lower()
        if any(k in fl for k in survival_keywords) and 'contrato' not in fl and 'resume' not in fl and 'certificado' not in fl:
            pdf_files.append(f)

    pdf_files.sort(key=lambda x: x.name)
    console.print(f"[cyan]S'han detectat {len(pdf_files)} manuals de supervivència a {downloads_dir}[/cyan]\n")

    total_chunks_added = 0
    total_images_added = 0
    total_pages_indexed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        overall_task = progress.add_task("[bold cyan]Processant manuals PDF...[/bold cyan]", total=len(pdf_files))

        for idx, pdf_path in enumerate(pdf_files):
            progress.update(overall_task, description=f"[yellow]Ingestant {idx+1}/{len(pdf_files)}: {pdf_path.name[:35]}...[/yellow]")

            try:
                doc, lib_doc, chunks = processor.process_pdf_file(
                    rag_id=rag_id,
                    file_path=pdf_path,
                    library_dir=store.library_dir,
                    assets_dir=store.assets_dir,
                    chunk_size=600,
                    chunk_overlap=80,
                    extract_images=True
                )

                store.add_document(doc)
                if lib_doc:
                    store.add_library_document(lib_doc)
                    total_images_added += lib_doc.image_count
                    total_pages_indexed += lib_doc.page_count

                store.add_chunks(chunks)
                total_chunks_added += len(chunks)

            except Exception as e:
                console.print(f"[red]Error processant {pdf_path.name}: {e}[/red]")

            progress.advance(overall_task, 1)

    # Save manifest with stats
    stats = store.get_stats()
    manifest = KnowledgePackManifest(
        rag_id=rag_id,
        name=rag_name,
        description=rag_desc,
        version="1.0.0",
        author="smart-local",
        created_at=time.time(),
        embedding_model=settings.embedding_model,
        generator_model=settings.generator_model,
        total_chunks=stats["total_chunks"],
        total_documents=stats["total_documents"],
        total_library_files=stats["total_library_files"],
        total_images_extracted=stats["total_images_extracted"],
        modalities=stats["modalities"],
        tags=tags
    )
    store.save_manifest(manifest)

    console.print("\n[bold green]✓ Ingestió completada amb èxit![/bold green]")
    console.print(f"- Manuals originals arxivats a la Biblioteca Digital: [cyan]{stats['total_library_files']}[/cyan]")
    console.print(f"- Pàgines indexades: [cyan]{total_pages_indexed}[/cyan]")
    console.print(f"- Imatges i diagrames extrets i vectoritzats: [cyan]{stats['total_images_extracted']}[/cyan]")
    console.print(f"- Total de chunks vectoritzats (EmbeddingGemma 2 768-d): [cyan]{stats['total_chunks']}[/cyan]")

    # Package into .gpk bundle
    console.print("\n[cyan]Empaquetant en arxiu portable distributable (.gpk)...[/cyan]")
    pkg_file = km.packager.package_rag(
        rag_id=rag_id,
        name=rag_name,
        version="1.0.0",
        description=rag_desc,
        tags=tags
    )
    console.print(f"[bold green]✓ Paquet distributable generat:[/bold green] [yellow]{pkg_file}[/yellow]")
    console.print(f"Mida del paquet: {pkg_file.stat().st_size / (1024 * 1024):.2f} MB")

    # Load on demand into RAM
    console.print("\n[cyan]Carregant embeddings del RAG a la memòria RAM...[/cyan]")
    km.load_rag(rag_id)
    console.print(f"[bold green]✓ RAG '{rag_id}' carregat a RAM i disponible per a consultes immediates![/bold green]")

if __name__ == "__main__":
    main()
