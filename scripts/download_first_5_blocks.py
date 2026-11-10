"""
Download the first 5 modular blocks of books from InfoLibros
Saves all PDF books into structured directories under data/downloads/books/<category>/<capsule>/
Generates a catalog.json in each folder for easy user inspection and validation.
Does NOT create the RAG or metadata yet, awaiting user approval.
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
import pymupdf
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

BASE_URL = "https://infolibros.org"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

TARGET_BLOCKS = [
    {
        "category": "ciencias",
        "capsule": "ciencias-de-la-tierra",
        "name": "Ciencias de la Tierra",
        "url": f"{BASE_URL}/libros-pdf-gratis/ciencias/ciencias-de-la-tierra/"
    },
    {
        "category": "ciencias",
        "capsule": "ciencias-aplicadas",
        "name": "Ciencias Aplicadas",
        "url": f"{BASE_URL}/libros-pdf-gratis/ciencias/ciencias-aplicadas/"
    },
    {
        "category": "ciencias",
        "capsule": "ciencias-naturales",
        "name": "Ciencias Naturales",
        "url": f"{BASE_URL}/libros-pdf-gratis/ciencias/ciencias-naturales/"
    },
    {
        "category": "ciencias",
        "capsule": "ciencias-sociales",
        "name": "Ciencias Sociales",
        "url": f"{BASE_URL}/libros-pdf-gratis/ciencias/ciencias-sociales/"
    },
    {
        "category": "informatica",
        "capsule": "algoritmo",
        "name": "Algoritmos y Programación",
        "url": f"{BASE_URL}/libros-pdf-gratis/informatica/algoritmo/"
    }
]

def scrape_topic_books(client: httpx.Client, topic_url: str) -> List[Dict[str, str]]:
    res = client.get(topic_url)
    if res.status_code != 200:
        return []

    html = res.text
    pdf_matches = list(dict.fromkeys(re.findall(
        r'href=[\"\'](https://pdf\.infolibros\.org/[^\"\'#]+\.pdf)[\"\']',
        html,
        re.IGNORECASE
    )))

    # Pair with headings
    paired = re.findall(
        r'<h[23][^>]*>(.*?)</h[23]>.*?href=[\"\'](https://pdf\.infolibros\.org/[^\"\'#]+\.pdf)[\"\']',
        html,
        re.DOTALL | re.IGNORECASE
    )
    url_to_title = {}
    for h, u in paired:
        clean_h = re.sub(r'<[^>]+>', '', h).strip()
        if not any(k in clean_h.lower() for k in ["ad", "rewarded", "descarg", "colecci"]):
            url_to_title[u] = clean_h

    books = []
    for u in pdf_matches:
        fname = u.split("/")[-1]
        title_guess = url_to_title.get(u)
        if not title_guess:
            title_guess = re.sub(r'-\d+\.pdf$', '', fname)
            title_guess = re.sub(r'\.pdf$', '', title_guess).replace("-", " ").title()

        books.append({
            "title": title_guess,
            "url": u,
            "filename": fname,
            "referer": topic_url
        })
    return books

def download_book(client: httpx.Client, pdf_url: str, dest_path: Path, referer: str) -> Optional[Dict[str, Any]]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {**DEFAULT_HEADERS, "Referer": referer}

    try:
        with client.stream("GET", pdf_url, headers=headers) as response:
            if response.status_code != 200:
                return None

            h = hashlib.sha256()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    h.update(chunk)

        # Check %PDF- header
        with open(dest_path, "rb") as f:
            if not f.read(5).startswith(b"%PDF-"):
                dest_path.unlink(missing_ok=True)
                return None

        sha256 = h.hexdigest()
        size_bytes = dest_path.stat().st_size

        # Count pages
        page_count = 1
        try:
            pdf_doc = pymupdf.open(str(dest_path))
            page_count = len(pdf_doc)
            pdf_doc.close()
        except Exception:
            pass

        return {
            "path": str(dest_path),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "page_count": page_count
        }
    except Exception:
        dest_path.unlink(missing_ok=True)
        return None

def main():
    console.print("[bold green]=======================================================[/bold green]")
    console.print("[bold green]  Descàrrega dels 5 Primers Blocs de Coneixement       [/bold green]")
    console.print("[bold green]  (Preservació de Llibres en PDF per a Validació)      [/bold green]")
    console.print("[bold green]=======================================================[/bold green]\n")

    base_downloads = Path("data/downloads/books")
    base_downloads.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        verify=False,
        timeout=30.0
    )

    summary_table = Table(title="Resum dels 5 Blocs Descarregats")
    summary_table.add_column("Bloc #", style="bold yellow")
    summary_table.add_column("Categoria", style="magenta")
    summary_table.add_column("Càpsula", style="bold cyan")
    summary_table.add_column("Nom del Tema", style="white")
    summary_table.add_column("Llibres", style="green")
    summary_table.add_column("Pàgines", style="cyan")
    summary_table.add_column("Mida Total", style="blue")

    total_downloaded_all = 0
    total_mb_all = 0.0

    for idx, block in enumerate(TARGET_BLOCKS, 1):
        cat = block["category"]
        capsule = block["capsule"]
        name = block["name"]
        url = block["url"]

        block_dir = base_downloads / cat / capsule
        block_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[bold cyan]▶ [{idx}/5] Escanejant bloc: {name} ({cat}/{capsule})[/bold cyan]")
        books_found = scrape_topic_books(client, url)
        console.print(f"  Trobats [green]{len(books_found)}[/green] llibres disponibles a {url}")

        catalog = []
        block_pages = 0
        block_bytes = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            dl_task = progress.add_task(f"  Descarregant llibres de {capsule}...", total=len(books_found))

            for b in books_found:
                fname = b["filename"]
                dest_file = block_dir / fname
                progress.update(dl_task, description=f"  [cyan]{fname[:35]}...")

                info = None
                if dest_file.exists() and dest_file.stat().st_size > 1000:
                    with open(dest_file, "rb") as f_chk:
                        h = hashlib.sha256(f_chk.read()).hexdigest()
                    try:
                        doc = pymupdf.open(str(dest_file))
                        pc = len(doc)
                        doc.close()
                    except Exception:
                        pc = 1
                    info = {
                        "path": str(dest_file),
                        "size_bytes": dest_file.stat().st_size,
                        "sha256": h,
                        "page_count": pc
                    }
                else:
                    info = download_book(client, b["url"], dest_file, referer=b["referer"])
                    time.sleep(0.5)  # Be polite to the CDN

                if info:
                    catalog.append({
                        "title": b["title"],
                        "filename": fname,
                        "url": b["url"],
                        "size_bytes": info["size_bytes"],
                        "sha256": info["sha256"],
                        "page_count": info["page_count"]
                    })
                    block_pages += info["page_count"]
                    block_bytes += info["size_bytes"]

                progress.advance(dl_task)

        # Write catalog.json in folder
        catalog_file = block_dir / "catalog.json"
        catalog_file.write_text(json.dumps({
            "block_index": idx,
            "category": cat,
            "capsule": capsule,
            "name": name,
            "source_url": url,
            "total_books": len(catalog),
            "total_pages": block_pages,
            "total_size_mb": round(block_bytes / (1024 * 1024), 2),
            "books": catalog
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        mb = block_bytes / (1024 * 1024)
        total_downloaded_all += len(catalog)
        total_mb_all += mb

        summary_table.add_row(
            str(idx),
            cat,
            capsule,
            name,
            str(len(catalog)),
            str(block_pages),
            f"{mb:.2f} MB"
        )
        console.print(f"  [bold green]✓ Completat bloc {idx}: {len(catalog)} llibres ({mb:.2f} MB)[/bold green]")

    console.print("\n")
    console.print(summary_table)
    console.print(f"\n[bold green]=======================================================[/bold green]")
    console.print(f"[bold green]  ✓ Els 5 blocs s'han descarregat satisfactòriament!  [/bold green]")
    console.print(f"  Total llibres: [bold cyan]{total_downloaded_all}[/bold cyan] en [bold cyan]{total_mb_all:.2f} MB[/bold cyan]")
    console.print(f"  Ubicació: [bold cyan]data/downloads/books/[/bold cyan]")
    console.print(f"  Pendent de la teva validació abans de crear RAGs i metadades.")
    console.print("[bold green]=======================================================[/bold green]\n")

if __name__ == "__main__":
    main()
