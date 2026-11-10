"""
End-to-End Capsule Production & Benchmark Pipeline
Executes the 6-stage protocol on a document collection:
1. Verify raw PDFs
2. Extract front-matter (pages 1-3)
3. AI Catalog enrichment (short/long desc, regional scope, normatives, canonical covers)
4. Visual concept analysis & filtering (ImageConceptAnalyzer)
5. Multimodal vectorization (EmbeddingGemma 2) & index.db creation
6. ZIP packaging, registry installation, and live query test
"""

import hashlib
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

from smart_local.config import settings
from smart_local.pipeline.catalog_extractor import FrontMatterExtractor
from smart_local.pipeline.catalog_enricher import CatalogEnricher
from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
from smart_local.pipeline.image_analyzer import ImageConceptAnalyzer
from smart_local.pipeline.processor import MultimodalProcessor
from smart_local.manager.registry import KnowledgeManager
from smart_local.knowledge.area_schema import KnowledgeAreaMeta, CategoryMeta, PortalSource, RagSpecification, AreaStats
from smart_local.models.generator import Gemma4E4B
from smart_local.knowledge.retriever import MultimodalRetriever
from smart_local.models.embedding import EmbeddingGemma

console = Console()

def run_end_to_end_test(
    category_slug: str = "ciencias",
    capsule_slug: str = "ciencias-de-la-tierra",
    topic_name: str = "Ciencias de la Tierra",
    domain: str = "Ciencias Exactas y Naturales",
    max_books: Optional[int] = 5,
    query_test: str = "tectonica de placas y sismologia"
) -> Dict[str, Any]:
    console.print("[bold green]=======================================================[/bold green]")
    console.print(f"[bold green]  Executant Protocol Extrem a Extrem: {topic_name}    [/bold green]")
    console.print("[bold green]=======================================================[/bold green]\n")

    input_dir = Path("data/downloads/books") / category_slug / capsule_slug
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    metrics = {}
    t_start_total = time.time()

    # -------------------------------------------------------------------------
    # FASE 1: Verificació d'Integritat dels PDFs
    # -------------------------------------------------------------------------
    t0 = time.time()
    all_pdfs = sorted(list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF")))
    if max_books and max_books > 0:
        target_pdfs = all_pdfs[:max_books]
    else:
        target_pdfs = all_pdfs

    verified_pdfs = []
    total_bytes = 0
    for p in target_pdfs:
        with open(p, "rb") as f:
            if f.read(5).startswith(b"%PDF-"):
                verified_pdfs.append(p)
                total_bytes += p.stat().st_size

    metrics["fase1_duracio_s"] = round(time.time() - t0, 3)
    metrics["fase1_pdfs_verificats"] = len(verified_pdfs)
    metrics["fase1_mida_mb"] = round(total_bytes / (1024 * 1024), 2)
    console.print(f"[bold cyan]✓ Fase 1 Completada:[/bold cyan] {len(verified_pdfs)} PDFs vàlids ({metrics['fase1_mida_mb']} MB) en {metrics['fase1_duracio_s']}s")

    # -------------------------------------------------------------------------
    # FASE 2: Extracció de Front-Matter (Pàgines 1-3)
    # -------------------------------------------------------------------------
    t0 = time.time()
    extractor = FrontMatterExtractor(default_pages=2)
    front_matters = []
    for p in verified_pdfs:
        fm = extractor.extract_from_pdf(p, max_pages=2)
        front_matters.append(fm)

    preview_path = input_dir / "preview_front_matter.json"
    preview_path.write_text(json.dumps(front_matters, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics["fase2_duracio_s"] = round(time.time() - t0, 3)
    console.print(f"[bold cyan]✓ Fase 2 Completada:[/bold cyan] Front-matter extret i desat a {preview_path.name} en {metrics['fase2_duracio_s']}s")

    # -------------------------------------------------------------------------
    # FASE 3: Catalogació i Enriquiment Semàntic amb IA
    # -------------------------------------------------------------------------
    t0 = time.time()
    enricher = CatalogEnricher()
    covers_dir = input_dir / "images" / "covers"
    covers_dir.mkdir(parents=True, exist_ok=True)

    enriched_books = []
    total_pages = 0
    regional_counts = {}

    for fm in front_matters:
        rec = enricher.enrich_book_front_matter(
            front_matter_data=fm,
            topic_name=topic_name,
            domain=domain,
            covers_dir=covers_dir
        )
        enriched_books.append(rec.model_dump())
        total_pages += rec.total_pages
        reg = rec.regional_scope.split("(")[0].strip()
        regional_counts[reg] = regional_counts.get(reg, 0) + 1

    catalog_path = input_dir / "catalog.json"
    catalog_data = {
        "topic": topic_name,
        "domain": domain,
        "canonical_style_id": "smartlocal-academic-minimalist-v1",
        "total_books": len(enriched_books),
        "total_pages": total_pages,
        "total_size_mb": metrics["fase1_mida_mb"],
        "books": enriched_books
    }
    catalog_path.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics["fase3_duracio_s"] = round(time.time() - t0, 3)
    metrics["fase3_regions_detectades"] = list(regional_counts.keys())
    console.print(f"[bold cyan]✓ Fase 3 Completada:[/bold cyan] catalog.json generat ({len(enriched_books)} llibres, regions: {list(regional_counts.keys())}) en {metrics['fase3_duracio_s']}s")

    # -------------------------------------------------------------------------
    # FASE 4 & 5: Anàlisi Visual Conceptual + Indexació Vectorial Multimodal
    # -------------------------------------------------------------------------
    t0 = time.time()
    stage_dir = Path("data/capsules") / capsule_slug
    stage_books = stage_dir / "books"
    stage_images = stage_dir / "images"
    stage_books.mkdir(parents=True, exist_ok=True)
    stage_images.mkdir(parents=True, exist_ok=True)

    # Copy verified PDFs to stage
    for p in verified_pdfs:
        shutil.copy2(p, stage_books / p.name)

    # Initialize processor with ImageConceptAnalyzer
    image_analyzer = ImageConceptAnalyzer()
    embedder = EmbeddingGemma()
    processor = MultimodalProcessor(embedder=embedder, image_analyzer=image_analyzer)

    # Create metadata object
    meta = KnowledgeAreaMeta(
        id=capsule_slug,
        name=topic_name,
        description=f"Càpsula monogràfica de {topic_name} ({domain}) amb catàleg enriquit i anàlisi conceptual.",
        category=CategoryMeta(
            slug=category_slug,
            name=category_slug.capitalize(),
            url=f"https://infolibros.org/libros-pdf-gratis/{category_slug}/"
        ),
        domain=domain,
        portal_source=PortalSource(
            name="InfoLibros",
            url=f"https://infolibros.org/libros-pdf-gratis/{category_slug}/{capsule_slug}/"
        ),
        tags=[capsule_slug, category_slug, domain.lower(), "modular-rag", "smartlocal"],
        rag_spec=RagSpecification(rag_id=capsule_slug)
    )
    meta.save_json(stage_dir / "metadata.json")

    # Copy catalog.json to stage
    shutil.copy2(catalog_path, stage_dir / "catalog.json")

    # Copy rendered covers
    if covers_dir.exists():
        shutil.copytree(covers_dir, stage_images / "covers", dirs_exist_ok=True)

    # Build package with KnowledgeCapsulePackager
    packager = KnowledgeCapsulePackager(processor=processor)
    zip_output_path = Path("data/packs") / f"{capsule_slug}-v1.0.0.zip"
    
    final_zip = packager.build_package(
        metadata_source=stage_dir / "metadata.json",
        pdfs_source_dir=stage_books,
        output_zip=zip_output_path,
        cover_image_path=stage_images / "covers" / f"{enriched_books[0]['id']}.png" if enriched_books else None
    )

    metrics["fase4_5_duracio_s"] = round(time.time() - t0, 3)
    metrics["zip_mida_mb"] = round(final_zip.stat().st_size / (1024 * 1024), 2)
    console.print(f"[bold cyan]✓ Fase 4 & 5 Completada:[/bold cyan] Paquet ZIP generat a {final_zip.name} ({metrics['zip_mida_mb']} MB) en {metrics['fase4_5_duracio_s']}s")

    # -------------------------------------------------------------------------
    # FASE 6: Instal·lació al Registre RAG i Prova de Consulta Multimodal
    # -------------------------------------------------------------------------
    t0 = time.time()
    km = KnowledgeManager()
    resolved_id = km.packager.import_package(final_zip)
    store = km.get_store(resolved_id)

    # Execute test retrieval
    retriever = MultimodalRetriever(embedder=embedder)
    results = retriever.retrieve(store, query=query_test, top_k=3)

    # Synthesize response with Gemma 4 E4B
    generator = Gemma4E4B()
    retrieved_dicts = [
        {
            "content": r.content,
            "source_path": r.source_path,
            "page_number": r.page_number,
            "modality": r.modality,
            "score": r.score,
            "asset_rel_path": getattr(r, "asset_path", None) or getattr(r, "asset_rel_path", None),
            "metadata": r.metadata
        }
        for r in results
    ]
    chat_response = generator._synthesize_local_response(
        messages=[{"role": "user", "content": f"Explica'm sobre: {query_test}"}],
        context_chunks=retrieved_dicts
    )

    metrics["fase6_duracio_s"] = round(time.time() - t0, 3)
    metrics["rag_resolved_id"] = resolved_id
    metrics["retrieval_count"] = len(results)
    metrics["temps_total_s"] = round(time.time() - t_start_total, 3)

    console.print(f"[bold cyan]✓ Fase 6 Completada:[/bold cyan] RAG instal·lat com '{resolved_id}', consulta '{query_test}' provada en {metrics['fase6_duracio_s']}s")

    return {
        "metrics": metrics,
        "sample_enriched_book": enriched_books[0] if enriched_books else {},
        "chat_response": chat_response,
        "results": retrieved_dicts
    }

if __name__ == "__main__":
    res = run_end_to_end_test(
        category_slug="ciencias",
        capsule_slug="ciencias-de-la-tierra",
        topic_name="Ciencias de la Tierra",
        domain="Ciencias Exactas y Naturales",
        max_books=5,
        query_test="tectonica de placas y fallas sismicas"
    )
    print("\n" + "="*50)
    print("RESPOSTA DEL XAT AMB INTERCALACIÓ DE FIGURES:")
    print("="*50)
    print(res["chat_response"])
