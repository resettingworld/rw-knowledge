import os
import sys
from pathlib import Path
from smart_local.pipeline.catalog_extractor import FrontMatterExtractor
from smart_local.pipeline.catalog_enricher import CatalogEnricher
from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
import json

capsules = [
    {
        "id": "ciencias-de-la-tierra",
        "topic": "Ciencias de la Tierra",
        "domain": "Ciencias Exactas y Naturales",
        "dir": Path("data/downloads/books/ciencias/ciencias-de-la-tierra"),
        "active_rag": Path("data/rags/ciencias-de-la-tierra")
    },
    {
        "id": "ciencias-aplicadas",
        "topic": "Ciencias Aplicadas",
        "domain": "Ciencias Aplicadas e Ingeniería",
        "dir": Path("data/downloads/books/ciencias/ciencias-aplicadas"),
        "active_rag": Path("data/rags/ciencias-aplicadas")
    },
    {
        "id": "ciencias-naturales",
        "topic": "Ciencias Naturales",
        "domain": "Ciencias Exactas y Naturales",
        "dir": Path("data/downloads/books/ciencias/ciencias-naturales"),
        "active_rag": Path("data/rags/ciencias-naturales")
    },
    {
        "id": "ciencias-sociales",
        "topic": "Ciencias Sociales",
        "domain": "Ciencias Sociales y Humanidades",
        "dir": Path("data/downloads/books/ciencias/ciencias-sociales"),
        "active_rag": Path("data/rags/ciencias-sociales")
    },
    {
        "id": "algoritmo",
        "topic": "Algoritmos y Programación",
        "domain": "Informática y Computación",
        "dir": Path("data/downloads/books/informatica/algoritmo"),
        "active_rag": Path("data/rags/algoritmo")
    }
]

extractor = FrontMatterExtractor()
enricher = CatalogEnricher()
packager = KnowledgeCapsulePackager()

for item in capsules:
    cap_dir = item["dir"]
    topic = item["topic"]
    domain = item["domain"]
    cap_id = item["id"]
    rag_dir = item["active_rag"]
    
    print(f"\n==========================================")
    print(f"Refreshing Catalog & Covers for: {cap_id} ({topic})")
    print(f"==========================================")

    # Clean existing procedural covers so they get re-rendered with new AI backdrop
    covers_dir = cap_dir / "images" / "covers"
    if covers_dir.exists():
        import shutil
        shutil.rmtree(covers_dir)
    covers_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract front matters
    raw_front_matters = extractor.scan_directory(cap_dir, max_pages=3)

    # 2. Enrich records & render covers
    enriched_records = []
    total_pages = 0
    total_bytes = 0

    for fm in raw_front_matters:
        if "error" in fm:
            continue
        rec = enricher.enrich_book_front_matter(
            front_matter_data=fm,
            topic_name=topic,
            domain=domain,
            covers_dir=covers_dir
        )
        enriched_records.append(rec.model_dump())
        total_pages += rec.total_pages
        total_bytes += rec.file_size_bytes

    active_style = enricher._select_style_for_topic(topic, domain)
    banner_file = cap_dir / "images" / "banner.jpg"
    if not banner_file.exists():
        banner_file = cap_dir / "images" / "banner.png"
    banner_rel = f"images/{banner_file.name}" if banner_file.exists() else "images/banner.jpg"

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

    catalog_out = cap_dir / "catalog.json"
    catalog_out.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved enriched catalog to {catalog_out} with {len(enriched_records)} books.")

    # Synchronize to data/capsules and active RAG
    capsule_export_dir = Path("data/capsules") / cap_id
    if capsule_export_dir.exists():
        (capsule_export_dir / "catalog.json").write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
        if covers_dir.exists():
            import shutil
            shutil.copytree(covers_dir, capsule_export_dir / "images" / "covers", dirs_exist_ok=True)

    if rag_dir.exists():
        (rag_dir / "catalog.json").write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
        if covers_dir.exists():
            import shutil
            shutil.copytree(covers_dir, rag_dir / "images" / "covers", dirs_exist_ok=True)

    # 3. Re-package into data/packs/<cap_id>-v1.0.0.zip
    metadata_path = cap_dir / "metadata.json"
    if not metadata_path.exists() and (capsule_export_dir / "metadata.json").exists():
        metadata_path = capsule_export_dir / "metadata.json"
    
    out_zip = Path(f"data/packs/{cap_id}-v1.0.0.zip")
    banner_img = cap_dir / "images" / "banner.jpg"
    if not banner_img.exists():
        banner_img = cap_dir / "images" / "banner.png"

    if metadata_path.exists():
        print(f"Repackaging {out_zip}...")
        packager.build_package(
            metadata_source=metadata_path,
            pdfs_source_dir=cap_dir,
            output_zip=out_zip,
            cover_image_path=banner_img if banner_img.exists() else None
        )
        print(f"Packaged {out_zip} ({round(out_zip.stat().st_size / (1024*1024), 2)} MB)")

print("\nAll 5 capsules refreshed with artistic covers and re-packaged!")
