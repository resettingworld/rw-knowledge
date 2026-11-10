import os
import sys
import shutil
import zipfile
import json
from pathlib import Path

# Add src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smart_local.pipeline.catalog_extractor import FrontMatterExtractor
from smart_local.pipeline.catalog_enricher import CatalogEnricher

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

os.makedirs("data/packs", exist_ok=True)

for item in capsules:
    cap_id = item["id"]
    topic = item["topic"]
    domain = item["domain"]
    cap_dir = item["dir"]
    rag_dir = item["active_rag"]

    print(f"\n==========================================")
    print(f"Processing: {cap_id} ({topic})")
    print(f"==========================================")

    covers_dir = cap_dir / "images" / "covers"
    covers_dir.mkdir(parents=True, exist_ok=True)

    # 1. Front-matter extraction
    raw_front_matters = extractor.scan_directory(cap_dir, max_pages=3)
    print(f"Extracted front-matter from {len(raw_front_matters)} documents.")

    # 2. Enrich records & render covers using the master AI backdrop
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
    print(f"✓ Catalog enriched: {len(enriched_records)} books, style: {active_style['name']}")

    # Sync to capsules and active rag
    capsule_export_dir = Path("data/capsules") / cap_id
    if capsule_export_dir.exists():
        (capsule_export_dir / "catalog.json").write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copytree(covers_dir, capsule_export_dir / "images" / "covers", dirs_exist_ok=True)
        if banner_file.exists():
            (capsule_export_dir / "images").mkdir(parents=True, exist_ok=True)
            shutil.copy2(banner_file, capsule_export_dir / banner_rel)

    if rag_dir.exists():
        (rag_dir / "catalog.json").write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copytree(covers_dir, rag_dir / "images" / "covers", dirs_exist_ok=True)
        if banner_file.exists():
            (rag_dir / "images").mkdir(parents=True, exist_ok=True)
            shutil.copy2(banner_file, rag_dir / banner_rel)

    # 3. Create zip package directly from rag_dir
    zip_path = Path(f"data/packs/{cap_id}-v1.0.0.zip")
    print(f"Compressing into {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(rag_dir):
            for f in files:
                full_path = Path(root) / f
                rel_path = full_path.relative_to(rag_dir)
                z.write(full_path, arcname=str(rel_path))
    print(f"✓ Package ready: {zip_path} ({round(zip_path.stat().st_size / (1024 * 1024), 2)} MB)")

print("\n==========================================")
print("All 5 capsules successfully enriched and packaged!")
print("==========================================")
