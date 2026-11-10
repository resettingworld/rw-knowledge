import json
import zipfile
from pathlib import Path
import pytest

from smart_local.knowledge.area_schema import KnowledgeAreaMeta, CategoryMeta, PortalSource, BookMeta, RagSpecification
from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
from smart_local.pipeline.infolibros_scraper import InfoLibrosScraper

def test_json_schema_validation(tmp_path):
    schema_path = Path("src/smart_local/schemas/knowledge_area.schema.json")
    assert schema_path.exists(), "Schema file must exist"

    schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema_data["title"] == "KnowledgeCapsuleMetadata"
    assert "books" in schema_data["required"]
    assert "rag_spec" in schema_data["required"]

    # Validate Pydantic model serialization matches
    meta = KnowledgeAreaMeta(
        id="ciencias-de-la-tierra",
        name="Ciencias de la Tierra",
        description="A test modular knowledge capsule for earth sciences.",
        category=CategoryMeta(slug="ciencias", name="Ciencias", url="https://infolibros.org/libros-pdf-gratis/ciencias/"),
        domain="Ciencias Exactas y Naturales",
        portal_source=PortalSource(url="https://infolibros.org/libros-pdf-gratis/ciencias/ciencias-de-la-tierra/"),
        rag_spec=RagSpecification(rag_id="ciencias-de-la-tierra"),
    )
    meta_json_path = tmp_path / "metadata.json"
    meta.save_json(meta_json_path)

    loaded = KnowledgeAreaMeta.load_json(meta_json_path)
    assert loaded.id == "ciencias-de-la-tierra"
    assert loaded.category.slug == "ciencias"
    assert loaded.domain == "Ciencias Exactas y Naturales"

def test_infolibros_scraper_capsule_list():
    scraper = InfoLibrosScraper()
    categories = scraper.list_categories()
    assert len(categories) >= 20, "Should discover at least 20 categories from InfoLibros"
    slugs = [a["slug"] for a in categories]
    assert "ciencias" in slugs
    assert "informatica" in slugs

    # Test modular capsules discovery
    ciencias_caps = scraper.list_capsules_in_category("ciencias")
    cap_slugs = [c["capsule_slug"] for c in ciencias_caps]
    assert "ciencias-de-la-tierra" in cap_slugs
    assert "ciencias-aplicadas" in cap_slugs

def test_capsule_packager_build_zip(tmp_path):
    # Setup dummy pdf and metadata
    pdfs_dir = tmp_path / "sample_books"
    pdfs_dir.mkdir()
    sample_pdf = pdfs_dir / "test_doc.pdf"

    # Create a minimal valid PDF
    import fitz  # pymupdf
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Capítulo 1: Introducción a los algoritmos y estructuras de datos.")
    doc.save(str(sample_pdf))
    doc.close()

    meta = KnowledgeAreaMeta(
        id="test-algoritmos",
        name="Algoritmos de Prueba",
        description="Cápsula de prueba unitaria de algoritmos.",
        category=CategoryMeta(slug="informatica", name="Informática", url="https://infolibros.org/libros-pdf-gratis/informatica/"),
        domain="Tecnología e Informática",
        portal_source=PortalSource(url="https://infolibros.org/libros-pdf-gratis/informatica/algoritmo/"),
        rag_spec=RagSpecification(rag_id="test-algoritmos"),
    )
    meta_file = tmp_path / "metadata.json"
    meta.save_json(meta_file)

    output_zip = tmp_path / "test-algoritmos-v1.0.0.zip"
    packager = KnowledgeCapsulePackager(packs_dir=tmp_path / "packs")

    result_zip = packager.build_package(
        metadata_source=meta_file,
        pdfs_source_dir=pdfs_dir,
        output_zip=output_zip
    )

    assert result_zip.exists()
    assert result_zip.suffix == ".zip"

    # Verify standard contents inside the zip
    with zipfile.ZipFile(result_zip, "r") as zf:
        namelist = zf.namelist()
        assert "metadata.json" in namelist
        assert "manifest.json" in namelist
        assert "index.db" in namelist
        assert any(n.startswith("books/") for n in namelist)

    # Test importing package
    rags_dest = tmp_path / "test_rags"
    rags_dest.mkdir()
    imported_id = packager.import_zip_package(result_zip, rags_dir=rags_dest)
    assert imported_id == "test-algoritmos"
    assert (rags_dest / "test-algoritmos" / "index.db").exists()
    assert (rags_dest / "test-algoritmos" / "metadata.json").exists()
