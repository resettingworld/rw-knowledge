import pytest
from pathlib import Path
from PIL import Image, ImageDraw

from smart_local.pipeline.image_analyzer import ImageConceptAnalyzer, ImageEvaluationResult
from smart_local.models.generator import Gemma4E4B

def test_image_analyzer_rejects_tiny_and_flat_images(tmp_path):
    analyzer = ImageConceptAnalyzer()

    # 1. Tiny icon
    tiny_img = tmp_path / "tiny.png"
    im1 = Image.new("RGB", (50, 50), color=(255, 255, 255))
    im1.save(tiny_img)

    res1 = analyzer.evaluate_image(
        image_path=tiny_img,
        page_text="Some text about biology",
        doc_title="Manual",
        page_num=1,
        width=50,
        height=50
    )
    assert not res1.is_relevant, "Tiny icon should be rejected"

    # 2. Flat monochrome block (large but zero entropy)
    flat_img = tmp_path / "flat.png"
    im2 = Image.new("RGB", (300, 300), color=(200, 200, 200))
    im2.save(flat_img)

    res2 = analyzer.evaluate_image(
        image_path=flat_img,
        page_text="Page text without any figure",
        doc_title="Manual",
        page_num=2,
        width=300,
        height=300
    )
    assert not res2.is_relevant, "Flat monochrome block should be rejected"

def test_image_analyzer_accepts_conceptual_diagram(tmp_path):
    analyzer = ImageConceptAnalyzer()

    # Create a diagram-like image with shapes and high color variance
    diag_img = tmp_path / "diagram.png"
    im = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.rectangle([50, 50, 200, 200], fill=(220, 50, 50), outline=(0, 0, 0), width=3)
    draw.ellipse([150, 100, 350, 250], fill=(50, 120, 220), outline=(0, 0, 0), width=3)
    draw.line([(0, 0), (400, 300)], fill=(0, 200, 50), width=4)
    im.save(diag_img)

    page_text = (
        "La corteza oceánica subduce bajo la continental generando sismicidad profunda. "
        "Figura 4.2: Esquema de la zona de subducción de placas tectónicas y fosa oceánica. "
        "Obsérvese el ángulo de inclinación y la fricción entre placas."
    )

    res = analyzer.evaluate_image(
        image_path=diag_img,
        page_text=page_text,
        doc_title="Geologia Moderna",
        page_num=42,
        width=400,
        height=300
    )

    assert res.is_relevant, "Diagram with figure caption and visual variance must be marked relevant"
    assert "subducción" in res.concept_name.lower() or "figura" in res.caption.lower()
    assert "diagram.png" in res.markdown_tag

def test_generator_interleaves_image_in_chat():
    generator = Gemma4E4B()

    context_chunks = [
        {
            "content": "La subducció tectònica és el procés pel qual una placa s'enfonsa sota una altra.",
            "source_path": "geologia.pdf",
            "page_number": 42,
            "modality": "text",
            "score": 0.88
        },
        {
            "content": "![Esquema de Subducció](images/img_subduccio.png)\n**Figura**: Subducció tectònica",
            "source_path": "geologia.pdf",
            "page_number": 42,
            "modality": "image",
            "asset_rel_path": "img_subduccio.png",
            "metadata": {
                "concept_name": "Esquema de Subducció de Plaques",
                "caption": "Diagrama de la litosfera oceànica entrant sota la continental"
            },
            "score": 0.85
        }
    ]

    response = generator._synthesize_local_response(
        messages=[{"role": "user", "content": "Com funciona la subducció?"}],
        context_chunks=context_chunks
    )

    # Verify that the markdown image is interleaved in the chat output
    assert "![Esquema de Subducció de Plaques](images/img_subduccio.png)" in response
    assert "geologia.pdf" in response
