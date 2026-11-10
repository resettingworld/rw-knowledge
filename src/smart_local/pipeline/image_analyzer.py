"""
Image Concept Analyzer for Multimodal RAG
Evaluates extracted images from book PDFs using multimodal/LLM reasoning
to identify whether an image is conceptually essential for illustrating key concepts,
generating rich semantic captions and discarding decorative/boilerplate noise.
"""

import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel
from PIL import Image, ImageStat
import httpx

from smart_local.config import settings

class ImageEvaluationResult(BaseModel):
    is_relevant: bool
    concept_name: str = ""
    caption: str = ""
    markdown_tag: str = ""
    confidence: float = 0.0

class ImageConceptAnalyzer:
    """
    Evaluates book PDF images using visual metrics, contextual caption detection,
    and multimodal LLM grounding (Gemma 4 E4B / Ollama Vision).
    """
    def __init__(
        self,
        ollama_base_url: str = settings.ollama_base_url,
        vision_model: str = "gemma-4-e4b",
        fallback_vision_model: str = "llava:latest"
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.vision_model = vision_model
        self.fallback_vision_model = fallback_vision_model

    def evaluate_image(
        self,
        image_path: Path,
        page_text: str,
        doc_title: str,
        page_num: int,
        width: int,
        height: int
    ) -> ImageEvaluationResult:
        """
        Determines whether the image is conceptually necessary and relevant.
        Returns ImageEvaluationResult with concept name and rich caption.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return ImageEvaluationResult(is_relevant=False)

        # 1. Structural & Heuristic Pre-Filtering
        # Minimum informative dimension (must be at least 150x150)
        if width < 150 or height < 150:
            return ImageEvaluationResult(is_relevant=False)

        # Aspect ratio check (reject banner strips, footer lines, or margins)
        aspect_ratio = width / max(height, 1)
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:
            return ImageEvaluationResult(is_relevant=False)

        # Image complexity & entropy check (reject solid colors, gradients, blank boxes)
        try:
            with Image.open(image_path) as im:
                im_rgb = im.convert("RGB")
                stat = ImageStat.Stat(im_rgb)
                # Standard deviation across RGB channels
                std_dev = sum(stat.stddev) / 3.0
                if std_dev < 12.0:  # Almost flat / solid color block
                    return ImageEvaluationResult(is_relevant=False)
        except Exception:
            return ImageEvaluationResult(is_relevant=False)

        # 2. Contextual Caption & Figure Extraction from Page Text
        caption_found, detected_concept = self._extract_caption_from_text(page_text)

        # 3. LLM Multimodal Analysis (if Ollama/LLM is available)
        llm_eval = self._evaluate_with_llm(
            image_path=image_path,
            page_text=page_text,
            doc_title=doc_title,
            page_num=page_num,
            caption_hint=caption_found
        )

        if llm_eval is not None:
            return llm_eval

        # 4. High-Precision Heuristic Fallback
        # If the page mentions a figure/diagram and the image has good visual complexity
        if caption_found:
            concept = detected_concept or f"Figura de {doc_title} (Pàg. {page_num})"
            desc = caption_found
            rel_name = image_path.name
            return ImageEvaluationResult(
                is_relevant=True,
                concept_name=concept,
                caption=desc,
                markdown_tag=f"![{concept}](images/{rel_name})",
                confidence=0.85
            )

        # If image is reasonably large and complex (likely a schematic or photo)
        if (width >= 250 and height >= 200) and len(page_text.strip()) > 100:
            # Extract first relevant sentence of the page as context
            first_sentence = page_text.strip().split(".")[0].strip()
            concept = f"Il·lustració tècnica sobre: {first_sentence[:80]}"
            rel_name = image_path.name
            return ImageEvaluationResult(
                is_relevant=True,
                concept_name=concept,
                caption=f"Diagrama explicatiu associat a: {first_sentence}",
                markdown_tag=f"![{concept}](images/{rel_name})",
                confidence=0.70
            )

        return ImageEvaluationResult(is_relevant=False)

    def _extract_caption_from_text(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Finds captions like 'Figura 2.1: ...', 'Diagrama 3: ...', 'Mapa ...' on the page."""
        if not text:
            return None, None

        caption_pattern = re.compile(
            r'(?:Figura|Fig\.|Diagrama|Esquema|Gráfico|Mapa|Ilustración|Tabla)\s*[\d\.\-]*[:\.]?\s*([^\.\n]+(?:\.[^\.\n]+)?)',
            re.IGNORECASE
        )
        match = caption_pattern.search(text)
        if match:
            raw_caption = match.group(0).strip()
            clean_concept = match.group(1).strip()
            if len(clean_concept) > 5:
                return raw_caption, clean_concept[:100]

        return None, None

    def _evaluate_with_llm(
        self,
        image_path: Path,
        page_text: str,
        doc_title: str,
        page_num: int,
        caption_hint: Optional[str]
    ) -> Optional[ImageEvaluationResult]:
        """Queries LLM with vision or prompt to classify educational relevance."""
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            prompt = (
                f"Analitza si aquesta imatge extreta del llibre '{doc_title}' (Pàgina {page_num}) "
                "és conceptualment necessària i rellevant per il·lustrar un concepte clau (esquema, diagrama, mapa, gràfic, anatomia) "
                "o si és només decorativa / soroll.\n"
                f"Context del text de la pàgina: {page_text[:600]}\n"
                f"Pista de peu d'imatge: {caption_hint or 'cap'}\n\n"
                "Respon ÚNICAMENT en format JSON:\n"
                "{\n"
                '  "is_relevant": true,\n'
                '  "concept_name": "Nom curt del concepte que il·lustra",\n'
                '  "caption": "Explicació detallada de què mostra la imatge per ensenyar a l\'usuari",\n'
                '  "confidence": 0.95\n'
                "}"
            )

            with httpx.Client(timeout=10.0) as client:
                res = client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.vision_model,
                        "prompt": prompt,
                        "images": [img_b64],
                        "stream": False,
                        "format": "json"
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    response_text = data.get("response", "")
                    parsed = json.loads(response_text)
                    is_rel = parsed.get("is_relevant", False)
                    c_name = parsed.get("concept_name", "")
                    c_caption = parsed.get("caption", "")
                    conf = float(parsed.get("confidence", 0.8))
                    rel_name = image_path.name

                    return ImageEvaluationResult(
                        is_relevant=is_rel,
                        concept_name=c_name,
                        caption=c_caption,
                        markdown_tag=f"![{c_name}](images/{rel_name})",
                        confidence=conf
                    )
        except Exception:
            pass

        return None
