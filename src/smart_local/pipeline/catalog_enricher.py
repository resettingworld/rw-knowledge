"""
AI Catalog Enricher & Visual Art Director for SmartLocal
Generates rich semantic metadata (short/long descriptions, regional scopes, normatives)
and directs evocative visual styles (watercolor, concept art, artistic collage, cybernetic isometric)
for knowledge capsules and individual book covers.
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFilter

from smart_local.config import settings

# -----------------------------------------------------------------------------
# ARTISTIC VISUAL STYLES: COLLAGE PUNK & DIY ZINE
# -----------------------------------------------------------------------------
COLLAGE_PUNK_DEFAULT = {
    "id": "collage-punk-v1",
    "name": "Collage Punk & Underground DIY Zine",
    "template": (
        "Underground Collage Punk & DIY Zine artwork for '{title}'. Domain: {domain} ({topic}). "
        "Aesthetic: 1980s DIY punk fanzine and Dadaist photomontage. "
        "Visual elements: Ripped xerox photocopier paper edges, gritty black-and-white halftone textures, "
        "vintage scientific engraving cutouts juxtaposed with energetic neon spray paint splatters (hot magenta, acid green, electric cyan, hazard yellow), "
        "distorted stencils, duct tape strips, torn newsprint typography, raw rebellious counter-culture energy, "
        "widescreen 16:9, typography-free composition."
    )
}

ARTISTIC_STYLES = {
    "geo_punk": {
        "id": "geo-punk-v1",
        "name": "Geo-Punk Collage (Ciencias de la Tierra)",
        "template": (
            "Underground Geo-Punk collage artwork for '{title}'. Domain: {domain} ({topic}). "
            "Aesthetic: Punk fanzine and Dadaist cut-and-paste photomontage. "
            "Visual elements: Ripped monochrome geological cross-sections of Earth's mantle and fault lines, "
            "vintage seismograph needle engravings, erupting volcano silhouette collaged with radioactive neon magma spray paint (acid orange #ff5500 and electric cyan #00f0ff), "
            "xerox dot-matrix halftone grain, masking tape strips, torn newsprint textures, 16:9, typography-free."
        )
    },
    "cyber_mechanic_punk": {
        "id": "cyber-mechanic-punk-v1",
        "name": "Cyber-Mechanic Punk (Ciencias Aplicadas)",
        "template": (
            "Industrial Cyber-Mechanic Punk collage for '{title}'. Domain: {domain} ({topic}). "
            "Aesthetic: Post-industrial DIY zine and cybernetic scrap-metal montage. "
            "Visual elements: Ripped blueprint schematics, high-contrast mechanical brass cogs and clockwork gears, "
            "refractive glass prism splitting laser beams, bionic hand cutouts, fluorescent hazard yellow and hot magenta spray stencils, "
            "photocopier streaks, electrical duct tape, raw tactile DIY printmaking, 16:9, typography-free."
        )
    },
    "bio_punk": {
        "id": "bio-punk-v1",
        "name": "Bio-Punk Zine (Ciencias Naturales)",
        "template": (
            "Bio-Punk underground collage artwork for '{title}'. Domain: {domain} ({topic}). "
            "Aesthetic: Radical ecological zine and mutant botanical montage. "
            "Visual elements: High-contrast xeroxed 19th-century Ernst Haeckel biological illustrations and deep-sea radiolarians, "
            "cellular mitosis diagrams violently collaged with toxic acid green (#39ff14) and shocking pink spray paint, "
            "ripped kraft paper edges, halftone screentone patterns, grunge printmaking textures, 16:9, typography-free."
        )
    },
    "anarcho_social_punk": {
        "id": "anarcho-social-punk-v1",
        "name": "Anarcho-Social Punk (Ciencias Sociales)",
        "template": (
            "Agitprop Anarcho-Social Punk collage for '{title}'. Domain: {domain} ({topic}). "
            "Aesthetic: Revolutionary street poster, constructivist montage and punk fanzine. "
            "Visual elements: Ripped archival newspaper headlines and legal code documents, high-contrast stencil silhouettes "
            "of human assemblies and civic forums, symbolic scales of justice cut from textured cardboard, "
            "bold crimson red and gold spray graffiti, distressed xerox photocopier degradation, tape stickers, 16:9, typography-free."
        )
    },
    "hacker_punk": {
        "id": "hacker-punk-v1",
        "name": "Hacker-Punk Cyberzine (Informática y Algoritmos)",
        "template": (
            "Cyberpunk DIY Hacker-Zine collage for '{title}'. Domain: {domain} ({topic}). "
            "Aesthetic: Underground netrunner zine and glitch photocopier montage. "
            "Visual elements: Ripped motherboard PCB circuit traces, xeroxed binary punchcards, glowing crystalline decision tree cutouts "
            "splattered with electric cyan (#00f0ff) and radioactive green spray paint, CRT monitor scanlines, silver duct tape, "
            "Dadaist algorithmic fragments on deep obsidian void, 16:9, typography-free."
        )
    },
    "general_punk": COLLAGE_PUNK_DEFAULT
}

class EnrichedBookRecord(BaseModel):
    id: str
    title: str
    author: str
    filename: str
    source_url: Optional[str] = None
    file_size_bytes: int = 0
    sha256: str
    total_pages: int = 1
    short_description: str
    long_description: str
    regional_scope: str
    normatives_and_standards: List[str] = Field(default_factory=list)
    academic_level: str
    document_type: str
    target_audience: str
    language: str = "es"
    cover_image: Dict[str, Any] = Field(default_factory=dict)
    key_thematic_axes: List[str] = Field(default_factory=list)

class CatalogEnricher:
    """
    Enriches book records by analyzing the front matter text and directing artistic visual styles.
    """
    def __init__(self, ollama_base_url: str = settings.ollama_base_url, model_name: str = settings.generator_model):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model_name = model_name

    def _select_style_for_topic(self, topic: str, domain: str) -> Dict[str, str]:
        t = (topic + " " + domain).lower()
        if "tierra" in t or "geolog" in t or "sismo" in t or "hidrolog" in t or "volcan" in t:
            return ARTISTIC_STYLES["geo_punk"]
        if "aplicad" in t or "ingenier" in t or "tecnolog" in t or "fisic" in t or "proces" in t:
            return ARTISTIC_STYLES["cyber_mechanic_punk"]
        if "natural" in t or "biolog" in t or "ecolog" in t or "planetar" in t or "ecosistem" in t:
            return ARTISTIC_STYLES["bio_punk"]
        if "social" in t or "human" in t or "sociolog" in t or "derecho" in t or "politic" in t:
            return ARTISTIC_STYLES["anarcho_social_punk"]
        if "algorit" in t or "informat" in t or "comput" in t or "dades" in t or "program" in t or "software" in t:
            return ARTISTIC_STYLES["hacker_punk"]
        return ARTISTIC_STYLES["general_punk"]

    def enrich_book_front_matter(
        self,
        front_matter_data: Dict[str, Any],
        topic_name: str = "Ciencias",
        domain: str = "Ciencias Exactas y Naturales",
        covers_dir: Optional[Path] = None,
        custom_art_path: Optional[Path] = None
    ) -> EnrichedBookRecord:
        """
        Takes raw front-matter and generates an enriched catalog entry with artistic visual direction.
        """
        filename = front_matter_data["filename"]
        doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(filename).stem)
        text = front_matter_data.get("front_matter_text", "")
        title_guess = front_matter_data.get("title_guess") or Path(filename).stem.replace("-", " ").title()
        author_guess = front_matter_data.get("author_guess") or "Institucional / InfoLibros"

        # 1. Regional Scope
        regional_scope = self._detect_regional_scope(text)

        # 2. Normatives and Standards
        normatives = self._detect_normatives(text)

        # 3. Academic Level & Document Type
        academic_level = self._detect_academic_level(text)
        doc_type = self._detect_document_type(text)
        target_audience = self._detect_target_audience(text, topic_name, academic_level)

        # 4. Descriptions
        short_desc, long_desc = self._generate_descriptions(title_guess, topic_name, domain, text, doc_type, regional_scope)

        # 5. Artistic Visual Direction
        selected_style = self._select_style_for_topic(topic_name, domain)
        specific_prompt = selected_style["template"].format(
            title=title_guess,
            domain=domain,
            topic=topic_name
        )

        cover_img_data = {
            "style_id": selected_style["id"],
            "style_name": selected_style["name"],
            "prompt_template": selected_style["template"],
            "specific_prompt": specific_prompt,
            "generator": "banana-pi",
            "rel_path": f"images/covers/{doc_id}.png",
            "width": 1024,
            "height": 576
        }

        # 6. Render artistic artwork if covers_dir is provided
        if covers_dir:
            covers_dir = Path(covers_dir)
            covers_dir.mkdir(parents=True, exist_ok=True)
            cover_file = covers_dir / f"{doc_id}.png"

            if custom_art_path and Path(custom_art_path).exists():
                import shutil
                shutil.copy2(custom_art_path, cover_file)
            elif not cover_file.exists():
                self._render_artistic_canvas(
                    title=title_guess,
                    topic=topic_name,
                    style_id=selected_style["id"],
                    region=regional_scope,
                    dest_path=cover_file
                )

        return EnrichedBookRecord(
            id=doc_id,
            title=title_guess,
            author=author_guess,
            filename=filename,
            source_url=None,
            file_size_bytes=front_matter_data.get("file_size_bytes", 0),
            sha256=front_matter_data.get("sha256", ""),
            total_pages=front_matter_data.get("total_pages", 1),
            short_description=short_desc,
            long_description=long_desc,
            regional_scope=regional_scope,
            normatives_and_standards=normatives,
            academic_level=academic_level,
            document_type=doc_type,
            target_audience=target_audience,
            language="es",
            cover_image=cover_img_data,
            key_thematic_axes=[topic_name, doc_type, regional_scope.split("(")[0].strip()]
        )

    def _render_artistic_canvas(self, title: str, topic: str, style_id: str, region: str, dest_path: Path):
        """
        Creates an authentic, high-impact 'Collage Punk' artwork canvas:
        Underground fanzine DIY aesthetic with ripped photocopier paper edges, high-contrast halftones,
        vibrant neon spray splatters (magenta/acid green/cyan), simulated duct tape strips,
        and distressed ransom/stencil cut-and-paste typography.
        """
        width, height = 1024, 576

        # Check if master capsule artistic image exists
        master_banner = None
        if dest_path:
            cand1 = dest_path.parent.parent / "banner.jpg"
            cand2 = dest_path.parent.parent / "banner.png"
            if cand1.exists():
                master_banner = cand1
            elif cand2.exists():
                master_banner = cand2

        # Punk color palettes
        if "geo" in style_id:
            bg_color = (15, 15, 18)          # Deep asphalt void
            neon_spray = (255, 85, 0)        # Radioactive magma orange
            accent_secondary = (0, 240, 255) # Electric cyan
            accent_tertiary = (255, 230, 0)  # Hazard yellow
        elif "mechanic" in style_id:
            bg_color = (18, 18, 20)
            neon_spray = (255, 230, 0)       # Hazard yellow
            accent_secondary = (0, 240, 255) # Blueprint cyan
            accent_tertiary = (255, 0, 127)  # Hot magenta
        elif "bio" in style_id:
            bg_color = (13, 19, 14)
            neon_spray = (57, 255, 20)       # Toxic acid green
            accent_secondary = (255, 0, 127) # Shocking pink
            accent_tertiary = (0, 240, 255)  # Bioluminescent cyan
        elif "social" in style_id:
            bg_color = (20, 16, 16)
            neon_spray = (225, 29, 72)       # Agitprop red
            accent_secondary = (251, 191, 36)# Gold
            accent_tertiary = (255, 255, 255)# White
        elif "hacker" in style_id:
            bg_color = (9, 13, 22)
            neon_spray = (0, 240, 255)       # Cyber cyan
            accent_secondary = (57, 255, 20) # Hacker green
            accent_tertiary = (168, 85, 247) # Neon violet
        else: # Default Punk
            bg_color = (13, 13, 13)
            neon_spray = (255, 0, 127)       # Hot punk pink
            accent_secondary = (57, 255, 20) # Acid green
            accent_tertiary = (0, 240, 255)  # Electric cyan

        if master_banner:
            base_canvas = Image.open(master_banner).convert("RGBA")
            base_canvas = base_canvas.resize((width, height), Image.Resampling.LANCZOS)
            # Apply gritty fanzine dark contrast overlay
            overlay = Image.new("RGBA", (width, height), (10, 10, 12, 110))
            img = Image.alpha_composite(base_canvas, overlay).convert("RGB")
            draw = ImageDraw.Draw(img)
        else:
            img = Image.new("RGB", (width, height), color=bg_color)
            draw = ImageDraw.Draw(img)

            # 1. Gritty photocopy xerox noise / halftones in background
            for gx in range(0, width, 24):
                for gy in range(0, height, 24):
                    dot_radius = 1 if ((gx + gy) % 48 == 0) else 2
                    draw.ellipse(
                        [(gx, gy), (gx + dot_radius, gy + dot_radius)],
                        fill=(45, 45, 52)
                    )

            # 2. Ripped collage paper diagonal slabs with jagged edges
            # Top-left torn paper slab (newsprint off-white)
            rip1 = [
                (0, 0), (320, 0), (280, 45), (295, 90), (250, 140),
                (265, 180), (210, 230), (0, 260)
            ]
            draw.polygon(rip1, fill=(235, 235, 235))
            # Ripped edge drop shadow
            draw.line(rip1[1:], fill=(20, 20, 20), width=3)

            # Bottom-right torn kraft cardboard slab
            rip2 = [
                (width, height), (width - 340, height), (width - 300, height - 50),
                (width - 315, height - 95), (width - 270, height - 145),
                (width - 280, height - 190), (width, height - 210)
            ]
            draw.polygon(rip2, fill=(180, 145, 105))
            draw.line(rip2[1:], fill=(20, 20, 20), width=3)

            # 3. Dynamic neon spray paint splatters
            cx, cy = width // 2, height // 2
            # Spray circle 1
            draw.ellipse(
                [(cx - 280, cy - 140), (cx - 160, cy - 20)],
                outline=neon_spray, width=3
            )
            for s in range(12):
                sx = int(cx - 220 + math.cos(s * 0.5) * (70 + (s * 8) % 35))
                sy = int(cy - 80 + math.sin(s * 0.5) * (70 + (s * 7) % 35))
                draw.ellipse([(sx - 3, sy - 3), (sx + 3, sy + 3)], fill=neon_spray)

            # Spray circle 2 (secondary neon)
            draw.ellipse(
                [(cx + 170, cy + 30), (cx + 280, cy + 140)],
                outline=accent_secondary, width=2
            )
            for s in range(10):
                sx = int(cx + 225 + math.cos(s * 0.6) * (60 + (s * 9) % 30))
                sy = int(cy + 85 + math.sin(s * 0.6) * (60 + (s * 6) % 30))
                draw.ellipse([(sx - 2, sy - 2), (sx + 2, sy + 2)], fill=accent_secondary)

            # 4. Simulated Duct Tape Strips across corners
            # Top-right tape
            tape1 = [(width - 160, 20), (width - 30, 90), (width - 50, 125), (width - 180, 55)]
            draw.polygon(tape1, fill=(234, 179, 8, 180), outline=(202, 138, 4))
            # Bottom-left tape
            tape2 = [(40, height - 110), (160, height - 40), (140, height - 10), (20, height - 80)]
            draw.polygon(tape2, fill=(200, 200, 200), outline=(120, 120, 120))

        cx, cy = width // 2, height // 2
        # 5. Ransom / Punk Stencil Centerplate for title
        plate_box = [cx - 370, cy - 95, cx + 370, cy + 95]
        # Distressed black cardboard plate with high-contrast neon border
        draw.rectangle(plate_box, fill=(10, 10, 14), outline=neon_spray, width=3)
        # Inner dashed stencil border
        draw.rectangle([cx - 364, cy - 89, cx + 364, cy + 89], outline=accent_secondary, width=1)

        # Stencil tag
        draw.text(
            (cx, cy - 60),
            f"/// {topic.upper()} [PUNK-ZINE] ///",
            fill=accent_tertiary,
            anchor="mm"
        )
        clean_title = title if len(title) <= 48 else title[:45] + "..."
        draw.text(
            (cx, cy - 12),
            clean_title.upper(),
            fill=(255, 255, 255),
            anchor="mm"
        )
        draw.text(
            (cx, cy + 35),
            f"★ DIY EDITION • {region[:40]} ★",
            fill=(180, 180, 195),
            anchor="mm"
        )
        draw.text(
            (cx, cy + 68),
            f"ESTIL: {style_id.upper()} | DADAIST XEROX COLLAGE",
            fill=neon_spray,
            anchor="mm"
        )

        img.save(dest_path, format="PNG", optimize=True)

        img.save(dest_path, format="PNG", optimize=True)

    def _detect_regional_scope(self, text: str) -> str:
        t = text.lower()
        if "universidad nacional de la pampa" in t or "la pampa" in t or "edunlpam" in t:
            return "Argentina (Universidad Nacional de La Pampa)"
        if "universidad nacional de tucuman" in t or "tucumán" in t or "tucuman" in t:
            return "Argentina (Universidad Nacional de Tucumán)"
        if "universidad nacional de la plata" in t or "la plata" in t or "unlp" in t:
            return "Argentina (Universidad Nacional de La Plata)"
        if "buenos aires" in t or "uba" in t:
            return "Argentina (Universidad de Buenos Aires)"
        if "argentina" in t:
            return "Argentina"
        if "universidad veracruzana" in t or "xalapa" in t or "veracruz" in t:
            return "Mèxic (Universidad Veracruzana)"
        if "unam" in t or "universidad nacional autónoma de méxico" in t:
            return "Mèxic (UNAM)"
        if "méxico" in t or "mexico" in t or "conacyt" in t or "semarnat" in t:
            return "Mèxic"
        if "universidad de chile" in t or "chile" in t:
            return "Xile"
        if "colombia" in t or "bogota" in t or "bogotá" in t or "universidad nacional de colombia" in t:
            return "Colòmbia"
        if "venezuela" in t or "upel" in t or "caracas" in t:
            return "Veneçuela"
        if "perú" in t or "peru" in t or "lima" in t:
            return "Perú"
        if "universidad del país vasco" in t or "euskal herriko" in t:
            return "Espanya (Universidad del País Vasco / EHU)"
        if "madrid" in t or "barcelona" in t or "españa" in t or "espanya" in t:
            return "Espanya"
        return "Internacional / Àmbit Global en Llengua Espanyola"

    def _detect_normatives(self, text: str) -> List[str]:
        normatives = []
        patterns = [
            r'\bISO\s*[-–]?\s*\d{3,5}(?:[-:]\d+)?\b',
            r'\bNOM\s*[-–]?\s*\d{3}(?:[-/]\w+)?\b',
            r'\bUNE\s*[-–]?\s*\d{3,5}\b',
            r'\bCOVENIN\s*[-–]?\s*\d{3,5}\b',
            r'\bIRAM\s*[-–]?\s*\d{3,5}\b',
            r'\b(?:Ley|Decreto|Código)\s+(?:Nacional|Federal|General|Civil|Penal|de\s+[A-ZÁÉÍÓÚ][a-z]+)\b',
            r'\bISBN\s*[\d-]{10,17}\b'
        ]
        for p in patterns:
            matches = re.findall(p, text, re.IGNORECASE)
            for m in matches:
                clean = m.strip()
                if clean not in normatives:
                    normatives.append(clean)
        if not normatives:
            normatives = ["Bones Pràctiques Acadèmiques i Estàndards Científics Generals"]
        return normatives[:5]

    def _detect_academic_level(self, text: str) -> str:
        t = text.lower()
        if "posgrado" in t or "maestría" in t or "doctorado" in t or "investigación avanzada" in t:
            return "Postgrau / Recerca Avançada"
        if "libro de texto para estudiantes universitarios" in t or "cátedra" in t or "licenciatura" in t or "facultad" in t:
            return "Grau Universitari / Educació Superior"
        if "educación secundaria" in t or "bachillerato" in t or "5to año" in t or "medio" in t:
            return "Educació Secundària / Batxillerat"
        return "Grau Universitari i Consulta Professional"

    def _detect_document_type(self, text: str) -> str:
        t = text.lower()
        if "apuntes de" in t or "apuntes de cátedra" in t:
            return "Apunts de Càtedra Universitària"
        if "manual de" in t or "manual práctico" in t:
            return "Manual Docent i Tècnic"
        if "libro de texto" in t:
            return "Llibre de Text Universitari"
        if "problemas y algoritmos" in t or "ejercicios" in t or "problemario" in t:
            return "Compendi d'Exercicis i Problemes Resolts"
        if "tesis" in t or "disertación" in t:
            return "Tesi Acadèmica"
        return "Tractat i Manual Monogràfic"

    def _detect_target_audience(self, text: str, topic: str, level: str) -> str:
        if "ingeniería" in text.lower() or "ingenieria" in text.lower():
            return f"Estudiants d'enginyeria, tècnics i investigadors en {topic}."
        if "geodesia" in text.lower() or "geofísica" in text.lower() or "topografía" in text.lower():
            return "Estudiants de Geodèsia, Geofísica, Geologia i Ciències Ambientals."
        return f"Estudiants de {level.lower()}, docents i investigadors en l'àmbit de {topic}."

    def _generate_descriptions(self, title: str, topic: str, domain: str, text: str, doc_type: str, region: str) -> Tuple[str, str]:
        first_content = ""
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 30 and not any(k in line.lower() for k in ["universidad", "rector", "isbn", "edición", "secretario"]):
                first_content = line
                break
        if not first_content:
            first_content = f"Estudi exhaustiu de {title} amb fonaments teòrics i aplicacions pràctiques."

        short_desc = f"{doc_type} d'abast a {region} que aprofundeix en els fonaments i desenvolupament de {title} dins de {topic}."

        long_desc = (
            f"Aquesta obra constitueix un {doc_type.lower()} centrat en la disciplina de {topic} ({domain}), "
            f"amb especial incidència en el marc acadèmic i professional de {region}. "
            f"El document aborda de manera sistemàtica els conceptes teòrics clau: {first_content} "
            f"L'enfocament metodològic combina explicacions analítiques, desenvolupament conceptual per capítols i referències tècniques, "
            f"oferint una base sòlida per a l'aprenentatge modular i la integració en sistemes de recuperació augmentada de coneixement (RAG)."
        )
        return short_desc, long_desc
