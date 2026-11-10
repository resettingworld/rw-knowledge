---
name: book-cataloging
description: Protocol i eines per analitzar el front-matter de llibres en PDF (2-3 primeres pàgines), classificar-los per àmbit regional, normatives i nivell acadèmic, i generar el catàleg enriquit (catalog.json) amb estil visual canònic per als paquets RAG.
---

# 📚 Skill: Catalogació Intel·ligent de Llibres i Front-Matter per a RAGs

Aquesta skill descriu com un agent d'IA ha d'utilitzar les eines de `smart-local` per analitzar llibres en PDF, extreure'n les primeres pàgines de crèdits/portada, avaluar el seu contingut acadèmic, geogràfic i legal, i generar un fitxer `catalog.json` estructurat i llest per a ser empaquetat als paquets de coneixement.

---

## 🛠️ Eines CLI Principals

L'agent disposa de 3 comandes integrades a `smart-local`:

### 1. Extreure Front-Matter en Text Brut per a l'Agent
```bash
# Extreu les 2 primeres pàgines no buides de tots els PDFs d'una carpeta
uv run smart-local extract-front-matter <carpeta_pdfs> --pages 2 --output <fitxer_preview.json>
```

**Què fa:**
- Escaneja cada PDF de la carpeta.
- Descarta portades en blanc i extreu el text íntegre de les 2 o 3 primeres pàgines (portada interior, crèdits, editorial, drets d'autor, universitat, consell editor).
- Guarda un JSON amb `filename`, `sha256`, `total_pages`, i `front_matter_text` perquè l'agent pugui llegir-lo directament.

---

### 2. Enriquir i Generar el `catalog.json` Automàticament
```bash
uv run smart-local enrich-catalog <carpeta_pdfs> \
  --topic "Ciencias de la Tierra" \
  --domain "Ciencias Exactas y Naturales" \
  --pages 2 \
  --output <carpeta_pdfs>/catalog.json
```

**Què genera:**
- **`short_description`**: Síntesi de 1 o 2 línies explicant l'abast i objectiu de l'obra.
- **`long_description`**: Explicació docent profunda del document, temàtiques i aplicacions.
- **`regional_scope`**: País, universitat o regió d'origen detectada (ex: `Argentina (Univ. Nacional de Tucumán)`, `Mèxic (UNAM)`, `Espanya`, etc.).
- **`normatives_and_standards`**: Normatives tècniques o marcs legals esmentats (ex: `ISO`, `UNE`, `NOM`, `Ley de Aguas`, `Código Civil`, `IRAM`).
- **`academic_level`**: Grau Universitari, Batxillerat, Postgrau / Recerca o Manual Professional.
- **`document_type`**: Apunts de Càtedra, Llibre de Text, Tesi, Compendi d'Exercicis o Manual Tècnic.
- **`cover_image`**: Informació del prompt canònic d'il·lustració i renderització a `images/covers/<id>.png`.

---

### 3. Visualitzar el Catàleg Formatat
```bash
uv run smart-local show-catalog <carpeta_pdfs>/catalog.json
```
Mostra una taula formatada a la terminal amb els títols, regions, normatives i descripcions sintetitzades.

---

## 🎨 Estil Visual Canònic: Collage Punk (`collage-punk-v1`)

Totes les imatges i portades generades per acompanyar els llibres i els temes segueixen estrictament la identitat visual **Collage Punk / Underground DIY Zine**, amb textures de fotocòpia Xerox, semitons d'alt contrast, vores de paper estripat, esquitxos d'esprai neó (magenta, verd àcid, cian elèctric, groc de perill) i elements retallats:

- **ID de l'Estil Canònic**: `collage-punk-v1`
- **Sub-estils Temàtics**:
  - `geo-punk-v1`: Talls geològics xeroxats, agulles de sismògraf i magma fluorescent (Ciencias de la Tierra).
  - `cyber-mechanic-punk-v1`: Blueprints estripats, engranatges industrials i prismes òptics (Ciencias Aplicadas).
  - `bio-punk-v1`: Gravats botànics i marins d'Ernst Haeckel amb esquitxos verd àcid i rosa xiclet (Ciencias Naturales).
  - `anarcho-social-punk-v1`: Retalls de premsa d'arxiu, siluetes d'assemblea i balances de justícia (Ciencias Sociales).
  - `hacker-punk-v1`: Traces de PCB, targetes perforades i arbres binaris en neó cian (Informática y Algoritmos).

- **Plantilla Canònica de Prompt**:
  ```text
  Underground Collage Punk & DIY Zine artwork for '{title}'.
  Domain: {domain} ({topic}).
  Aesthetic: 1980s DIY punk fanzine and Dadaist photomontage.
  Visual elements: Ripped xerox photocopier paper edges, gritty black-and-white halftone textures,
  vintage scientific engraving cutouts juxtaposed with energetic neon spray paint splatters
  (hot magenta, acid green, electric cyan, hazard yellow), distorted stencils, duct tape strips,
  torn newsprint typography, raw rebellious counter-culture energy, widescreen 16:9, typography-free composition.
  ```
- **Propietats a `catalog.json`**:
  ```json
  "banner_image": {
    "rel_path": "images/banner.jpg",
    "style_id": "collage-punk-v1",
    "style_name": "Collage Punk & Underground DIY Zine",
    "prompt_template": "..."
  },
  "cover_image": {
    "style_id": "geo-punk-v1",
    "style_name": "Geo-Punk Collage (Ciencias de la Tierra)",
    "prompt_template": "...",
    "specific_prompt": "...",
    "generator": "banana-pi",
    "rel_path": "images/covers/nom_document.png",
    "width": 1024,
    "height": 576
  }
  ```

---

## 📦 Estructura del `catalog.json` dins del Paquet

El fitxer `catalog.json` queda persistit a l'arrel de cada carpeta de descàrrega i s'integra automàticament dins del paquet final `.zip`:

```json
{
  "topic": "Ciencias de la Tierra",
  "domain": "Ciencias Exactas y Naturales",
  "canonical_style_id": "smartlocal-academic-minimalist-v1",
  "total_books": 32,
  "total_pages": 4175,
  "total_size_mb": 169.74,
  "books": [
    {
      "id": "apuntes_de_sismologia_luis_a_estrada",
      "title": "Sismologia para Ingenieros",
      "author": "Luis A. Estrada / Universidad Nacional de Tucumán",
      "filename": "apuntes-de-sismologia-luis-a-estrada.pdf",
      "file_size_bytes": 1421054,
      "sha256": "3a4...",
      "total_pages": 52,
      "short_description": "Apunts de Càtedra Universitària d'abast a Argentina...",
      "long_description": "Aquesta obra constitueix un manual docent centrat en la sismologia...",
      "regional_scope": "Argentina (Universidad Nacional de Tucumán)",
      "normatives_and_standards": [
        "Reglamento INPRES-CIRSOC 103",
        "Norma Sísmica Argentina"
      ],
      "academic_level": "Grau Universitari / Educació Superior",
      "document_type": "Apunts de Càtedra Universitària",
      "target_audience": "Estudiants de Geodèsia, Geofísica i Enginyeria Civil.",
      "language": "es",
      "cover_image": { ... }
    }
  ]
}
```

---

## 🔄 Flux de Treball Recomanat per a l'Agent

1. **Llegir els PDFs**: Executar `smart-local extract-front-matter <directori> -o preview.json`.
2. **Revisió de Contingut**: Inspeccionar el camp `front_matter_text` per identificar si hi ha universitats, editorials o legislació específica no detectada automàticament.
3. **Generar Catàleg Enriquit**: Executar `smart-local enrich-catalog <directori> --topic "<nom_tema>" --domain "<macro_domini>"`.
4. **Validació Visual**: Executar `smart-local show-catalog <directori>/catalog.json` per verificar la qualitat abans de procedir a la vectorització i empaquetament final del RAG.

---

## 🏛️ Fonts de Biblioteques Digitals Suportades

El sistema disposa de scrapers i pipelines dedicats per a 4 grans biblioteques obertes:

| Biblioteca | Scraper | Enfocament | Filtració de Ficció |
|---|---|---|---|
| **InfoLibros.org** | `infolibros_scraper.py` | Ciències exactes, enginyeria, informàtica | Per categories temàtiques estructurades |
| **Project Gutenberg** | `gutenberg_scraper.py` | Història de la ciència, matemàtiques, medicina, economia | Exclusió de les 19 prestatgeries de ficció; selecció de 22 vetted technical bookshelves |
| **Textos.info** | `textos_info_scraper.py` | Filosofia, teoria política, assaig científic, tractats | Descàrrega directa de PDF via referer amb blacklisting de paraules clau literàries |
| **PlanetaLibro.net** | `planetalibro_scraper.py` | Estratègia, filosofia, ciència, doctrina política | Conversió dinàmica d'episodis/fragments HTML de lector a PDF vectorial via PyMuPDF |

---

## ⚖️ Regles Arquitectòniques Fonamentals

1. **Límit Màxim de 50 Llibres per Càpsula RAG (`<= 50`)**:
   - Cap paquet o índex SQLite ha de superar els 50 volums per garantir una cerca vectorial ultra ràpida i alta densitat semàntica.
   - Les col·leccions grans s'han de fragmentar en sub-càpsules especialitzades (ex: `part-1`, `part-2` o per especialitats).
2. **Filtre Estricte de No-Ficció**:
   - Queden expressament exclosos gèneres de novel·la, narrativa de ficció, poesia, comèdia i teatre.
   - Es prioritza el coneixement humà tècnic, empíric, filosòfic, estratègic i científic.
3. **Identitat Visual Unificada**:
   - Totes les càpsules han de contenir `catalog.json` i portades generades sota l'estil canònic **Collage Punk** (`collage-punk-v1`).

