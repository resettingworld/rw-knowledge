# Knowledge Capsule Format Specification (`rw-knowledge`)

## 1. Overview
A **Resetting World Knowledge Capsule** is a self-contained `.zip` or `.gpk` archive engineered to make technical books searchable offline without re-indexing on the target device.

## 2. Hard Constraints
- **Book Limit**: Maximum **50 PDFs per capsule** ($\le 50$). Larger topics must be split into domain sub-capsules.
- **Content Policy**: Non-fiction, technical, scientific, medical, agricultural, and philosophical works only. Fiction, drama, and poetry are strictly excluded.
- **Visual Style**: Covers and banners follow the canonical **Collage Punk** (`collage-punk-v1`) specification.

## 3. Archive Directory Structure

```text
<capsule-id>-v<version>.zip
├── metadata.json          # KnowledgeAreaMeta (domain, portal source, stats, BookMeta[])
├── manifest.json          # KnowledgePackManifest (rag_id, chunks, modalities, sha256)
├── catalog.json           # Enriched book catalog (short/long desc, region, normatives, covers)
├── index.db               # Pre-computed SQLite DB (chunks, FTS5, 768-d f32 vectors)
├── books/                 # Preserved intact original PDF documents
│   └── *.pdf
└── images/
    ├── banner.png         # Capsule Collage Punk banner (16:9)
    ├── covers/*.png       # Per-book Collage Punk covers
    └── img_*.png|jpeg     # Extracted PDF diagrams and scientific figures
```

## 4. Vector Compatibility
- Vectors are stored as little-endian IEEE 754 `f32` blobs at **768 dimensions** (`EmbeddingGemma 2`).
- Compliant readers MAY truncate vectors to leading **256** or **128** dimensions (Matryoshka Representation Learning) to cut RAM usage by up to 70% on SBCs like Raspberry Pi 5.
