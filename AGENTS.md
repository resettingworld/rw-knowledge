# AGENTS.md — `rw-knowledge`

Instructions for AI coding and curation agents working in **`resettingworld/rw-knowledge`** (Pillar 01: Offline AI & Modular Knowledge Capsules).

---

## 1. Repository Purpose
`rw-knowledge` is the offline knowledge runtime and capsule packaging ecosystem of **Resetting World**. It indexes technical books into self-contained `.zip` / `.gpk` capsules with embedded SQLite FTS5 + vector indexes (`index.db`), preserved PDFs (`books/`), extracted figures (`images/`), and enriched metadata (`catalog.json`, `metadata.json`, `manifest.json`).

---

## 2. Hard Invariants
1. **Zero Cloud at Runtime**: Querying, retrieval, triage, and generation must execute 100% locally (`localhost:11434` Ollama / local quantized models / Rust `--zero-llm` mode) with zero external network calls.
2. **Capsule Size Cap (`<= 50` books)**: No capsule may contain more than 50 books. Partition larger collections by sub-topic or `part-N`.
3. **Technical / Non-Fiction Only**: Strictly filter out novels, drama, poetry, and commercial fiction during scraping/curation.
4. **Collage Punk Visual Standard (`collage-punk-v1`)**: All generated capsule banners and book covers must conform to the canonical `collage-punk-v1` themes (`geo-punk-v1`, `cyber-mechanic-punk-v1`, `bio-punk-v1`, `anarcho-social-punk-v1`, `hacker-punk-v1`).
5. **Original Document Preservation**: Every ingested PDF must remain intact in `books/` inside the capsule and be tracked by SHA-256 hash.

---

## 3. Architecture & Specs
- See [`docs/CAPSULE_SPEC.md`](./docs/CAPSULE_SPEC.md) for the exact ZIP directory layout, SQLite schema expectations, and Matryoshka vector truncation (`768d` -> `256d`/`128d`).
- Supported runtimes:
  - Python FastAPI + MultimodalProcessor (`smart-local`, port `8800`)
  - Rust low-power engine (`smart-engine`, port `8801`, Raspberry Pi 5 target)

---

## 4. Verification Before Commit
- Validate JSON schemas (`metadata.json`, `manifest.json`, `catalog.json`) strictly against Pydantic models.
- Run `pytest` and `cargo test` without mocks hiding missing files or broken imports.
- Ensure temporary `_staging_*` directories are never committed.
