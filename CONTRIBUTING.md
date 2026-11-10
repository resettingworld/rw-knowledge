# Contributing to `rw-knowledge`

Thank you for helping preserve and structure human technical knowledge for offline survival and scientific autonomy.

## Ways to Contribute

1. **Curate & Vet Knowledge Sources**:
   - Propose vetted open-domain or Creative Commons libraries (agronomy, medicine, water sanitation, electrical engineering, physics, math, carpentry).
   - Strictly exclude novels, poetry, drama, and commercial fiction.
2. **Enforce Capsule Standards**:
   - **Hard Limit**: Maximum **50 books per capsule** (`<= 50`). Partition larger domains into focused sub-capsules.
   - **Required Artifacts**: Every capsule must include `metadata.json`, `manifest.json`, `index.db`, `catalog.json`, preserved originals in `books/`, and extracted figures in `images/`.
   - **Visual Style**: Cover art and catalog banners must conform to the canonical **Collage Punk** (`collage-punk-v1`) specification.
3. **Engine & Performance Improvements**:
   - Optimize Matryoshka vector truncation (`768d -> 256d/128d`), SQLite FTS5 hybrid scoring, or ARM64 SIMD operations in `rust-engine`.
   - Ensure `smart-local serve` (port `8800`) and `smart-engine serve` (port `8801`) run with zero external network calls.

## Running Tests

```bash
# Python runtime tests
uv run pytest

# Rust low-power engine tests
cd rust-engine && cargo test
```
