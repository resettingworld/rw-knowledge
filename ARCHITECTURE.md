# System Architecture — Resetting World

## 1. High-Level Topology

```
                        +-------------------------------+
                        |   resettingworld.com Portal   |
                        |  (web/ - CA / ES / EN - 8085) |
                        +---------------+---------------+
                                        | distributes .zip capsules
                                        v
+-------------------------------------------------------------------------------+
|                           OFF-GRID COMPUTE NODE                               |
|             (Raspberry Pi 5  |  Laptop  |  Android/Termux)                    |
|                                                                               |
|  +-----------------------+     +------------------------+                     |
|  |   Mastra Chat UI      |     |   OpenAI / HTTP API    |                     |
|  |   (mastra-app:3000)   |---->|   smart-local (8800)   |                     |
|  +-----------------------+     |   smart-engine (8801)  |                     |
|                                +-----------+------------+                     |
|                                            |                                  |
|                        +-------------------+-------------------+              |
|                        |                                       |              |
|                        v                                       v              |
|             +---------------------+                 +----------------------+  |
|             |   rw-knowledge      |                 |    rw-protocol       |  |
|             | SQLite index.db     |<--------------->| DTN Envelope (CBOR)  |  |
|             | FTS5 + 768d/256d    | sync chunks     | Ed25519 Signed       |  |
|             | Preserved PDFs      |                 | LoRa / BLE / Wi-Fi   |  |
|             +---------------------+                 +----------+-----------+  |
|                                                                |              |
+----------------------------------------------------------------|--------------+
                                                                 v
                                                      +---------------------+
                                                      |     rw-hardware     |
                                                      | SX1262 LoRa Radio   |
                                                      | 12V LiFePO4 + MPPT  |
                                                      +---------------------+
```

---

## 2. Knowledge Layer (`src/smart_local/` & `rust-engine/`)

### Storage Layout (`data/rags/<rag_id>/`)
- `index.db`: SQLite database holding document metadata, page-level text chunks, extracted visual asset references, FTS5 full-text indexes, and `f32` embedding vectors.
- `library/`: Intact original PDF files verified by SHA-256 checksum.
- `assets/`: Extracted diagrams, tables, and schemas rendered from PDFs.
- `metadata.json` / `manifest.json` / `catalog.json`: Standardized schemas (`KnowledgeAreaMeta`, `KnowledgePackManifest`).

### Dual Runtime Strategy
- **Python Runtime (`smart-local`, Port 8800)**: Handles capsule creation, front-matter extraction, Collage Punk cover generation, Ollama inference (`gemma-4-e4b`), and LRU cache management (`max_cached_rags = 10`).
- **Rust Runtime (`smart-engine`, Port 8801)**: Memory-lean binary for edge SBCs. Performs System 1 keyword triage in microseconds and cosine search over Matryoshka-truncated vectors (`256d` / `128d`) with an optional `--zero-llm` verbatim extraction mode.

---

## 3. Communication Layer (`repos/rw-protocol/`)
- Asynchronous store-and-forward bundles encapsulated with Ed25519 signatures and CBOR serialization.
- Built to tolerate intermittent LoRa (868/915 MHz), BLE, Wi-Fi Direct, and physical USB Sneakernet handoffs.

---

## 4. Hardware Layer (`repos/rw-hardware/`)
- Designed around a 5W–15W continuous power budget using LiFePO4 12.8V chemistry, MPPT solar charging, inline DC fusing, and passive aluminum conduction cooling.
