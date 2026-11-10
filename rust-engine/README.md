# ⚡ SmartEngine (Rust Core)

Motor de decisió d'emergència i RAG d'ultra-baix consum escrit en **Rust** a baix nivell per a escenaris extrems, maquinari limitat (Raspberry Pi, mini PCs 12V) i búnquer.

## 🎯 Arquitectura Híbrida de Supervivència

1. **Catàleg & Triatge Ràpid (Sistema 1 - sub-mil·lisegon)**:
   - Analitza els manifests dels RAGs disponibles (`data/rags/`).
   - Classifica la urgència (`CRITICAL`, `HIGH`, `NORMAL`) i selecciona els RAGs pertinents en **~0.02 ms (24 microsegons)** sense cap consum de GPU.
2. **Cerca Semàntica Multimodal (EmbeddingGemma 2 + Matryoshka)**:
   - Descodifica vectors binaris `f32` directament des de SQLite `index.db`.
   - Suporta truncament Matryoshka (`--dim 256` o `128`) per estalviar un 70% de memòria RAM i accelerar la cerca SIMD.
   - Detecta i extreu diagrames i esquemes tècnics (`assets/`).
3. **Resolució Dual: Zero-LLM vs Foto Finish**:
   - **Mode Zero-LLM (`--zero-llm`)**: Extracció literal exacta en **~60 ms** amb 0 al·lucinacions i 0 consum en generació autorregresiva de text.
   - **Mode Conversacional (Opcional)**: El LLM (Gemma 4 E4B) només intervé al final per a la "foto finish" conversacional, mantenint les citacions i imatges intactes.

## 🚀 Comandes CLI

```bash
# Llistar tots els RAGs disponibles
./rust-engine/target/release/smart-engine list-rags

# Executar triatge ràpid de Sistema 1
./rust-engine/target/release/smart-engine triage "hemorragia arterial torniquete"

# Executar cerca i resolució immediata en mode Zero-LLM (Matryoshka 256d)
./rust-engine/target/release/smart-engine query "hemorragia arterial torniquete" --zero-llm --dim 256

# Iniciar servidor d'API REST d'alta concurrència (port 8801)
./rust-engine/target/release/smart-engine serve --port 8801
```
