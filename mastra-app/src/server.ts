import express from 'express';
import cors from 'cors';
import path from 'path';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { survivalAgent } from './agent';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const SMART_LOCAL_API_URL = process.env.SMART_LOCAL_API_URL || 'http://localhost:8800';
const RUST_ENGINE_URL = process.env.RUST_ENGINE_URL || 'http://localhost:8801';

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 100 * 1024 * 1024 } });

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// ==========================================
// RAG & Knowledge Package Management Routes
// ==========================================

// 1. List all installed knowledge packages (RAGs)
app.get('/api/rags', async (_req, res) => {
  try {
    const response = await fetch(`${SMART_LOCAL_API_URL}/v1/rags`);
    if (!response.ok) throw new Error('Error al consultar paquetes RAG');
    const data = await response.json();
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 2. Create a new knowledge package (RAG)
app.post('/api/rags', async (req, res) => {
  try {
    const { rag_id, name, description, tags } = req.body;
    if (!rag_id) {
      return res.status(400).json({ error: 'El identificador del paquete (rag_id) es obligatorio' });
    }
    const response = await fetch(`${SMART_LOCAL_API_URL}/v1/rags`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rag_id, name, description, tags: tags || [] })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Error al crear paquete');
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 3. Delete an entire knowledge package (RAG)
app.delete('/api/rags/:ragId', async (req, res) => {
  try {
    const { ragId } = req.params;
    const response = await fetch(`${SMART_LOCAL_API_URL}/v1/rags/${ragId}`, {
      method: 'DELETE'
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Error al eliminar paquete');
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 4. Toggle on-demand RAM loading/unloading
app.post('/api/rags/:ragId/load', async (req, res) => {
  try {
    const { ragId } = req.params;
    const response = await fetch(`${SMART_LOCAL_API_URL}/v1/rags/${ragId}/load`, { method: 'POST' });
    const data = await response.json();
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/rags/:ragId/unload', async (req, res) => {
  try {
    const { ragId } = req.params;
    const response = await fetch(`${SMART_LOCAL_API_URL}/v1/rags/${ragId}/unload`, { method: 'POST' });
    const data = await response.json();
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ==========================================
// Digital Library (Books / Documents) Routes
// ==========================================

// 5. List books/documents inside a specific package
app.get('/api/rags/:ragId/library', async (req, res) => {
  try {
    const { ragId } = req.params;
    const response = await fetch(`${SMART_LOCAL_API_URL}/v1/rags/${ragId}/library`);
    if (!response.ok) throw new Error(`Error al obtener libros de ${ragId}`);
    const data = await response.json();
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 6. Download a book's original preserved PDF file
app.get('/api/rags/:ragId/library/download/:docId', async (req, res) => {
  try {
    const { ragId, docId } = req.params;
    const targetUrl = `${SMART_LOCAL_API_URL}/v1/rags/${ragId}/library/${docId}/download`;
    const response = await fetch(targetUrl);
    if (!response.ok) {
      return res.status(response.status).json({ error: 'Fitxer no trobat a la biblioteca' });
    }

    const cd = response.headers.get('content-disposition');
    if (cd) res.setHeader('content-disposition', cd);
    res.setHeader('content-type', response.headers.get('content-type') || 'application/pdf');

    const buffer = await response.arrayBuffer();
    res.send(Buffer.from(buffer));
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 7. Upload and index a new book/document into a package
app.post('/api/rags/:ragId/library/upload', upload.single('file'), async (req, res) => {
  try {
    const { ragId } = req.params;
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: 'No s\'ha seleccionat cap fitxer per pujar' });
    }

    console.log(`[Mastra] Pujant nou llibre a '${ragId}': ${file.originalname} (${(file.size / 1024).toFixed(1)} KB)`);

    // Prepare multipart form data for backend
    const formData = new FormData();
    const blob = new Blob([file.buffer], { type: file.mimetype });
    formData.append('file', blob, file.originalname);

    const targetUrl = `${SMART_LOCAL_API_URL}/v1/rags/${ragId}/library/upload`;
    const response = await fetch(targetUrl, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Error al processar el document');
    res.json(data);
  } catch (err: any) {
    console.error('[Mastra] Error al pujar llibre:', err);
    res.status(500).json({ error: err.message });
  }
});

// 8. Delete a book/document from a knowledge package (updates RAG dimensions)
app.delete('/api/rags/:ragId/library/:docId', async (req, res) => {
  try {
    const { ragId, docId } = req.params;
    console.log(`[Mastra] Eliminant document '${docId}' del paquet '${ragId}'...`);
    const targetUrl = `${SMART_LOCAL_API_URL}/v1/rags/${ragId}/library/${docId}`;
    const response = await fetch(targetUrl, { method: 'DELETE' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Error al eliminar document');
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Fast Sistema 1 Triage Endpoint powered by Rust Engine
app.post('/api/triage', async (req, res) => {
  try {
    const { query } = req.body;
    const response = await fetch(`${RUST_ENGINE_URL}/v1/triage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query || '' })
    });
    if (!response.ok) throw new Error('Error al realitzar triatge ràpid');
    const data = await response.json();
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// View extracted images
app.get(['/api/assets/:assetName', '/api/assets/:ragId/:assetName'], async (req, res) => {
  try {
    const assetName = req.params.assetName || req.params[0] || '';
    const ragId = req.params.ragId || 'supervivencia-es';
    const cleanAssetName = assetName.replace(/^assets\//, '').replace(/^images\//, '');
    const targetUrl = `${SMART_LOCAL_API_URL}/v1/rags/${ragId}/assets/${cleanAssetName}`;
    const response = await fetch(targetUrl);
    if (!response.ok) return res.status(response.status).json({ error: 'Imatge no trobada' });
    res.setHeader('content-type', response.headers.get('content-type') || 'image/png');
    const buffer = await response.arrayBuffer();
    res.send(Buffer.from(buffer));
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ==========================================
// Chat API Endpoint (Hybrid Rust Core + LLM)
// ==========================================

app.post('/api/chat', async (req, res) => {
  try {
    const { message, rag_id, zero_llm, dim } = req.body;
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'El missatge és obligatori.' });
    }

    const selectedRag = rag_id || 'all';
    const isZeroLlm = Boolean(zero_llm);
    const targetDim = Number(dim) || 256;

    console.log(`[Mastra] Kev atenent consulta (zero_llm: ${isZeroLlm}, dim: ${targetDim}, rag: ${selectedRag}): "${message.substring(0, 60)}..."`);

    // 1. Prioritize Native Rust Engine (Kev Sistema 1 Triage + SIMD Matryoshka + Zero-LLM / Foto Finish)
    try {
      const rustResponse = await fetch(`${RUST_ENGINE_URL}/v1/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: message,
          target_rags: selectedRag === 'all' || !selectedRag ? undefined : [selectedRag],
          zero_llm: isZeroLlm,
          dim: targetDim,
          top_k: 4
        })
      });

      if (rustResponse.ok) {
        const engineData = await rustResponse.json();
        const structuredCitations = (engineData.retrieval || []).map((r: any) => ({
          rag_id: r.rag_id,
          source: r.source_path,
          page_number: r.page_number,
          score: r.score,
          citation_label: r.citation
        }));

        return res.json({
          reply: engineData.resolution,
          agent: isZeroLlm ? 'Kev (Mode Búnquer Zero-LLM)' : 'Kev (Gemma 4 E4B + SmartEngine)',
          engine: 'rust-native',
          triage: engineData.triage,
          images: engineData.images || [],
          rag_id: engineData.triage?.selected_rags?.[0]?.rag_id || selectedRag,
          citations: structuredCitations,
          latency_ms: engineData.total_latency_ms,
          zero_llm: engineData.zero_llm,
          timestamp: Date.now()
        });
      }
    } catch (rustErr) {
      console.warn('[Mastra] Rust engine no disponible, fent fallback a Python RAG...', rustErr);
    }

    // 2. Fallback to Python OpenAI-compatible completions endpoint
    const ragResponse = await fetch(`${SMART_LOCAL_API_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: `gemma-4-e4b:${selectedRag}`,
        rag_id: selectedRag,
        messages: [{ role: 'user', content: message }],
        temperature: 0.2,
        max_tokens: 3000,
        top_k: 6
      })
    });

    if (!ragResponse.ok) {
      const errData = await ragResponse.json().catch(() => ({}));
      throw new Error(errData.detail || `Error del model Gemma 4: ${ragResponse.statusText}`);
    }

    const data = await ragResponse.json();
    const replyText = data.choices?.[0]?.message?.content || '';
    const structuredCitations = data.citations || [];

    res.json({
      reply: replyText,
      agent: 'Kev (Gemma 4 E4B)',
      engine: 'python-fallback',
      rag_id: selectedRag,
      citations: structuredCitations,
      timestamp: Date.now()
    });
  } catch (err: any) {
    console.error('[Mastra] Error generant resposta:', err);
    res.status(500).json({ error: err.message || 'Error intern al servidor Mastra' });
  }
});

app.listen(PORT, () => {
  console.log(`=======================================================`);
  console.log(`  🚀 Mastra Survival Agent & Chat UI en marxa!         `);
  console.log(`  URL local: http://localhost:${PORT}                  `);
  console.log(`  Backend RAG: ${SMART_LOCAL_API_URL}                  `);
  console.log(`=======================================================`);
});
