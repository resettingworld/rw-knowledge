import { Agent } from '@mastra/core/agent';
import { createTool } from '@mastra/core/tools';
import { createOpenAI } from '@ai-sdk/openai';
import { z } from 'zod';

const SMART_LOCAL_API_URL = process.env.SMART_LOCAL_API_URL || 'http://localhost:8800';

const localAI = createOpenAI({
  baseURL: `${SMART_LOCAL_API_URL}/v1`,
  apiKey: 'sk-smartlocal-local'
});

// Tool to search the Spanish Survival Digital Library
export const searchSurvivalManualsTool = createTool({
  id: 'searchSurvivalManuals',
  description: 'Busca fragmentos específicos, diagramas, esquemas y páginas exactas dentro de la biblioteca de manuales de supervivencia y primeros auxilios en español.',
  inputSchema: z.object({
    query: z.string().describe('Consulta de búsqueda para los manuales de supervivencia'),
    top_k: z.number().optional().default(4).describe('Número de fragmentos relevantes a recuperar')
  }),
  execute: async ({ context }) => {
    try {
      const response = await fetch(`${SMART_LOCAL_API_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma-4-e4b:supervivencia-es',
          messages: [{ role: 'user', content: context.query }],
          temperature: 0.2,
          max_tokens: 300,
          top_k: context.top_k || 4
        })
      });

      if (!response.ok) {
        return { error: `Error en la llamada a la API: ${response.statusText}` };
      }

      const data = await response.json();
      return {
        answer_preview: data.choices?.[0]?.message?.content,
        citations: data.citations || []
      };
    } catch (err: any) {
      return { error: `Fallo al buscar en la biblioteca: ${err.message}` };
    }
  }
});

// Mastra Agent with Gemma 4 E4B + EmbeddingGemma 2 RAG
export const kevAgent = new Agent({
  name: 'Kev',
  instructions: `
Ets en Kev, l'Agent Intel·ligent Multimodal de la Biblioteca Digital SmartLocal.
El teu coneixement prové de tots els manuals oficials, tractats i llibres tècnics de la biblioteca digital (Supervivència, Primers Auxilis, Ciències, Informàtica, Algorísmia, Filosofia, Història, etc.).

NORMES CRÍTIQUES:
1. Identifica't sempre com a en Kev si et demanen qui ets.
2. Respon SEMPRE en l'idioma en què et parla l'usuari (si et parla en català, respon en català clar i natural; si en castellà, en castellà).
3. No cal que l'usuari esculli cap paquet de coneixement: tu analitzes automàticament tots els manuals de la biblioteca que siguin pertinents.
4. Per a CADA consell, concepte tècnic o instrucció concreta, HAS DE CITAR LA FONT EXACTA amb el nom del llibre/manual i el número de pàgina: [📖 Manual: <nom_manual>.pdf, Pàg. <numero>].
5. Si la informació prové d'un diagrama o esquema extret, cita [🖼️ Diagrama/Imatge: ...].
6. Estructura la resposta en passos clars (1, 2, 3), explicacions fonamentades i advertències si escau.
7. Afegeix SEMPRE al final la secció '📚 Referències Consultades' amb la llista de fonts i pàgines emprades.
`,
  model: localAI.chat('gemma-4-e4b:all'),
  tools: {
    searchManuals: searchSurvivalManualsTool
  }
});

// Backwards compatibility alias
export const survivalAgent = kevAgent;
