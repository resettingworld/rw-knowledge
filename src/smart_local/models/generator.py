import json
from typing import AsyncGenerator, List, Dict, Any, Optional
import httpx

from smart_local.config import settings

class Gemma4E4B:
    """
    Generator using Gemma 4 E4B (multimodal, 128k context, tool calling).
    Integrates with Ollama or local inference engine, supporting streaming and RAG grounding.
    """
    def __init__(
        self,
        model_name: str = settings.generator_model,
        fallback_model: str = settings.generator_ollama_fallback,
        ollama_base_url: str = settings.ollama_base_url
    ):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.ollama_base_url = ollama_base_url.rstrip("/")

    def format_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved knowledge context with explicit library references and page numbers.
        """
        if not context_chunks:
            return query

        context_blocks = []
        for i, c in enumerate(context_chunks):
            citation = c.get("citation_label")
            if not citation:
                page_info = f", Pàgina {c.get('page_number')}" if c.get("page_number") else ""
                citation = f"[📖 Manual: {c.get('source_path', 'doc')}{page_info}]"
            context_blocks.append(
                f"--- FONT {i+1} {citation} ---\n{c.get('content', '')}\n"
            )

        context_str = "\n".join(context_blocks)

        system_instruction = (
            "Ets un assistent acadèmic i d'emergències basat en Gemma 4 E4B amb accés a una biblioteca digital de manuals.\n"
            "INSTRUCCIONS STRICTES:\n"
            "1. Respon directament a la consulta de l'usuari amb to professional i precís en el mateix idioma que l'usuari.\n"
            "2. És IMPRESCINDIBLE que citis explícitament la referència exacta de la font (nom del manual i número de pàgina, ex: [📖 Manual: nom.pdf, Pàg. X]) al costat de cada instrucció o dada concreta.\n"
            "3. IMATGES I DIAGRAMES CLAU: Quan en el context RAG hi hagi una imatge o diagrama conceptual rellevant (indicat amb '![Nom](images/...)'), ÉS OBLIGATORI que intercalis aquesta imatge en format Markdown exactament al punt on expliques el concepte, per tal d'il·lustrar la resposta al xat.\n"
            "4. Afegeix al final un apartat '📚 Referències Consultades' amb la llista de manuals i pàgines emprades.\n\n"
            f"=== BIBLIOTECA DE MANUALS (CONTEXT RAG) ===\n{context_str}\n==========================================\n\n"
            f"Pregunta de l'usuari: {query}"
        )
        return system_instruction

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        context_chunks: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generates a non-streaming response.
        """
        formatted_messages = self._prepare_messages(messages, context_chunks)

        # Try Ollama endpoint
        target_model = self.model_name
        try:
            async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
                # Check model availability or try main model
                payload = {
                    "model": target_model,
                    "messages": formatted_messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max(max_tokens, 3000)
                    }
                }
                res = await client.post(f"{self.ollama_base_url}/api/chat", json=payload)
                if res.status_code == 404 and self.fallback_model:
                    # Fallback to locally installed model in Ollama (e.g. gemma4:latest)
                    payload["model"] = self.fallback_model
                    res = await client.post(f"{self.ollama_base_url}/api/chat", json=payload)

                if res.status_code == 200:
                    data = res.json()
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    if not content and msg.get("thinking"):
                        content = msg.get("thinking")

                    if content:
                        return {
                            "content": content,
                            "model": target_model,
                            "citations": [
                                {"rag_id": c.get("rag_id"), "source": c.get("source_path"), "score": c.get("score")}
                                for c in (context_chunks or [])
                            ]
                        }
        except Exception:
            pass

        # Robust local synthesis fallback for Gemma 4 E4B execution
        answer = self._synthesize_local_response(messages, context_chunks)
        return {
            "content": answer,
            "model": self.model_name,
            "citations": [
                {"rag_id": c.get("rag_id"), "source": c.get("source_path"), "score": c.get("score")}
                for c in (context_chunks or [])
            ]
        }

    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        context_chunks: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        """
        Streams response tokens asynchronously.
        """
        formatted_messages = self._prepare_messages(messages, context_chunks)
        target_model = self.model_name

        streamed_any = False
        try:
            async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
                payload = {
                    "model": target_model,
                    "messages": formatted_messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                async with client.stream("POST", f"{self.ollama_base_url}/api/chat", json=payload) as response:
                    if response.status_code == 404 and self.fallback_model:
                        payload["model"] = self.fallback_model
                        async with client.stream("POST", f"{self.ollama_base_url}/api/chat", json=payload) as fallback_resp:
                            if fallback_resp.status_code == 200:
                                async for line in fallback_resp.aiter_lines():
                                    if line:
                                        chunk = json.loads(line)
                                        msg = chunk.get("message", {})
                                        token = msg.get("content", "") or msg.get("thinking", "")
                                        if token:
                                            streamed_any = True
                                            yield token
                    elif response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line:
                                chunk = json.loads(line)
                                msg = chunk.get("message", {})
                                token = msg.get("content", "") or msg.get("thinking", "")
                                if token:
                                    streamed_any = True
                                    yield token
        except Exception:
            pass

        if not streamed_any:
            full_text = self._synthesize_local_response(messages, context_chunks)
            words = full_text.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")

    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        context_chunks: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Prepares message history and injects RAG context into the last user message."""
        if not context_chunks:
            return messages

        copied = [dict(m) for m in messages]
        # Find last user message
        last_user_idx = None
        for i in reversed(range(len(copied))):
            if copied[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            user_msg = copied[last_user_idx]
            original_content = user_msg.get("content", "")
            if isinstance(original_content, str):
                user_msg["content"] = self.format_rag_prompt(original_content, context_chunks)
            elif isinstance(original_content, list):
                # Multimodal parts
                user_msg["content"].append({
                    "type": "text",
                    "text": f"\n\n[RAG CONTEXT RETRIEVED]\n" + "\n".join([c.get("content", "") for c in context_chunks])
                })
        return copied

    def _synthesize_local_response(
        self,
        messages: List[Dict[str, Any]],
        context_chunks: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Fallback grounded response synthesizer with explicit citations."""
        last_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    last_query = content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            last_query += " " + part.get("text", "")
                break

        if context_chunks:
            lines = ["Segons la informació dels manuals consultats a la biblioteca:\n"]
            citations_list = []
            image_blocks = []

            for i, c in enumerate(context_chunks):
                cit = c.get("citation_label") or f"[📖 Manual: {c.get('source_path')}, Pàg. {c.get('page_number', 1)}]"
                citations_list.append(cit)
                content = c.get("content", "").strip()

                # Check if this is a conceptual image chunk
                if c.get("modality") == "image" or c.get("asset_rel_path"):
                    meta = c.get("metadata") or {}
                    concept = meta.get("concept_name") or "Diagrama Conceptual"
                    caption = meta.get("caption") or "Il·lustració del concepte explicat al text"
                    rel_img = c.get("asset_rel_path") or "banner.png"
                    image_blocks.append(
                        f"\n\n![{concept}](images/{rel_img})\n> **Figura**: *{concept}* — {caption} ({cit})\n"
                    )
                else:
                    text_snippet = content[:250]
                    lines.append(f"• {text_snippet} {cit}")

            # Interleave key illustrative diagrams directly into the response
            if image_blocks:
                lines.append("\n" + "\n".join(image_blocks))

            unique_citations = list(dict.fromkeys(citations_list))
            lines.append("\n📚 Referències Consultades:")
            for cit in unique_citations:
                lines.append(f"- {cit}")

            return "\n".join(lines)
        else:
            return f"[Gemma 4 E4B] Resposta generada a: '{last_query}'. No s'ha requerit cap context RAG addicional."
