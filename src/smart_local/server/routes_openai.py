import json
import time
import uuid
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from smart_local.config import settings
from smart_local.knowledge.schema import RetrievalResult

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = settings.generator_model
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    rag_id: Optional[str] = None
    top_k: Optional[int] = 6
    min_score: Optional[float] = None
    modality: Optional[str] = None

class EmbeddingRequest(BaseModel):
    model: str = settings.embedding_model
    input: Union[str, List[str]]
    user: Optional[str] = None

@router.get("/models")
async def list_models(request: Request):
    """OpenAI-compatible models list endpoint."""
    km = request.app.state.knowledge_manager
    models = [
        {
            "id": settings.generator_model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "smart-local",
            "description": "Gemma 4 E4B multimodal generative model"
        },
        {
            "id": settings.embedding_model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "smart-local",
            "description": "EmbeddingGemma 2 multimodal (768-d) embedding model"
        }
    ]

    # Add dynamic model aliases for all discovered RAGs
    for r in km.list_rags():
        rag_id = r["rag_id"]
        models.append({
            "id": f"{settings.generator_model}:{rag_id}",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "smart-local",
            "description": f"Gemma 4 E4B with specialized RAG: {r['name']} ({'loaded' if r['is_loaded'] else 'on-demand'})"
        })

    return {"object": "list", "data": models}

@router.post("/embeddings")
async def create_embeddings(req: EmbeddingRequest, request: Request):
    """OpenAI-compatible embeddings endpoint powered by EmbeddingGemma 2."""
    embedder = request.app.state.embedder
    inputs = [req.input] if isinstance(req.input, str) else req.input

    data = []
    total_tokens = 0
    for idx, text in enumerate(inputs):
        vec = embedder.embed_text(text)
        total_tokens += len(text.split())
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": vec
        })

    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }

@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible chat completion endpoint.
    Retrieves grounded context from specified RAG on-demand,
    then executes Gemma 4 E4B to formulate response.
    """
    km = request.app.state.knowledge_manager
    generator = request.app.state.generator
    retriever = request.app.state.retriever

    # Resolve target RAG if specified in body or model name (e.g. gemma-4-e4b:finance)
    target_rag_id = req.rag_id
    if not target_rag_id and ":" in req.model:
        parts = req.model.split(":", 1)
        target_rag_id = parts[1]

    context_chunks: List[Dict[str, Any]] = []

    # Extract last user query
    query_text = ""
    for m in reversed(req.messages):
        if m.role == "user":
            if isinstance(m.content, str):
                query_text = m.content
            elif isinstance(m.content, list):
                for item in m.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        query_text += " " + item.get("text", "")
            break

    # If RAG is requested: either a specific RAG or across ALL available RAGs (Kev Auto-Triage)
    if target_rag_id in ["all", "auto", "*"] or not target_rag_id:
        all_rags = km.list_rags()
        all_results = []
        for r_item in all_rags:
            r_id = r_item.get("rag_id")
            if not r_id:
                continue
            store = km.get_store(r_id, auto_load=True)
            if store and query_text:
                res = retriever.retrieve(
                    store=store,
                    query=query_text.strip(),
                    top_k=req.top_k or 4,
                    min_score=req.min_score,
                    modality=req.modality
                )
                all_results.extend(res)
        all_results.sort(key=lambda x: x.score, reverse=True)
        context_chunks = [r.model_dump() for r in all_results[:req.top_k or 6]]
    elif target_rag_id:
        store = km.get_store(target_rag_id, auto_load=True)
        if not store:
            raise HTTPException(status_code=404, detail=f"RAG '{target_rag_id}' not found.")

        if query_text:
            results = retriever.retrieve(
                store=store,
                query=query_text.strip(),
                top_k=req.top_k or 6,
                min_score=req.min_score,
                modality=req.modality
            )
            context_chunks = [r.model_dump() for r in results]

    messages_payload = [m.model_dump() for m in req.messages]

    effective_max_tokens = max(req.max_tokens or 1500, 1024)

    # Handle streaming vs non-streaming
    if req.stream:
        async def stream_generator():
            request_id = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())

            async for token in generator.stream_response(
                messages=messages_payload,
                context_chunks=context_chunks,
                temperature=req.temperature or 0.7,
                max_tokens=effective_max_tokens
            ):
                chunk_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Final finish reason event
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ],
                "rag_citations": [
                    {"rag_id": c.get("rag_id"), "source": c.get("source_path"), "score": c.get("score")}
                    for c in context_chunks
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming
    resp = await generator.generate_response(
        messages=messages_payload,
        context_chunks=context_chunks,
        temperature=req.temperature or 0.7,
        max_tokens=effective_max_tokens
    )

    request_id = f"chatcmpl-{uuid.uuid4()}"
    completion_data = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp["content"]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(str(m.content).split()) for m in req.messages),
            "completion_tokens": len(resp["content"].split()),
            "total_tokens": sum(len(str(m.content).split()) for m in req.messages) + len(resp["content"].split())
        },
        "citations": resp.get("citations", [])
    }
    return completion_data
