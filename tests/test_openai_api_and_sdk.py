import pytest
from starlette.testclient import TestClient
from smart_local.server.app import app
from smart_local.sdk.client import SmartLocalSDK
from openai import OpenAI

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_models_endpoint(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    model_ids = [m["id"] for m in data["data"]]
    assert "gemma-4-e4b" in model_ids
    assert "embeddinggemma-2" in model_ids

def test_embeddings_endpoint(client):
    res = client.post("/v1/embeddings", json={
        "model": "embeddinggemma-2",
        "input": "Coneixement multimodal per RAG"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 768

def test_rag_lifecycle_and_chat_completion(client):
    rag_id = "test_openai_rag"

    # 1. Create RAG
    res = client.post("/v1/rags", json={
        "rag_id": rag_id,
        "name": "Manual Oficial Gemma",
        "description": "Especificacions tècniques de Gemma 4 i EmbeddingGemma"
    })
    assert res.status_code == 200

    # 2. Ingest text into RAG
    res = client.post(f"/v1/rags/{rag_id}/ingest/text", json={
        "title": "manual_gemma",
        "content": "Gemma 4 E4B compta amb aproximadament 4.5 bilions de paràmetres efectius i suporta àudio natiu."
    })
    assert res.status_code == 200

    # 3. Dynamic On-demand Load & Unload
    res = client.post(f"/v1/rags/{rag_id}/load")
    assert res.status_code == 200
    assert res.json()["status"] == "loaded"

    res = client.post(f"/v1/rags/{rag_id}/unload")
    assert res.status_code == 200
    assert res.json()["status"] == "unloaded"

    # 4. Chat Completion querying RAG (auto-loads on-demand!)
    res = client.post("/v1/chat/completions", json={
        "model": f"gemma-4-e4b:{rag_id}",
        "messages": [
            {"role": "user", "content": "Quants paràmetres té Gemma 4 E4B?"}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    })
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0
    # Check that citations were returned
    assert len(data.get("citations", [])) > 0
    assert data["citations"][0]["rag_id"] == rag_id

def test_chat_streaming(client):
    rag_id = "test_stream_rag"
    client.post("/v1/rags", json={"rag_id": rag_id, "name": "Streaming RAG"})
    client.post(f"/v1/rags/{rag_id}/ingest/text", json={
        "title": "stream_doc",
        "content": "EmbeddingGemma 2 uneix text, àudio i visió en 768 dimensions."
    })

    with client.stream("POST", "/v1/chat/completions", json={
        "model": "gemma-4-e4b",
        "rag_id": rag_id,
        "messages": [{"role": "user", "content": "Descriu EmbeddingGemma 2"}],
        "stream": True,
        "max_tokens": 20
    }) as res:
        assert res.status_code == 200
        events = []
        for line in res.iter_lines():
            if line.startswith("data: "):
                events.append(line)
        assert len(events) > 1
        assert any("[DONE]" in e for e in events)

def test_packaging_endpoints(client):
    rag_id = "test_pkg_api"
    client.post("/v1/rags", json={"rag_id": rag_id, "name": "Package Test"})
    client.post(f"/v1/rags/{rag_id}/ingest/text", json={
        "title": "doc1",
        "content": "Contingut a empaquetar per distribució."
    })

    # Trigger package
    res = client.post(f"/v1/rags/{rag_id}/package", json={
        "name": "Paquet Distribució",
        "version": "1.0.0"
    })
    assert res.status_code == 200
    task_id = res.json()["task_id"]

    # Poll task
    import time
    for _ in range(30):
        t_res = client.get(f"/v1/rags/{rag_id}/tasks/{task_id}")
        if t_res.json()["status"] == "completed":
            break
        time.sleep(0.1)

    # Download package
    dl_res = client.get(f"/v1/rags/{rag_id}/download")
    assert dl_res.status_code == 200
    assert len(dl_res.content) > 0
