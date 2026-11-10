import pytest
import httpx
from openai import AsyncOpenAI
from smart_local.server.app import app

@pytest.mark.asyncio
async def test_official_openai_sdk_compatibility():
    """
    Validates that the official OpenAI Python SDK can communicate directly
    with our local Gemma 4 E4B + EmbeddingGemma 2 server.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://local.test/v1") as http_client:
        client = AsyncOpenAI(
            base_url="http://local.test/v1",
            api_key="sk-local-test",
            http_client=http_client
        )

        # 1. Models list
        models = await client.models.list()
        ids = [m.id for m in models.data]
        assert "gemma-4-e4b" in ids
        assert "embeddinggemma-2" in ids

        # 2. Embeddings
        emb = await client.embeddings.create(
            model="embeddinggemma-2",
            input="Proves d'embedding multimodal"
        )
        assert len(emb.data) == 1
        assert len(emb.data[0].embedding) == 768

        # 3. Chat completion
        completion = await client.chat.completions.create(
            model="gemma-4-e4b",
            messages=[{"role": "user", "content": "Hola Gemma 4!"}],
            max_tokens=20
        )
        assert completion.choices[0].message.content is not None
        assert len(completion.choices[0].message.content) > 0
