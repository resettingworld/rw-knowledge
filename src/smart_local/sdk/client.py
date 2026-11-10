from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Generator
import httpx
from openai import OpenAI

class RAGAdminClient:
    """Admin and knowledge management client for SmartLocal RAGs."""
    def __init__(self, base_url: str, api_key: str = "sk-smartlocal-local"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def list(self) -> List[Dict[str, Any]]:
        """Lists all RAG spaces and their loaded status."""
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=30.0) as client:
            res = client.get("/v1/rags")
            res.raise_for_status()
            return res.json().get("rags", [])

    def create(self, rag_id: str, name: Optional[str] = None, description: str = "", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Creates a new specific RAG knowledge base."""
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=30.0) as client:
            payload = {"rag_id": rag_id, "name": name, "description": description, "tags": tags or []}
            res = client.post("/v1/rags", json=payload)
            res.raise_for_status()
            return res.json()

    def get(self, rag_id: str) -> Dict[str, Any]:
        """Gets info and statistics about a RAG space."""
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=30.0) as client:
            res = client.get(f"/v1/rags/{rag_id}")
            res.raise_for_status()
            return res.json()

    def load(self, rag_id: str) -> Dict[str, Any]:
        """Dynamically loads embeddings and index into memory on demand."""
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=30.0) as client:
            res = client.post(f"/v1/rags/{rag_id}/load")
            res.raise_for_status()
            return res.json()

    def unload(self, rag_id: str) -> Dict[str, Any]:
        """Dynamically unloads embeddings from memory to save RAM."""
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=30.0) as client:
            res = client.post(f"/v1/rags/{rag_id}/unload")
            res.raise_for_status()
            return res.json()

    def delete(self, rag_id: str) -> Dict[str, Any]:
        """Deletes a RAG knowledge base."""
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=30.0) as client:
            res = client.delete(f"/v1/rags/{rag_id}")
            res.raise_for_status()
            return res.json()

    def ingest_text(self, rag_id: str, title: str, content: str) -> Dict[str, Any]:
        """Ingests raw text into a RAG."""
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=60.0) as client:
            payload = {"title": title, "content": content}
            res = client.post(f"/v1/rags/{rag_id}/ingest/text", json=payload)
            res.raise_for_status()
            return res.json()

    def ingest_files(self, rag_id: str, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """Enqueues multimodal files (docs, images, audio) for background ingestion."""
        files = []
        opened = []
        try:
            for p in file_paths:
                path_obj = Path(p)
                f = open(path_obj, "rb")
                opened.append(f)
                files.append(("files", (path_obj.name, f)))

            with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=60.0) as client:
                res = client.post(f"/v1/rags/{rag_id}/ingest/files", files=files)
                res.raise_for_status()
                return res.json()
        finally:
            for f in opened:
                f.close()

    def package(self, rag_id: str, name: Optional[str] = None, version: str = "1.0.0", description: str = "", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enqueues a task to package the RAG into a portable .gpk distributable bundle."""
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=30.0) as client:
            payload = {"name": name, "version": version, "description": description, "tags": tags or []}
            res = client.post(f"/v1/rags/{rag_id}/package", json=payload)
            res.raise_for_status()
            return res.json()

    def download_package(self, rag_id: str, destination_dir: Union[str, Path] = ".") -> Path:
        """Downloads the packaged .gpk bundle for distribution."""
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=120.0) as client:
            with client.stream("GET", f"/v1/rags/{rag_id}/download") as response:
                response.raise_for_status()
                # Parse filename from header or fallback
                cd = response.headers.get("content-disposition", "")
                filename = f"{rag_id}.gpk"
                if "filename=" in cd:
                    filename = cd.split("filename=")[1].strip('"')
                out_path = destination_dir / filename
                with open(out_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                return out_path

    def import_package(self, package_path: Union[str, Path], rag_id_override: Optional[str] = None) -> Dict[str, Any]:
        """Imports an exported .gpk bundle into this server's RAG registry."""
        pkg_path = Path(package_path)
        with open(pkg_path, "rb") as f:
            files = {"package": (pkg_path.name, f, "application/gzip")}
            data = {"rag_id_override": rag_id_override} if rag_id_override else {}
            with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=120.0) as client:
                res = client.post("/v1/rags/import", files=files, data=data)
                res.raise_for_status()
                return res.json()

    def get_task(self, rag_id: str, task_id: str) -> Dict[str, Any]:
        """Queries the status and progress of a background worker task."""
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=30.0) as client:
            res = client.get(f"/v1/rags/{rag_id}/tasks/{task_id}")
            res.raise_for_status()
            return res.json()


class SmartLocalSDK:
    """
    Unified client providing full standard OpenAI SDK compatibility alongside
    specialized RAG lifecycle and multimodal management methods.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "sk-smartlocal-local"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        # Native OpenAI client instance configured for local endpoint
        self.openai = OpenAI(base_url=self.base_url, api_key=api_key)

        # Proxied shortcuts for standard OpenAI endpoints
        self.chat = self.openai.chat
        self.embeddings = self.openai.embeddings
        self.models = self.openai.models

        # Extended RAG management capabilities
        self.rags = RAGAdminClient(base_url=self.base_url.replace("/v1", ""), api_key=api_key)

    def query_rag(
        self,
        rag_id: str,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.7,
        top_k: int = 4
    ):
        """
        Convenience method to query Gemma 4 E4B with a specific RAG knowledge base.
        """
        extra_body = {"rag_id": rag_id, "top_k": top_k}
        return self.chat.completions.create(
            model=f"gemma-4-e4b:{rag_id}",
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            temperature=temperature,
            extra_body=extra_body
        )
