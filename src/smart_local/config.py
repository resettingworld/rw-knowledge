from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # App and paths
    app_name: str = "SmartLocal RAG Engine"
    version: str = "1.0.0"
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data")
    rags_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "rags")
    packs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "packs")
    uploads_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "uploads")

    # Models
    generator_model: str = "gemma-4-e4b"
    generator_ollama_fallback: str = "gemma4:latest"
    embedding_model: str = "embeddinggemma-2"
    embedding_dim: int = 768

    # Ollama integration
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = 120.0

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8800
    api_key: str = "sk-smartlocal-local"

    # Dynamic loading & cache settings
    max_cached_rags: int = 10
    auto_load_on_query: bool = True

    model_config = {
        "env_prefix": "SMART_LOCAL_",
        "extra": "ignore"
    }

    def ensure_directories(self) -> None:
        self.rags_dir.mkdir(parents=True, exist_ok=True)
        self.packs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
