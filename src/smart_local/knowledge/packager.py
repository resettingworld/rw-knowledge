import os
import shutil
import tarfile
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

from smart_local.config import settings
from smart_local.knowledge.schema import KnowledgePackManifest

class KnowledgePackager:
    """
    Packages a RAG knowledge base into a portable, distributable bundle (.gpk archive),
    and imports existing packages for on-demand loading.
    """
    def __init__(self, rags_dir: Path = settings.rags_dir, packs_dir: Path = settings.packs_dir):
        self.rags_dir = Path(rags_dir)
        self.packs_dir = Path(packs_dir)
        self.rags_dir.mkdir(parents=True, exist_ok=True)
        self.packs_dir.mkdir(parents=True, exist_ok=True)

    def package_rag(
        self,
        rag_id: str,
        name: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
        tags: Optional[list] = None
    ) -> Path:
        """
        Creates a distributable .gpk (tar.gz) package containing:
        - manifest.json
        - index.db
        - assets/
        """
        rag_dir = self.rags_dir / rag_id
        if not rag_dir.exists():
            raise FileNotFoundError(f"RAG '{rag_id}' does not exist at {rag_dir}")

        manifest_file = rag_dir / "manifest.json"
        manifest_data = {}
        if manifest_file.exists():
            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Update manifest
        manifest = KnowledgePackManifest(
            rag_id=rag_id,
            name=name or manifest_data.get("name", rag_id),
            description=description or manifest_data.get("description", ""),
            version=version or manifest_data.get("version", "1.0.0"),
            tags=tags or manifest_data.get("tags", []),
            created_at=time.time(),
            embedding_model=settings.embedding_model,
            generator_model=settings.generator_model
        )

        manifest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        package_filename = f"{rag_id}-v{manifest.version}.gpk"
        package_path = self.packs_dir / package_filename

        # Create tar.gz bundle
        with tarfile.open(package_path, "w:gz") as tar:
            for item in rag_dir.iterdir():
                tar.add(item, arcname=item.name)

        return package_path

    def import_package(self, package_path: Path, rag_id_override: Optional[str] = None) -> str:
        """
        Imports and extracts a .gpk (tar.gz) or standard .zip knowledge bundle into the active RAG directory.
        Returns the resolved rag_id.
        """
        package_path = Path(package_path)
        if not package_path.exists():
            raise FileNotFoundError(f"Package file not found: {package_path}")

        if package_path.suffix.lower() == ".zip":
            from smart_local.pipeline.capsule_packager import KnowledgeCapsulePackager
            packager = KnowledgeCapsulePackager(packs_dir=self.packs_dir)
            return packager.import_zip_package(
                zip_path=package_path,
                rags_dir=self.rags_dir,
                rag_id_override=rag_id_override
            )

        # Inspect manifest from tar.gz archive
        temp_dir = self.packs_dir / f"_tmp_{int(time.time() * 1000)}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(package_path, "r:gz") as tar:
                try:
                    tar.extractall(temp_dir, filter="data")
                except TypeError:
                    tar.extractall(temp_dir)

            manifest_path = temp_dir / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("Corrupt package: manifest.json missing")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_rag_id = rag_id_override or manifest.get("rag_id")
            if not target_rag_id:
                target_rag_id = package_path.stem.split("-v")[0]

            destination = self.rags_dir / target_rag_id
            if destination.exists():
                shutil.rmtree(destination)

            shutil.copytree(temp_dir, destination)
            return target_rag_id
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def list_packages(self) -> Dict[str, Any]:
        """Lists all packaged bundles (.gpk and .zip) in the packs directory."""
        packages = []
        found_paths = list(self.packs_dir.glob("*.gpk")) + list(self.packs_dir.glob("*.zip"))
        for p in found_paths:
            packages.append({
                "filename": p.name,
                "path": str(p),
                "format": p.suffix.lstrip("."),
                "size_bytes": p.stat().st_size,
                "modified_at": p.stat().st_mtime
            })
        return {"packages": packages}
