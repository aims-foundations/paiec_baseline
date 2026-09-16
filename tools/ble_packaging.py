"""Shared, explicit file selection for BLE and BLE-acquisition archives."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


def source_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or symlinked source: {relative}")
    if root.is_symlink() or any((root / p).is_symlink()
                               for p in path.relative_to(root).parents if str(p) != "."):
        raise ValueError(f"Symlinked source directory: {relative}")
    if root.resolve() not in path.resolve().parents:
        raise ValueError(f"Source outside its directory: {relative}")
    return path


def payload_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Unsafe path in public payload manifest")
    path = root / name
    if not path.is_file() or root.resolve() not in path.resolve().parents:
        raise ValueError("Missing or external file in public payload manifest")
    return path


def ble_files(root: Path, *, public: bool, mode: str, repo_id: str | None = None):
    """Return archive members from reviewed source and public payload manifests."""
    source = root / "ble"
    if any((source / name).exists() for name in ("subjects.txt", "subjects.txt.example")):
        raise ValueError("subjects.txt is unsupported; the platform selects subjects")
    names = ["model.py", "labeling.py", "requirements.txt"]
    names += [str(p.relative_to(source)) for d in ("src", "comp_pipeline")
              for p in (source / d).rglob("*.py") if "__pycache__" not in p.parts]
    files = {name: source_file(source, name) for name in names}
    config_name = "submission_config.example.json" if public else "submission_config.json"
    config = json.loads(source_file(source, config_name).read_text())
    if public and any(config.get("api_keys", {}).values()):
        raise ValueError("Public configuration must contain only blank API keys")
    if mode == "hf":
        if not repo_id or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
            raise ValueError("HF mode requires an owner/repository identifier")
        config["hf_data_repo"] = repo_id
        files["models.txt"] = (repo_id + "\n").encode()
    elif mode == "bundle":
        if repo_id is not None:
            raise ValueError("Bundle mode does not take a repository identifier")
        data = root / "payload/data"
        manifest_path = payload_file(data, "corpus_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if not (manifest.get("corpus_scope") == "paec-public-training"
                and manifest.get("db_repo") == "aims-foundations/measurement-db"
                and re.fullmatch(r"[0-9a-fA-F]{40}", str(manifest.get("revision", "")))):
            raise ValueError("Rebuild the payload from the official public training corpus")
        files["data/corpus_manifest.json"] = manifest_path
        for name in manifest.get("files", []):
            files["data/" + name] = payload_file(data, name)
        embeddings = root / "payload/embeddings"
        if embeddings.exists():
            embedding_manifest = payload_file(embeddings, "manifest.json")
            metadata = json.loads(embedding_manifest.read_text())
            if (metadata.get("corpus_scope") != "paec-public-training"
                    or metadata.get("data_revision") != manifest["revision"]):
                raise ValueError("Embedding provenance does not match the public payload")
            files["embeddings/manifest.json"] = embedding_manifest
            for name in metadata.get("benchmarks", {}):
                relative = f"items/{name}.parquet"
                files["embeddings/" + relative] = payload_file(embeddings, relative)
    else:
        raise ValueError("Choose bundle or hf packaging")
    files["submission_config.json"] = (json.dumps(config, indent=2) + "\n").encode()
    return files


def write_archive(files, output: Path) -> Path:
    """Replace output only after the complete archive has been written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as f:
        temporary = Path(f.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for name, value in sorted(files.items()):
                if isinstance(value, bytes):
                    archive.writestr(name, value)
                else:
                    archive.write(value, name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def arguments(description: str):
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--public", action="store_true", help="Use only blank example credentials")
    group.add_argument("--private", action="store_true", help="Include local BLE credentials")
    parser.add_argument("mode", choices=("bundle", "hf"))
    parser.add_argument("repo_id", nargs="?")
    return parser.parse_args()


def main():
    args = arguments("Build a BLE archive with explicit credential handling")
    root = Path(__file__).resolve().parents[1]
    name = "ble_public.zip" if args.public else "ble_submission.zip"
    files = ble_files(root, public=args.public, mode=args.mode, repo_id=args.repo_id)
    path = write_archive(files, root / "dist" / name)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
    print("Public archive: configure your own credentials before submitting."
          if args.public else "Private submission archive: do not publish this file.")


if __name__ == "__main__":
    main()
