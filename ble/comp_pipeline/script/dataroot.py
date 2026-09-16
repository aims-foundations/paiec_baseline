"""
dataroot.py — Where the databank parquet files live at competition runtime.

The competition sandbox has no general internet access, so the public-training
corpus tables and item embeddings must be available locally. Resolution order
for the data root:

  1. env var BLF_DATA_ROOT
  2. "data_root" in submission_config.json (absolute, or relative to zip root)
  3. <zip root>/data              (payload bundled inside the zip)
  4. HF cache snapshot of the repo named by "hf_data_repo" in
     submission_config.json (pre-downloaded by the platform via models.txt);
     the payload's data/ subdir inside that snapshot.

The payload layout (built by tools/prepare_data.py):
    data/<benchmark_dir>/{benchmarks,subjects,items,response[,traces]}.parquet
    data/corpus_manifest.json      {"corpus_scope": ..., "eligible_dirs": ...}
    embeddings/manifest.json, embeddings/items/<benchmark_dir>.parquet
"""

import json
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_ZIP_ROOT = os.path.dirname(os.path.dirname(_HERE))
_PUBLIC_DATA_REPO = "aims-foundations/measurement-db"
_CORPUS_SCOPE = "paec-public-training"

_lock = threading.Lock()
_resolved = {}  # "data" -> path or None, "emb" -> path or None


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _recorded_revision(value):
    """Check build provenance; no particular historical commit is required."""
    if not isinstance(value, str) or len(value) != 40:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _valid_data_root(path):
    manifest = _read_json(os.path.join(path, "corpus_manifest.json"))
    return bool(
        isinstance(manifest, dict)
        and manifest.get("corpus_scope") == _CORPUS_SCOPE
        and manifest.get("db_repo") == _PUBLIC_DATA_REPO
        and _recorded_revision(manifest.get("revision"))
    )


def _valid_emb_root(path):
    manifest = _read_json(os.path.join(path, "manifest.json"))
    return bool(
        isinstance(manifest, dict)
        and manifest.get("corpus_scope") == _CORPUS_SCOPE
        and _recorded_revision(manifest.get("data_revision"))
    )


def _config():
    path = os.path.join(_ZIP_ROOT, "submission_config.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _hf_snapshot_dir(repo_id):
    """Local snapshot dir of a pre-downloaded HF repo (offline-safe)."""
    if not repo_id:
        return None
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    for repo_type in ("model", "dataset"):
        try:
            return snapshot_download(
                repo_id, repo_type=repo_type, local_files_only=True
            )
        except LocalEntryNotFoundError:
            pass  # The snapshot may have been prefetched as the other repo type.
    return None


def data_root():
    """Directory holding <benchmark_dir>/*.parquet, or None if unavailable."""
    with _lock:
        if "data" in _resolved:
            return _resolved["data"]
        root = None
        cfg = _config()
        cands = []
        if os.environ.get("BLF_DATA_ROOT"):
            cands.append(os.environ["BLF_DATA_ROOT"])
        if cfg.get("data_root"):
            p = cfg["data_root"]
            cands.append(p if os.path.isabs(p) else os.path.join(_ZIP_ROOT, p))
        cands.append(os.path.join(_ZIP_ROOT, "data"))
        snap = _hf_snapshot_dir(cfg.get("hf_data_repo", ""))
        if snap:
            cands.append(os.path.join(snap, "data"))
            cands.append(snap)
        for c in cands:
            if c and _valid_data_root(c):
                root = c
                break
        _resolved["data"] = root
        if root:
            print(f"[dataroot] databank root: {root}", file=sys.stderr)
        else:
            print("[dataroot] no local databank payload found", file=sys.stderr)
        return root


def emb_root():
    """Directory holding embeddings/manifest.json + items/, or None."""
    with _lock:
        if "emb" in _resolved:
            return _resolved["emb"]
        cfg = _config()
        cands = []
        if os.environ.get("BLF_EMB_ROOT"):
            cands.append(os.environ["BLF_EMB_ROOT"])
        cands.append(os.path.join(_ZIP_ROOT, "embeddings"))
        snap = _hf_snapshot_dir(cfg.get("hf_data_repo", ""))
        if snap:
            cands.append(os.path.join(snap, "embeddings"))
        root = None
        for c in cands:
            if c and _valid_emb_root(c):
                root = c
                break
        _resolved["emb"] = root
        return root


def _load(bench_dir, table):
    """Read one parquet table from the resolved local payload."""
    import pandas as pd

    root = data_root()
    if not root:
        raise FileNotFoundError(
            "No local databank payload; bundle data/ or pre-download "
            "the configured hf_data_repo"
        )
    path = os.path.join(root, bench_dir, f"{table}.parquet")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def manifest():
    root = data_root()
    if root:
        return _read_json(os.path.join(root, "corpus_manifest.json"))
    return None


def repo_files():
    """Set of '<dir>/<table>.parquet' paths declared by the local manifest."""
    man = manifest()
    files = man.get("files") if isinstance(man, dict) else None
    if not isinstance(files, list) or not files:
        raise FileNotFoundError(
            "Local corpus_manifest.json has no non-empty files list"
        )
    return set(files)


def corpus_dirs():
    """Eligible benchmark dirs declared by the local manifest."""
    man = manifest()
    dirs = man.get("eligible_dirs") if isinstance(man, dict) else None
    if not isinstance(dirs, list) or not dirs:
        raise FileNotFoundError(
            "Local corpus_manifest.json has no non-empty eligible_dirs list"
        )
    return sorted(dirs)
