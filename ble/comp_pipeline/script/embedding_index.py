"""
embedding_index.py — In-memory semantic index over the pre-embedded databank items.

The query is embedded with the model and dimensions declared by the supplied
embeddings/manifest.json, then compared to precomputed public item vectors.
Both item and query vectors must have unit length so their dot product equals
cosine similarity. Items with identical content share one search result.

Payload schema (prepare before evaluation; see the repository README):
  manifest.json: model, dimensions, benchmarks (mapping of directory names),
                 corpus_scope, data_revision
  items/<benchmark_dir>.parquet: item_id (string), content_sha1 (string),
                                embedding (list of floats)

Vectors are loaded once and retained as float32 matrices. Memory for the
matrices is approximately item_count * dimensions * 4 bytes, plus metadata.
Query embeddings need OPENAI_API_KEY; reading the local index does not.
"""

import json
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .openai_api import post_json


def _default_emb_root():
    """Competition adaptation: embeddings root resolved by dataroot.emb_root()
    (zip payload or pre-downloaded HF repo); None if not packaged."""
    from .dataroot import emb_root

    return emb_root()


_QUERY_CACHE_SIZE = 512  # embedded queries kept (LRU); repeats are common
_LOAD_WORKERS = 16


class EmbeddingIndex:
    def __init__(self, root=None):
        self.root = root or _default_emb_root() or ""
        self.items_dir = os.path.join(self.root, "items")
        self.manifest_path = os.path.join(self.root, "manifest.json")
        self._manifest = None
        self._mats = {}  # dir -> (matrix float32 [n, dim], item_ids ndarray)
        self._lock = threading.Lock()
        self._qcache = OrderedDict()  # query text -> unit vector
        self._qlock = threading.Lock()

    # -- manifest ------------------------------------------------------------

    def manifest(self):
        """Read the fixed evaluation payload's manifest once."""
        if not os.path.isfile(self.manifest_path):
            return None
        if self._manifest is None:
            with open(self.manifest_path) as f:
                self._manifest = json.load(f)
        return self._manifest

    def embedded_dirs(self):
        """Benchmark dirs whose embeddings are complete on disk."""
        man = self.manifest()
        if man is None:
            return set()
        return {
            d
            for d in man.get("benchmarks", {})
            if os.path.exists(os.path.join(self.items_dir, f"{d}.parquet"))
        }

    # -- vector loading ------------------------------------------------------

    def _load_dir(self, d):
        df = pd.read_parquet(
            os.path.join(self.items_dir, f"{d}.parquet"),
            columns=["item_id", "content_sha1", "embedding"],
        )
        # Within-benchmark duplicate contents share one vector; keep one row
        # each so a query can't fill its top-k with copies of the same text.
        df = df.drop_duplicates("content_sha1", keep="first")
        mat = np.ascontiguousarray(
            np.stack(df["embedding"].to_numpy()), dtype=np.float32
        )
        return mat, df["item_id"].to_numpy()

    def ensure_loaded(self, dirs):
        """Load vectors for the given dirs (those embedded so far); returns the
        subset actually available. Thread-safe; each dir is loaded once."""
        avail = self.embedded_dirs()
        want = [d for d in dirs if d in avail]
        with self._lock:
            missing = [d for d in want if d not in self._mats]
            if missing:
                t0 = time.time()
                with ThreadPoolExecutor(max_workers=_LOAD_WORKERS) as ex:
                    for d, res in zip(missing, ex.map(self._load_dir, missing)):
                        self._mats[d] = res
                n_vec = sum(self._mats[d][0].shape[0] for d in missing)
                print(
                    f"[embedding_index] loaded {len(missing)} benchmarks "
                    f"({n_vec:,} vectors) in {time.time() - t0:.0f}s",
                    file=sys.stderr,
                )
        return want

    # -- query embedding -----------------------------------------------------

    def embed_query(self, text):
        text = str(text).strip() or " "
        with self._qlock:
            if text in self._qcache:
                self._qcache.move_to_end(text)
                return self._qcache[text]
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set; cannot embed the query")
        man = self.manifest()
        kwargs = {
            "model": man["model"],
            "input": [text],
            "dimensions": man["dimensions"],
            "encoding_format": "float",
        }
        resp = post_json("embeddings", kwargs, timeout=60)
        vec = np.asarray(resp["data"][0]["embedding"], dtype=np.float32)
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
        with self._qlock:
            self._qcache[text] = vec
            if len(self._qcache) > _QUERY_CACHE_SIZE:
                self._qcache.popitem(last=False)
        return vec

    # -- search --------------------------------------------------------------

    def search(self, query, dirs, k=8):
        """Top-k items across `dirs` by cosine similarity to `query`.

        Returns (hits, n_searched, n_skipped): hits is a list of
        (benchmark_dir, item_id, similarity) sorted by similarity desc;
        n_skipped counts requested dirs with no embeddings yet.
        """
        loaded = self.ensure_loaded(dirs)
        if not loaded:
            return [], 0, len(dirs)
        q = self.embed_query(query)
        cands = []
        for d in loaded:
            mat, ids = self._mats[d]
            scores = mat @ q
            top = (
                np.argpartition(scores, -min(k, len(scores)))[-k:]
                if len(scores) > k
                else np.arange(len(scores))
            )
            for i in top:
                cands.append((d, str(ids[i]), float(scores[i])))
        cands.sort(key=lambda h: -h[2])
        return cands[:k], len(loaded), len(dirs) - len(loaded)


_INDEX = None
_INDEX_LOCK = threading.Lock()


def get_index() -> EmbeddingIndex | None:
    """Share the public embedding index across predictions; None if unavailable."""
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            idx = EmbeddingIndex()
            if idx.manifest() is None:
                return None
            _INDEX = idx
    return _INDEX
