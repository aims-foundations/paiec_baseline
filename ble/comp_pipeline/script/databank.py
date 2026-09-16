"""
databank.py — Closed-world access layer over the measurement-db databank.

All retrieval reads local parquet files through dataroot.py. `get_trace` reads
local traces when the public training release provides them.

The searchable benchmark directories come directly from the public payload's
manifest. All predictions share this fixed corpus and its loaded public tables;
the database holds no target subject, revealed labels, or prediction state.

subject_id and item_id inside these tables are public corpus join keys. They
are unrelated to hidden runtime identifiers and are not prediction inputs.
"""

import concurrent.futures as cf
import math
import os
import re
from collections import OrderedDict
from functools import lru_cache

from .dataroot import _load, data_root, repo_files

_MAX_ROWS = 40  # max table rows returned to the agent
_MAX_ITEM_CHARS = 6000  # cap on item content shown to the agent
_MAX_TRACE_CHARS = 4000  # cap on trace text shown to the agent
_RESPONSE_CACHE_SIZE = 16  # per-benchmark response tables kept in memory


# ---------------------------------------------------------------------------
# Minimal BM25 (no external dependency)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _TOKEN_RE.findall(str(text).lower())


class _BM25:
    """Rank keyword overlap, weighting uncommon terms and adjusting for length."""

    def __init__(self, docs, k1=1.5, b=0.75):
        """docs: list of strings."""
        self.k1, self.b = k1, b
        self.doc_toks = [_tokens(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_toks]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if docs else 1.0
        self.df = {}
        for toks in self.doc_toks:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(docs)

    def scores(self, query):
        q = _tokens(query)
        out = [0.0] * self.n
        for t in q:
            df = self.df.get(t)
            if not df:
                continue
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for i, toks in enumerate(self.doc_toks):
                tf = toks.count(t)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / self.avg_len
                )
                out[i] += idf * tf * (self.k1 + 1) / denom
        return out

    def top(self, query, k):
        s = self.scores(query)
        order = sorted(range(self.n), key=lambda i: -s[i])
        return [(i, s[i]) for i in order[:k] if s[i] > 0]


# ---------------------------------------------------------------------------
# MeasurementDB
# ---------------------------------------------------------------------------


class MeasurementDB:
    def __init__(self, corpus_benchmarks: list[str]):
        """Use the supplied public benchmark directories as the retrieval scope."""
        self.corpus = sorted(corpus_benchmarks)
        self._meta = None  # dir -> (bench_row, subjects_df)
        self._files = None  # set of repo file paths
        self._bench_index = None  # (_BM25, [dirs]) over metadata text
        self._item_df = {}  # dir -> items_df
        self._item_index = {}  # dir -> (_BM25, items_df)
        self._responses = OrderedDict()  # dir -> response df (LRU)
        self._difficulty = {}  # dir -> per-item mean response Series

    # -- metadata ------------------------------------------------------------

    def _load_meta(self):
        if self._meta is not None:
            return self._meta
        meta = {}

        def _one(d):
            return d, _load(d, "benchmarks").iloc[0], _load(d, "subjects")

        with cf.ThreadPoolExecutor(max_workers=16) as ex:
            for directory, benchmark, subjects in ex.map(_one, self.corpus):
                meta[directory] = (benchmark, subjects)
        self._meta = meta
        return meta

    def _repo_files(self):
        if self._files is None:
            self._files = repo_files()
        return self._files

    def _has(self, d, table):
        return f"{d}/{table}.parquet" in self._repo_files()

    def _resolve(self, benchmark_id):
        """Map a benchmark id/dir to its corpus dir, or an error string."""
        bid = str(benchmark_id).strip()
        meta = self._load_meta()
        if bid in meta:
            return bid, ""
        # match on benchmark_id field if it differs from the dir name
        for d, (bench, _) in meta.items():
            if bench["benchmark_id"] == bid:
                return d, ""
        return None, (
            f"Unknown benchmark {bid!r}. Use search_benchmarks to find "
            f"valid benchmark ids."
        )

    # -- benchmark discovery -------------------------------------------------

    def _card(self, d, brief=False):
        bench, _subjects = self._load_meta()[d]
        dom = ", ".join(bench["domain"]) if bench["domain"] is not None else ""
        mod = ", ".join(bench["modality"]) if bench["modality"] is not None else ""
        head = (
            f"[{d}] {bench['name']} (released {bench['release_date']}) — "
            f"domains: {dom}; modality: {mod}; items: {bench['n_items']}; "
            f"models evaluated: {bench['n_subjects']}"
        )
        if brief:
            desc = str(bench["description"] or "")[:300]
            return head + f"\n  {desc}"
        lines = [head, f"Description: {bench['description']}"]
        conds = bench.get("test_conditions")
        if conds is not None and len(conds):
            shown = list(conds)[:15]
            more = f" (+{len(conds) - 15} more)" if len(conds) > 15 else ""
            lines.append(
                f"Test conditions ({len(conds)}): "
                + "; ".join(str(c) for c in shown)
                + more
            )
        lines.append(
            f"Responses per item pair (trials): {bench['n_trials']}; "
            f"total responses: {bench['n_responses']}"
        )
        if bench.get("saturation") is not None:
            lines[-1] += (
                f"; mean score across all models (saturation): {bench['saturation']}"
            )
        lines.append(
            "Available data: "
            + ", ".join(t for t in ("items", "response", "traces") if self._has(d, t))
            + (
                "; item ground-truth answers included"
                if bench["has_ground_truth"]
                else ""
            )
        )
        return "\n".join(lines)

    def search_benchmarks(self, query, k=10):
        meta = self._load_meta()
        if self._bench_index is None:
            dirs = sorted(meta)
            docs = []
            for d in dirs:
                bench, _ = meta[d]
                dom = " ".join(bench["domain"]) if bench["domain"] is not None else ""
                mod = (
                    " ".join(bench["modality"]) if bench["modality"] is not None else ""
                )
                docs.append(f"{d} {bench['name']} {dom} {mod} {bench['description']}")
            self._bench_index = (_BM25(docs), dirs)
        idx, dirs = self._bench_index
        hits = idx.top(query, k)
        if not hits:
            return f"No benchmarks matched {query!r}. Try broader keywords."
        cards = [self._card(dirs[i], brief=True) for i, _ in hits]
        return (
            f"{len(hits)} benchmarks matched {query!r} "
            f"(searchable corpus: {len(dirs)} benchmarks):\n\n" + "\n".join(cards)
        )

    def get_benchmark(self, benchmark_id):
        d, err = self._resolve(benchmark_id)
        return err if err else self._card(d)

    # -- models --------------------------------------------------------------

    def find_model(self, model_name):
        """Locate public evaluation records by normalized model-name substring."""
        meta = self._load_meta()
        pat = str(model_name).strip().lower()
        by_model = {}
        for d, (bench, subjects) in meta.items():
            for name in subjects["normalized_name"].dropna().unique():
                if pat in str(name).lower():
                    by_model.setdefault(str(name), []).append(
                        (d, str(bench["release_date"] or ""))
                    )
        if not by_model:
            return f"No model matching {model_name!r} found in the databank."
        lines = []
        for name in sorted(by_model, key=lambda n: -len(by_model[n])):
            bms = sorted(by_model[name], key=lambda x: x[1], reverse=True)
            shown = ", ".join(f"{d} ({rd})" for d, rd in bms[:_MAX_ROWS])
            more = f" (+{len(bms) - _MAX_ROWS} more)" if len(bms) > _MAX_ROWS else ""
            lines.append(f"{name} — {len(bms)} benchmarks: {shown}{more}")
        return "\n".join(lines[:20])

    # -- scores --------------------------------------------------------------

    def _response(self, d):
        """Response table for one benchmark dir, LRU-cached."""
        if d in self._responses:
            self._responses.move_to_end(d)
            return self._responses[d]
        if not self._has(d, "response"):
            return None
        df = _load(d, "response")
        df["test_condition"] = (
            df["test_condition"].fillna("") if "test_condition" in df else ""
        )
        self._responses[d] = df
        if len(self._responses) > _RESPONSE_CACHE_SIZE:
            self._responses.popitem(last=False)
        return df

    def get_scores(self, benchmark_id, model="", test_condition=""):
        """Average public response scores within each subject and test condition.

        Each recorded trial contributes one row to the mean. n_items counts
        distinct items, while n_responses counts all trials in that group.
        """
        d, err = self._resolve(benchmark_id)
        if err:
            return err
        resp = self._response(d)
        if resp is None:
            return (
                f"Benchmark {d!r} has no per-item response data (items and "
                f"metadata only)."
            )
        _, subjects = self._load_meta()[d]
        df = resp
        if model:
            sids = [
                s
                for _, r in subjects.iterrows()
                if str(model).lower() in str(r["normalized_name"]).lower()
                for s in [r["subject_id"]]
            ]
            if not sids:
                return (
                    f"Model {model!r} not found in benchmark {d!r}. "
                    f"Models present: "
                    + ", ".join(subjects["normalized_name"].dropna().unique()[:30])
                )
            df = df[df["subject_id"].isin(sids)]
        if test_condition:
            df = df[df["test_condition"] == test_condition]
            if df.empty:
                conds = resp["test_condition"].unique()
                return (
                    f"No responses under condition {test_condition!r}. "
                    f"Conditions: " + "; ".join(map(str, conds[:20]))
                )
        agg = (
            df.groupby(["subject_id", "test_condition"])
            .agg(
                mean_score=("response", "mean"),
                n_items=("item_id", "nunique"),
                n_rows=("response", "size"),
            )
            .reset_index()
            .merge(
                subjects[
                    ["subject_id", "normalized_name", "display_name", "release_date"]
                ],
                on="subject_id",
                how="left",
            )
            .sort_values("mean_score", ascending=False)
        )
        lines = [
            f"Scores on [{d}] (mean response over items; 1.0 = all correct):",
            "model | variant | condition | mean_score | n_items | n_responses | model_release",
        ]
        for _, r in agg.head(_MAX_ROWS).iterrows():
            cond = r["test_condition"] or "-"
            variant = (
                r["display_name"] if r["display_name"] != r["normalized_name"] else "-"
            )
            lines.append(
                f"{r['normalized_name']} | {variant} | {cond} | "
                f"{r['mean_score']:.3f} | "
                f"{r['n_items']} | {r['n_rows']} | {r['release_date']}"
            )
        if len(agg) > _MAX_ROWS:
            lines.append(
                f"... ({len(agg) - _MAX_ROWS} more rows; filter by model "
                f"or test_condition to narrow)"
            )
        return "\n".join(lines)

    # -- items ---------------------------------------------------------------

    def _items_df(self, d):
        """Items table only (no BM25 index — cheap enough for content lookups)."""
        if d not in self._item_df:
            self._item_df[d] = _load(d, "items")
        return self._item_df[d]

    def _items(self, d):
        if d not in self._item_index:
            items = self._items_df(d)
            idx = _BM25(items["content"].fillna("").tolist())
            self._item_index[d] = (idx, items)
        return self._item_index[d]

    def _item_difficulty(self, d):
        """Per-item mean response across all subjects/trials (None if no data)."""
        if d not in self._difficulty:
            resp = self._response(d)
            self._difficulty[d] = (
                resp.groupby("item_id")["response"].mean() if resp is not None else None
            )
        return self._difficulty[d]

    def semantic_search_items(self, query, k=8, benchmark_id=""):
        """Embedding search over public corpus items (see embedding_index.py)."""
        from .embedding_index import get_index

        idx = get_index()
        if idx is None:
            raise FileNotFoundError(
                "semantic search requires a local embeddings payload"
            )
        dirs = self.corpus
        if benchmark_id:
            d, err = self._resolve(benchmark_id)
            if err:
                return err
            dirs = [d]
        k = min(max(int(k) if k else 8, 1), 20)
        hits, n_searched, n_skipped = idx.search(query, dirs, k=k)
        if not hits:
            return (
                f"No embedded items available to search yet "
                f"({n_skipped} benchmarks not embedded). Use the keyword "
                f"tools instead."
            )

        scope = (
            f"benchmark [{dirs[0]}]"
            if benchmark_id
            else f"{n_searched} corpus benchmarks"
            + (f" ({n_skipped} more not embedded yet)" if n_skipped else "")
        )
        out = [
            (
                f"Top {len(hits)} semantically similar items across {scope} "
                f"for: {str(query)[:200]!r}"
            )
        ]
        for d, item_id, sim in hits:
            head = f"[{d} / item {item_id}]  similarity {sim:.2f}"
            rows = self._items_df(d)
            matches = rows[rows["item_id"].astype(str) == item_id]
            if matches.empty:
                raise ValueError(
                    f"embedding item {d}/{item_id} is absent from the corpus"
                )
            row = matches.iloc[0]
            content = str(row["content"] or "")[:400].replace("\n", " ")
            # Embeddings store string ids; response tables retain native ids.
            difficulty = self._item_difficulty(d)
            if difficulty is not None and row["item_id"] in difficulty.index:
                head += f"  mean score across models: {difficulty[row['item_id']]:.2f}"
            out.append(f"{head}\n  {content}")
        out.append(
            "Use get_benchmark / get_scores on the benchmarks above to "
            "see how the target and its peers perform there, or get_item "
            "for an item in full."
        )
        return "\n\n".join(out)

    def search_items(self, benchmark_id, query, k=8):
        """Rank item text by BM25; report mean score across public subjects/trials."""
        d, err = self._resolve(benchmark_id)
        if err:
            return err
        idx, items = self._items(d)
        hits = idx.top(query, k)
        if not hits:
            return f"No items in [{d}] matched {query!r}."
        diff = self._item_difficulty(d)
        out = [f"{len(hits)} items in [{d}] matched {query!r}:"]
        for i, _ in hits:
            row = items.iloc[i]
            dline = ""
            if diff is not None and row["item_id"] in diff.index:
                dline = f"  mean score across models: {diff[row['item_id']]:.2f}"
            content = str(row["content"] or "")[:400].replace("\n", " ")
            out.append(f"[item {row['item_id']}]{dline}\n  {content}")
        return "\n\n".join(out)

    def get_item(self, benchmark_id, item_id="", sample=0, model=""):
        d, err = self._resolve(benchmark_id)
        if err:
            return err
        _, items = self._items(d)
        if item_id:
            rows = items[items["item_id"] == str(item_id)]
            if rows.empty:
                return f"Item {item_id!r} not found in [{d}]. Use search_items first."
        else:
            rows = items.sample(n=min(max(int(sample) or 1, 1), 5), random_state=0)
        out = []
        for _, row in rows.iterrows():
            content = str(row["content"] or "")
            if len(content) > _MAX_ITEM_CHARS:
                content = content[:_MAX_ITEM_CHARS] + " ...[truncated]"
            block = [f"[item {row['item_id']} of benchmark {d}]", content]
            if row.get("correct_answer"):
                block.append(f"Correct answer: {row['correct_answer']}")
            diff = self._item_difficulty(d)
            if diff is not None and row["item_id"] in diff.index:
                block.append(
                    f"Mean score across all models: {diff[row['item_id']]:.2f}"
                )
            if model:
                block.append(self._model_rows(d, model, row["item_id"]))
            out.append("\n".join(block))
        return "\n\n---\n\n".join(out)

    def _model_rows(self, d, model, item_id):
        resp = self._response(d)
        if resp is None:
            return f"(no response data in [{d}])"
        _, subjects = self._load_meta()[d]
        sids = [
            r["subject_id"]
            for _, r in subjects.iterrows()
            if str(model).lower() in str(r["normalized_name"]).lower()
        ]
        rows = resp[(resp["subject_id"].isin(sids)) & (resp["item_id"] == item_id)]
        if rows.empty:
            return f"({model!r} has no responses on this item)"
        lines = [f"Responses of {model!r} on this item:"]
        for _, r in rows.head(20).iterrows():
            cond = r["test_condition"] or "-"
            lines.append(
                f"  trial {r['trial']} | condition {cond} | score {r['response']}"
            )
        return "\n".join(lines)

    # -- traces --------------------------------------------------------------

    def get_trace(self, benchmark_id, model, item_id):
        d, err = self._resolve(benchmark_id)
        if err:
            return err
        root = data_root()
        path = os.path.join(root, d, "traces.parquet") if root else None
        if not path or not os.path.exists(path):
            return f"Benchmark [{d}] has no trace data."
        _, subjects = self._load_meta()[d]
        sids = [
            r["subject_id"]
            for _, r in subjects.iterrows()
            if str(model).lower() in str(r["normalized_name"]).lower()
        ]
        if not sids:
            return f"Model {model!r} not found in benchmark [{d}]."
        import pyarrow.parquet as pq

        tbl = pq.read_table(
            path, filters=[("item_id", "=", str(item_id)), ("subject_id", "in", sids)]
        )
        df = tbl.to_pandas()
        if df.empty:
            return f"No trace for {model!r} on item {item_id!r} in [{d}]."
        out = []
        for _, r in df.head(3).iterrows():
            trace = str(r["trace"] or "")
            if len(trace) > _MAX_TRACE_CHARS:
                trace = trace[:_MAX_TRACE_CHARS] + " ...[truncated]"
            cond = r.get("test_condition") or "-"
            out.append(f"[trace: trial {r['trial']}, condition {cond}]\n{trace}")
        return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Shared public data
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_db() -> MeasurementDB:
    """Load the fixed public corpus once per process for reuse across predictions.

    Only public tables and retrieval indexes are shared. Per-prediction beliefs
    and revealed observations stay in the agent's question and conversation.
    """
    from .dataroot import corpus_dirs

    return MeasurementDB(corpus_dirs())
