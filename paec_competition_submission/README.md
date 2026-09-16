# PAEC baseline — BLF belief-state predictor

The [Predictive AI Evaluation Competition](https://aimslab.stanford.edu/competition)
calls `predict([subject, item], labeled)` once per target response. This baseline
asks an LLM to estimate P(correct), retrieve related public evidence, and revise
its estimate before submitting. It fits no parameters to the response tables.

This README supports local development and researcher walkthroughs. The
submission code contains the function-level explanation of the method.

A separate [mean predictor baseline](mean_submission/README.md) estimates each
subject–benchmark pair's success rate from acquired labels, with 0.5 at budget
0. It requires no API calls or training corpus and uses default random
acquisition. Its anonymous benchmark input field is prepared for the next
streaming infrastructure release; see its README for packaging and checks.

## Start with a walkthrough

Run these commands from this directory, in a Python 3.10+ environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r submission/requirements.txt
python tools/smoke_test.py --mock --walkthrough
```

Mock mode uses small synthetic tables and scripted LLM replies. It needs no
API keys or downloaded corpus, exercises the real retrieval and agent loop,
and shows when evidence reaches the next turn. Its example probabilities are
illustrative. It also checks invalid calls, API/tool failures, deadlines,
probability bounds, revealed-label handling, and streaming acquisition decisions without
revealed evidence.

## Read the method in this order

| File | What to look for |
| --- | --- |
| `submission/model.py` | `predict` contains question preparation, agent settings, the agent call, validation, and clipping. Read `build_question` next; configuration is loaded once above the functions, before the agent imports. |
| `submission/src/agent/agent.py` | `run_agent`: the complete estimate/retrieve/revise/submit loop and its run logs. |
| `submission/src/agent/prompts.py` | The instructions governing evidence selection and probability revision. |
| `submission/src/agent/belief_state.py` | The running estimate, cumulative evidence, and shared probability bounds. |
| `submission/src/agent/tools.py` | How an LLM-supplied belief is stored and a tool action is executed. |
| `submission/comp_pipeline/script/agent_tools.py` | The retrieval tools' arguments and intended uses. |
| `submission/comp_pipeline/script/databank.py` | Keyword ranking, score aggregation, item lookup, and trace retrieval. |
| `submission/labeling.py` | The optional uncertainty-sampling acquisition function; participants can replace it with their own strategy. |

Supporting code: `src/agent/llm_client.py` makes the prediction LLM requests;
`src/config/config.py` holds loop settings; `dataroot.py` locates local tables;
`embedding_index.py` performs optional semantic retrieval.
`comp_pipeline/script/openai_api.py` sends OpenAI requests using the Python
standard library, so Luna predictions and query embeddings need no inference SDK.

## One prediction, precisely

1. The input is `[subject, item]`, two dictionaries of strings. The subject
   fields are `normalized_name`, `provider`, `release_date`, `access_date`,
   `harness`, `harness_version`, `reasoning_effort`, and
   `subject_features_extra`. The item fields are `item_content`,
   `item_features`, and `interactors`; these and the subject attributes enter
   the target prompt. The next streaming release also supplies an anonymous
   `benchmark_id` for grouping evidence, used by the separate mean baseline.
2. Revealed observations are `[[subject, item], label]` entries, where label is
   0 or 1. Only entries matching the complete target subject dictionary are
   included. Each contributes its verdict and the first 300 characters of
   whitespace-normalized item text. Supporting examples omit item features and
   interactors to bound context size. Identical visible descriptions can refer
   to independent response targets, so no label directly supplies the answer.
3. The LLM forms an initial estimate, chooses one tool, and includes its
   current `updated_belief`. Python validates and stores that estimate.
   Luna requests require a tool call and use strict argument schemas. Optional
   fields sent as `null` are restored to their original omitted/default behavior.
   The tool's evidence is then presented to the LLM on its NEXT turn.
   The LLM calculates each revised probability through prompted reasoning;
   there is no separate Bayesian calculation in Python.
4. The LLM decides when to submit. The fixed limit is 10 turns by default,
   including submission, so at most 9 retrievals are possible. The final turn
   offers only `submit`. The confidence field is descriptive and does not
   trigger an automatic stopping rule.
5. A successful submission returns a finite Python float. Belief updates and
   final submission share `[0.02, 0.98]` bounds defined in `belief_state.py`.
   These bounds are a baseline choice; the competition permits `[0, 1]`.
   The submit action's `probability` is authoritative if it differs from
   its accompanying `updated_belief.p`.

Missing or malformed tool calls, API/tool execution errors, and expired
deadlines raise. An unfinished run never returns the current belief or the
initial 0.5. A successful search with no matches is an ordinary evidence result.

The retrieval layer reads benchmark directories from the public payload's
manifest and passes them directly to `MeasurementDB`. Every prediction uses
the same public database; its loaded tables and indexes are shared to avoid
repeated loading. Revealed labels and beliefs stay local to each prediction.
No per-subject corpus files or subject/item hashes are needed.

## Test-time adaptation: implement `labeling.py`

Participants implement `acquisition_function(input, prediction=None, labeled=None, context=None)`
in `labeling.py`. The platform computes the current candidate's prediction using
previously acquired labels, then passes that prediction to this hook. Return
`True` to acquire the current response, or `False` to skip it permanently.
The hook can adapt to prior evidence without another BLF call.

```python
def acquisition_function(input: list[dict], prediction: float | None = None,
                         labeled: list | None = None, context: dict | None = None) -> bool | float:
    if context is None:
        # When called by an older evaluator, return a representative-length
        # priority. The new evaluator supplies context and uses the policy below.
        content = str(input[1].get("item_content") or "")
        return float(-abs(len(content) - 1200))
    if prediction is None:
        raise ValueError("Streaming uncertainty acquisition requires prediction")
    if context["labels_remaining"] <= 0:
        return False
    # Reserve enough candidates to use the remaining budget. Earlier in the
    # stream, prefer uncertain predictions. No additional model call is needed.
    must_query = context["items_remaining"] <= context["labels_remaining"]
    return bool(must_query or abs(prediction - 0.5) <= 0.15)
```

Each eligible subject–benchmark pair has at least **80 distinct items**, split
once into persistent acquisition and evaluation pools in a 50/50 ratio. All runs
of an item stay in the same pool. One purchase reveals one recorded binary
response. Acquisition items are never scored; evaluation labels are never
revealed. The stream order and existing pool assignments are shared across
submissions.

One adaptive acquisition trajectory supplies nested budgets **0, 1, 3, 7, 15, 31**
per pair. `context` includes `subject_id`, `benchmark_id`, `labels_remaining`,
`labels_acquired`, `max_labels` (31), and `items_remaining` (including the current
candidate). `labeled` contains all examples revealed so far across sampled
pairs, using the existing predictor input format. Acquisition proceeds
sequentially; evaluation uses up to 16 concurrent calls at each checkpoint.
All groups finish a checkpoint before any group receives higher-budget labels.

The sample hook prefers probabilities within 0.15 of 0.5 and reserves enough
remaining candidates to fill the budget. Its benefit over random acquisition
needs empirical validation. Omitting `labeling.py` uses deterministic random
selection. A failed or invalid streaming hook fails the run. Policies may acquire
fewer labels; feedback reports the actual number at each cap.

Work is at most approximately `A + 6E` predictor calls, where `A` and `E` refer
to acquisition candidates and distinct evaluation inputs. Repeated evaluation
inputs reuse predictions within each checkpoint. The approximately 1,000-item
sample limit is before the 50/50 split, so it is not a 1,000-call limit.

Scoring averages the six Brier scores with weights
`[0.1, 0.2, 0.2, 0.2, 0.2, 0.1]`, then averages subject–benchmark pairs equally.
This normalized area on `log2(1 + budget)` remains on the Brier scale. Private
formative feedback reports the curve and calibration ECE with anonymous IDs.

### Compatibility with existing acquisition hooks

Existing `acquisition_function(input) -> float` submissions remain valid. The
runtime inspects the signature and calls them with **one argument only**; a
participant exception is never retried using another call shape. Finite numeric
priorities, including negative scores and zero, retain their ranking meaning.

For these legacy hooks, the platform scores every acquisition candidate once
with no acquired labels available, commits all priorities, and reveals the top
`n` candidates per subject–benchmark pair at budgets 0, 1, 3, 7, 15, 31. Ties use
the fixed random candidate order. The resulting label prefixes are nested.
This compatibility path uses pool ranking; it does not convert priorities into
streaming decisions or make another predictor call on behalf of the hook.
Evaluation items and their outcomes never enter this ranking pool.

New hooks may declare `prediction=None`, `labeled=None`, and `context=None`.
The runtime supplies supported named arguments (including keyword-only subsets),
or all four positional arguments when that is the declared interface. These
hooks return a native boolean decision. If `prediction` is accepted, the runtime
computes it once and passes it in; a hook needing only labels/context avoids that
extra prediction. The example hook also returns a numeric priority when invoked
without context by an older evaluator.

Both paths share the fixed evaluation pool, label caps, curve weights, and
checkpoint commitments. This preserves the old **function interface and ranking
semantics**, not old numerical results: the evaluation split and budgets have
changed. Predictors that assume exactly five labels still need to be updated.

## Prepare a real run

1. On a machine with internet access, download the current public training data
   and embeddings:

   ```bash
   python tools/prepare_data.py --out payload
   ```

   The output must be a new or empty directory. The tool downloads eligible
   `benchmarks.parquet`, `subjects.parquet`, `items.parquet`,
   `response.parquet`, and available `traces.parquet` files. Upstream
   `responses.parquet` files are stored locally as `response.parquet`.
   Each build uses the current `main` branches of the HF datasets
   `aims-foundations/measurement-db` and `aims-foundations/measurement-db-embed`.
   No historical commit is hardcoded. The versions actually downloaded are
   recorded in the payload manifests for provenance; runtime stays offline.
   Authenticate with HuggingFace beforehand if either repository requires
   access approval. HF credentials are for preparation, not the submission ZIP.
   Add `--no-embeddings` to prepare only the seven keyword/table tools.

2. Create your private configuration:

   ```bash
   cp -n submission/submission_config.example.json submission/submission_config.json
   chmod 600 submission/submission_config.json
   ```

   `llm` selects the predictor model, independently of the input subject.
   The default is `openai/gpt-5.6-luna` (GPT-5.6 Luna).
   Luna uses the Responses API with `reasoning_effort="medium"` by default.
   Set `reasoning_effort` in the private config to tune the predictor's effort;
   this is separate from the target subject's reasoning-effort attribute.
   Encrypted reasoning state is passed between retrieval turns within each
   prediction. Requests use `store=false`. The initial 32,000-token limit per
   turn includes both reasoning and visible output tokens. A turn explicitly
   cut off by the output-token limit can retry with 64,000 then 128,000 tokens
   within the same prediction deadline. Partial tool calls are never executed;
   reasoning effort stays unchanged, and usage includes these extra attempts.
   The Luna path does not import `litellm`, `openai`, or `anthropic`.
   Other predictor providers use an optional, lazily imported `litellm`; using
   them requires arranging its availability separately.
   `max_steps` and `question_timeout` control its turn limit and deadline.
   The competition example uses five turns and 240 seconds, leaving up to
   four retrievals before the final submission turn while retaining medium
   reasoning. This shorter turn budget leaves more room within the evaluation
   time limit. Individual API attempts are capped at 120 seconds; retries share
   the remaining question deadline. Local `BLF_LLM` and `BLF_MAX_STEPS`
   environment variables override the corresponding settings.

   Supply provider credentials via your environment or the private config's
   `api_keys`. The default predictor and optional semantic retrieval both use
   `OPENAI_API_KEY`. Never publish this private config
   or a submission ZIP containing credentials.
   Edit the key fields locally; do not put keys in shell commands or chat.

3. Inspect one real prediction:

   ```bash
   python tools/smoke_test.py --walkthrough
   ```

   This makes paid API requests for one prediction with example revealed labels
   and validates the streaming acquisition hook. It automatically uses `payload/data`,
   or you can set `BLF_DATA_ROOT` to another prepared data directory. Temporary
   smoke test evidence files are cleaned up after the walkthrough.

4. Package a private competition submission:

   ```bash
   ./tools/build_zip.sh --private bundle
   ```

   The ZIP places `model.py` at its root and includes supporting source plus
   `payload/data` and optional `payload/embeddings`. Its config must supply
   the credentials needed in the evaluation sandbox. To share a credential-free
   archive, use `./tools/build_zip.sh --public bundle`; it creates
   `blf_public.zip` using only the blank example configuration. The
   `--public hf <repo_id>` and `--private hf <repo_id>` build
   mode is for an organizer-approved payload mirror already prepared for local
   prefetching; the data preparation tool does not republish the corpus.

The competition accepts `requirements.txt`, but additional package installation
is disabled by default. That file is a local setup list, not a guarantee that
Codabench will install it. Bundled Luna runs use NumPy, pandas, and PyArrow for
data processing plus Python's standard library for inference. HuggingFace is
needed during local data preparation, not for the bundled runtime path.
To check this dependency boundary locally, use a fresh environment:

```bash
python -m venv /tmp/paec-runtime-check
/tmp/paec-runtime-check/bin/python -m pip install numpy pandas pyarrow
/tmp/paec-runtime-check/bin/python tools/smoke_test.py --mock
```

A local smoke test verifies behavior in that environment; it does not reproduce
the organizer's complete container or network policy.

The source release does not redistribute training data. Obtain it from the
competition's [public training release](https://huggingface.co/datasets/aims-foundations/measurement-db)
and follow its dataset card and upstream benchmark licenses and attribution.

For hosted API debugging, set `"api_preflight": true` in the private
`submission_config.json`. At import, this makes one Responses request using a
fixed public coin-flip prompt and one embedding request. It keeps the configured
reasoning effort and discards both results. A startup request failure reports a
short HTTP/provider code or connection-error type, without credentials or hidden
inputs. This adds two paid calls per model load and is disabled by default.
Mock smoke tests replace these calls and test their failure handling separately.

## Optional semantic retrieval

Without an embeddings payload, the baseline offers seven keyword/table tools.
Semantic retrieval adds an eighth tool when precomputed item vectors are
available locally. By default, `tools/prepare_data.py` downloads them from
the HF dataset `aims-foundations/measurement-db-embed`. It does not generate
new embeddings or make paid embedding API calls during preparation.

The embedding repository stores `<benchmark_dir>.parquet` files at its root.
Preparation copies eligible files into `embeddings/items/` and verifies that
their item IDs and content hashes match the downloaded database. Each file has:

- `item_id`: the original item id, converted to a string.
- `content_sha1`: SHA-1 of the original content's UTF-8 bytes, for deduplication.
- `embedding`: a list of floats containing the normalized vector.

The published vectors use `text-embedding-3-small` with 1,536 dimensions.
Preparation writes `embeddings/manifest.json` with these settings, the included
benchmarks, and both source revisions. The embedding repository need not cover
every public benchmark; uncovered benchmarks remain searchable by keyword.
Mismatched item text or invalid vectors fail preparation instead of silently
disabling semantic search.

At runtime, `OPENAI_API_KEY` is needed to embed search queries with the same
model and dimensions. The index ranks stored vectors by dot product (cosine
similarity for unit vectors). It never generates corpus vectors.
Matrix memory is roughly `number_of_vectors * dimensions * 4` bytes, plus
metadata. The configured embedding model must be available to the supplied key.

## Runtime limits and outputs

The loop checks its deadline before and after operations and passes the
remaining budget to each LLM request. Local file reads cannot be interrupted by
that check. Temporary OpenAI network failures and HTTP 408/429/500/502/503/504
responses receive at most two retries, respecting server backoff hints within
the existing per-request time budget. Temporary proxy CONNECT rejections can
receive up to eight total attempts within that same budget, with backoff capped
at eight seconds plus jitter: these failures occur before the HTTPS API request
is sent. Permanent proxy denials are not retried. Embedding attempts have a 15-second socket
timeout within their 60-second budget, leaving time to retry a stalled connection.
Each retry creates a fresh HTTP request so proxy handling preserves HTTPS port 443.
Responses keep their 120-second attempt limit. Network errors report a fixed
cause category and attempt count without copying credentials or request content.
Persistent failures still raise; no
replacement probability is returned.
Benchmark the full workload: the sampling cap counts distinct subject–item
pairs across acquisition and evaluation pools. Evaluating six label budgets
can therefore require more predictor calls than the sampling cap.

`run_agent` returns the forecast, final explanation, prompts, `belief_history`,
and `tool_log`, plus `run_id` and `log_dir` for inspecting saved evidence.
The public `predict` entry point returns only the probability. The agent creates
a fresh `run_*` directory beneath `BLF_OUT/runs/searches/` for each invocation
(default root: the system temporary directory's `paec_blf_runs/`). Repeated
identical inputs get separate log directories and independent agent runs.
These identifiers do not enter the prompts. Treat traces as local research
artifacts, not files to include in the release.
