# Bayesian Linguistic Evaluator (BLE)

The [Predictive AI Evaluation Competition](https://aimslab.stanford.edu/competition)
calls `predict([subject, item], labeled)` once per target response. This baseline
asks an LLM to estimate P(correct), retrieve related public evidence, and revise
its estimate before submitting. It fits no parameters to the response tables.

The LLM expresses Bayesian reasoning in its prompts; there is no separate
numerical posterior-update formula. This README covers the method and local
setup; shared submission instructions are in the [repository README](../README.md).

A separate [empirical mean predictor](../empirical_mean/README.md) estimates each
subject–benchmark pair's success rate from acquired labels, with 0.5 at budget
0. It requires no API calls or training corpus and uses default random
acquisition. Its anonymous benchmark input field is prepared for the next
streaming infrastructure release; see its README for packaging and checks.

## Start with a walkthrough

Run all commands in this README from the repository root, in a Python 3.10+
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ble/requirements.txt
python tools/smoke_test.py --mock --walkthrough
```

Mock mode uses small synthetic tables and scripted LLM replies. It needs no
API keys or downloaded corpus, exercises the real retrieval and agent loop,
and shows when evidence reaches the next turn. Its example probabilities are
illustrative. It also checks invalid calls, API/tool failures, deadlines,
probability bounds, revealed-label handling, and streaming acquisition decisions using previously acquired evidence.

## Read the method in this order

| File | What to look for |
| --- | --- |
| `ble/model.py` | `predict` contains question preparation, agent settings, the agent call, validation, and clipping. Read `build_question` next; configuration is loaded once above the functions, before the agent imports. |
| `ble/src/agent/agent.py` | `run_agent`: the complete estimate/retrieve/revise/submit loop and its run logs. |
| `ble/src/agent/prompts.py` | The instructions governing evidence selection and probability revision. |
| `ble/src/agent/belief_state.py` | The running estimate, cumulative evidence, and shared probability bounds. |
| `ble/src/agent/tools.py` | How an LLM-supplied belief is stored and a tool action is executed. |
| `ble/comp_pipeline/script/agent_tools.py` | The retrieval tools' arguments and intended uses. |
| `ble/comp_pipeline/script/databank.py` | Keyword ranking, score aggregation, item lookup, and trace retrieval. |
| `ble/labeling.py` | The optional uncertainty-sampling acquisition function; participants can replace it with their own strategy. |

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

## Acquisition policy

`acquisition_function(input, prediction=None, labeled=None, context=None)` uses
the prediction already computed for the current acquisition candidate,
conditioned on previously acquired labels. It requests the candidate's response
when the probability is within 0.15 of 0.5, or when every remaining candidate
must be selected to fill the remaining budget. It returns a boolean decision
and makes no additional model call.

The policy's benefit over random acquisition requires empirical validation.
Its one-argument compatibility path returns an item-length priority for older
evaluators; the streaming interface uses the uncertainty policy above. Pool
assignment, label budgets, and scoring follow the competition's
[test-time adaptation and evaluation rules](https://aimslab.stanford.edu/competition).

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
   cp -n ble/submission_config.example.json ble/submission_config.json
   chmod 600 ble/submission_config.json
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
   ./tools/build_zip.sh --private bundle  # dist/ble_submission.zip
   ```

   The ZIP places `model.py` and `labeling.py` at its root and includes supporting source plus
   `payload/data` and optional `payload/embeddings`. Its config must supply
   the credentials needed in the evaluation sandbox. To share a credential-free
   archive, use `./tools/build_zip.sh --public bundle`; it creates
   `dist/ble_public.zip` using only the blank example configuration. The
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

Runtime environment settings retain their `BLF_` names for compatibility.
Generated archives are written to `dist/`; shared upload and status commands
are in the [repository README](../README.md#submitting-to-codabench).
