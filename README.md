# PAEC Predictor Baselines

This repository contains reference implementations of two predictors for the
[Predictive AI Evaluation Competition](https://aimslab.stanford.edu/competition).
The BLF predictor retrieves evidence from the public measurement corpus and
updates an explicit belief state; its acquisition function selects uncertain
examples. The mean predictor estimates success rates from acquired labels and
uses the platform's default random acquisition. Additional baselines can be
added alongside these implementations.

## Repository layout

All baseline source is under [`paec_competition_submission/`](paec_competition_submission/):

- `submission/model.py` implements the required `predict(input, labeled)`
  entry point.
- `submission/labeling.py` implements uncertainty sampling using BLF predictions
  conditioned on previously acquired labels.
- `submission/src/` contains the vendored BLF agent and configuration code.
- `submission/comp_pipeline/` contains the offline retrieval tools.
- `tools/prepare_data.py` prepares the external data payload.
- `tools/build_zip.sh` builds a competition-compatible archive.
- `tools/smoke_test.py` exercises the submission locally.

Generated payloads, private configuration, runtime state, logs, and submission
ZIP files are intentionally excluded from Git.

## Local smoke test

From the competition directory, run the mock test without paid LLM calls:

```bash
cd paec_competition_submission
python3 tools/smoke_test.py --mock
```

## Private submission configuration

The tracked configuration is credential-free. Create an ignored private copy
only when building a real competition submission:

```bash
cd paec_competition_submission
cp submission/submission_config.example.json \
   submission/submission_config.json
```

Use dedicated, spend-limited API keys. Never commit the private configuration
or publish an archive containing those keys.

After preparing the public data payload, build with an explicit mode:

```bash
./tools/build_zip.sh --public bundle   # blf_public.zip, blank credentials
./tools/build_zip.sh --private bundle  # blf_submission.zip, local credentials
```

Public builds use only the example configuration, even when a private
configuration exists. Recipients must supply their own credentials to run BLF.

See the [competition baseline documentation](paec_competition_submission/README.md)
for payload preparation, packaging, and runtime details.

## Mean predictor baseline

The [mean predictor](paec_competition_submission/mean_submission/README.md)
provides a simpler comparison: it predicts the mean acquired response for each
subject–benchmark pair, using 0.5 before any matching labels are observed. It
uses the platform's default random acquisition and needs no model API or data
payload. Run `python paec_competition_submission/tools/build_mean_zip.py` to
create `paec_competition_submission/mean_submission.zip`. Its group matching
requires `benchmark_id` in the supplied item dictionaries; see the baseline's
README for evaluator compatibility.

## Codabench authentication and status

From the repository root, save a Codabench token without putting credentials
in chat or shell history:

```bash
python submit_api.py --login
python submit_api.py --list
python submit_api.py --status SUBMISSION_ID
```

Login prompts for your username and password and stores only the API token at
`~/.config/paiec/codabench-token` with permissions `600`. It never saves the
password or includes this token in the competition ZIP. `CODABENCH_TOKEN` takes
precedence; `CODABENCH_TOKEN_FILE` overrides the saved-token location.
The login, list, and status commands do not submit or rerun evaluations.
Detailed logs remain subject to the competition's participant-access policy.

To upload a prepared archive, use `python submit_api.py PATH_TO_ZIP`.

## Developing additional baselines

Commit source, documentation, and synthetic tests. Keep generated data, run
outputs, submission archives, and private configuration in the ignored paths.
After staging the intended files, review and check the exact staged contents:

```bash
git diff --cached
python paec_competition_submission/tools/check_public_source.py
```

The check detects prohibited artifact paths and common credential formats;
manual review is still needed to establish the provenance of example data.
