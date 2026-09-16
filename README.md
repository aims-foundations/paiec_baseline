# PAIEC Predictor Baselines

This repository provides reference predictors for the
[Predictive AI Evaluation Competition](https://aimslab.stanford.edu/competition).
Each predictor estimates the probability that an AI system will answer an item
correctly, using the supplied attributes and any labels acquired during evaluation.
The implementations offer starting points for developing predictors and comparing
the value of test-time adaptation.

| Predictor | Method | Acquisition | Requirements |
| --- | --- | --- | --- |
| [Bayesian Linguistic Evaluator (BLE)](ble/README.md) | An LLM retrieves public evidence and revises an explicit belief state. | Uncertainty sampling using the current prediction. | Public data payload and model API credentials. |
| [Empirical Mean](empirical_mean/README.md) | The mean acquired response for the target subject–benchmark pair; 0.5 without matching labels. | The evaluator's default random policy. | Python standard library only. |
| [Empirical Mean with BLE Acquisition](empirical_mean_ble_acquisition/README.md) | The same empirical mean predictor. | Uncertainty sampling from a separate BLE estimate. | Public data payload and API credentials for acquisition only. |

## Repository layout

```text
ble/                    Bayesian Linguistic Evaluator and its README
empirical_mean/         Empirical mean predictor and its README
empirical_mean_ble_acquisition/  Empirical mean with BLE acquisition
tools/                  Data preparation, packaging, and local tests
check_submission_zip.py Submission archive validator
submit_api.py           Codabench authentication and submission utility
```

Each predictor directory contains `model.py`, implementing
`predict(input, labeled) -> float`. The two BLE acquisition variants also supply
`labeling.py` for their policies. Build scripts place these entry points at the ZIP root,
as required by the evaluator. Generated archives go into the ignored `dist/`
directory; downloaded public training data go into the ignored `payload/`
directory. Predictor setup and implementation details belong in each
predictor's README.

## Quick start

Run commands from the repository root with Python 3.10 or later. The empirical
mean baseline can be built and checked without downloading data or making API
calls:

```bash
python tools/build_mean_zip.py
python check_submission_zip.py dist/empirical_mean.zip
```

For BLE, follow its [setup and walkthrough](ble/README.md#start-with-a-walkthrough).
Its public build uses a blank configuration; a private submission build includes
locally configured API credentials. Only credential-free archives should be
shared. The BLE-acquisition mean baseline has its own
[packaging instructions](empirical_mean_ble_acquisition/README.md#setup-and-packaging).
All predictors use the competition's published
[input and evaluation rules](https://aimslab.stanford.edu/competition).

## Submitting to Codabench

Install `requests` for the submission utility, then authenticate with an approved
competition account. Login prompts for credentials and saves the API token at
`~/.config/paiec/codabench-token` with owner-only permissions; it does not save
the password or add the token to a submission archive.

```bash
python -m pip install requests
python submit_api.py --login
python submit_api.py dist/empirical_mean.zip
python submit_api.py --list
python submit_api.py --status SUBMISSION_ID
```

Replace the archive path with `dist/ble_submission.zip` to submit BLE.
`CODABENCH_TOKEN` overrides the saved token, and `CODABENCH_TOKEN_FILE` overrides
its location. Login, list, and status commands do not submit evaluations.

## Developing additional baselines

Add each predictor in its own directory with a method README, a `model.py` entry
point, and an optional `labeling.py`. Shared development tools belong in `tools/`.
Commit source, documentation, and synthetic tests; keep credentials, datasets,
run outputs, and generated archives in ignored paths. The existing tests run
without paid API calls:

```bash
python -m unittest discover -s tools -p 'test_*.py'
```

After selectively staging changes, review the staged diff and run the source
check below. It detects prohibited artifact paths and common credential formats;
manual review is still needed to establish the provenance of example data.

```bash
git diff --cached
python tools/check_public_source.py
```
