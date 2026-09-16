# Empirical Mean with BLE Acquisition

This baseline combines the [empirical mean predictor](../empirical_mean/README.md)
with the uncertainty policy of the [Bayesian Linguistic Evaluator](../ble/README.md).
BLE estimates the probability of success for acquisition candidates, using the
previously revealed labels. Evaluation predictions are the mean acquired response
for the target subject–benchmark pair, with 0.5 when no matching labels are
available. The combination separates the contribution of acquisition from that
of BLE's item-specific predictions.

Uncertainty sampling need not improve a group mean estimate. It can select an
unrepresentative subset of items, so the baseline should be treated as a
controlled comparison with random acquisition. The implementation includes no
private evaluation data or recorded acquisition trajectories.

## Method and interface

`model.py` reuses `empirical_mean.model.predict` without importing the BLE
predictor or making API calls. The same subject matching, anonymous
`benchmark_id`, probability bounds of `[0, 1]`, and unsmoothed mean apply at every
label budget. Only evidence supplied to the current prediction is used.

`labeling.py` implements
`acquisition_function(input, *, labeled=None, context=None) -> bool`. It omits
the optional `prediction` argument because the evaluator's mean prediction is
not the uncertainty estimate needed here. Instead, the hook calls BLE with the
current candidate and previously revealed labels, then applies BLE's acquisition
policy: query when the probability is within 0.15 of 0.5, or when all remaining
candidates are needed to fill the remaining budget. Exhausted budgets return
`False`; forced selections return `True` without an unnecessary model call.

BLE is loaded only when an acquisition decision needs it. Its API failures
propagate rather than silently switching policies. The hook requires the
streaming acquisition interface, and the mean predictor requires anonymous
benchmark IDs on both target items and acquired examples. The ordinary
participant-facing prediction interface is unchanged.

## Setup and packaging

Run commands from the repository root. Follow the
[BLE setup instructions](../ble/README.md#prepare-a-real-run) to install its
dependencies, prepare `payload/` from the public measurement corpus, and configure
`ble/submission_config.json` for a private API run. This baseline reuses BLE's
configuration, prompts, retrieval code, and acquisition policy; it does not
maintain a separate copy of the method.

```bash
python tools/build_mean_ble_zip.py --public bundle
python check_submission_zip.py --static-only dist/empirical_mean_ble_acquisition_public.zip
```

The public archive includes blank example credentials, the predictor source,
and the prepared public payload. It never reads the local private BLE
configuration. `--static-only` validates the archive without making prediction
or acquisition calls; the synthetic tests below check behavior separately.
For a private competition submission using locally configured credentials:

```bash
python tools/build_mean_ble_zip.py --private bundle
python submit_api.py dist/empirical_mean_ble_acquisition_submission.zip
```

Do not publish the private archive. Within either ZIP, `model.py` and
`labeling.py` are at the root; the BLE code, configuration, and payload are under
`ble/`. The alternative `--public hf owner/repository` and
`--private hf owner/repository` modes use an organizer-approved payload mirror;
the generated `models.txt` stays at the ZIP root.

## Tests and computational cost

The synthetic checks exercise acquisition through the streaming interface,
verify that evaluation uses only the empirical mean, and load the packaged
submission in a fresh process. BLE responses are mocked, so the tests make no
paid API calls:

```bash
python -m unittest discover -s tools -p 'test_*.py'
```

API cost is incurred for acquisition candidates whose decisions require BLE.
Each such estimate may use several LLM turns and retrieval queries; evaluation
at all six budgets requires only mean calculations. Acquisition remains
sequential and can dominate runtime. An evaluator that imposes a separate
acquisition timeout must allow enough time for the BLE prediction within the
hook. Independent stochastic BLE runs may select different labels; a comparison
requiring exactly matched labels should reuse one recorded acquisition trajectory
inside the evaluator, without releasing private outcomes.
