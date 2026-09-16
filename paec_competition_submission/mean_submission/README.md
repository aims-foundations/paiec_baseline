# Acquired-label mean baseline

For each target, this predictor returns the fraction of successful responses
among acquired labels from the same subject–benchmark pair. The subject's
complete visible attributes and the item's anonymous `benchmark_id` identify
the group. Every target item in that group receives the same probability.
When no matching labels are available, including budget 0, the prediction is
0.5. The estimate is the ordinary sample mean, without smoothing: one observed
failure gives 0, and one observed success gives 1.

The ZIP omits `labeling.py`, so the streaming evaluator uses its default
deterministic random acquisition policy. The evaluator supplies nested label
sets at budgets 0, 1, 3, 7, 15, and 31. Only labels supplied to the current call
are used; nothing is retained between calls or budgets. This provides a simple
comparison for BLF's item-dependent predictions. It requires no training data,
API credentials, model requests, or third-party Python packages.

The streaming inputs must include the anonymous `benchmark_id` in both target
and acquired item dictionaries. This requires the evaluator update that adds
this field; the implementation is intended for that release. At positive
budgets, inputs without `benchmark_id` raise an error to avoid pooling labels
from different benchmarks.

From the repository root, build and validate the archive with:

```bash
python paec_competition_submission/tools/build_mean_zip.py
python paec_competition_submission/check_submission_zip.py \
  paec_competition_submission/mean_submission.zip
python -m unittest discover -s paec_competition_submission/tools \
  -p test_mean_baseline.py
```

The archive contains only `model.py` and this README. Building and checking it
does not submit an evaluation or deploy infrastructure. A lower score than the
constant-0.5 baseline can result from estimating group success rates alone;
this baseline does not model differences in difficulty between items.
