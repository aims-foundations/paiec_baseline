"""Predict success using the acquired-label mean for a subject and benchmark.

Only the supplied, budget-limited evidence is used. Predictions are constant
across items in the same group; no fitted parameters or shared state are kept.
"""


def predict(input: list[dict], labeled: list | None = None) -> float:
    """Return successes / observations, or 0.5 when no matching labels exist.

    Subjects match on their complete visible attributes. Benchmarks match on
    the anonymous ``benchmark_id`` supplied in the item dictionary. Each entry
    in ``labeled`` is one observed binary response, in [[subject, item], y] form.
    The empirical mean is not smoothed or clipped away from 0 and 1.
    """
    subject, item = input
    if not labeled:
        return 0.5

    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("The mean predictor requires an anonymous item benchmark_id.")

    successes = 0
    count = 0
    for (observed_subject, observed_item), label in labeled:
        if observed_subject != subject:
            continue
        observed_benchmark = observed_item.get("benchmark_id")
        if not isinstance(observed_benchmark, str) or not observed_benchmark:
            raise ValueError("Acquired items must include an anonymous benchmark_id.")
        if observed_benchmark != benchmark:
            continue
        if label not in (0, 1):
            raise ValueError("Acquired responses must be binary (0 or 1).")
        successes += int(label)
        count += 1
    return float(successes / count) if count else 0.5
