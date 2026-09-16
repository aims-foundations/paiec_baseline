"""Acquire labels using BLE uncertainty, independently of the mean predictor."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if not (_ROOT / "ble").is_dir():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ble.labeling import acquisition_function as uncertainty_policy


def _ble_prediction(input, labeled):
    # Import only when an acquisition decision actually needs a model request.
    from ble.model import predict

    return predict(input, labeled=labeled)


def acquisition_function(input: list[dict], *, labeled: list | None = None,
                         context: dict | None = None) -> bool:
    """Use BLE's probability for acquisition and only previously revealed labels.

    Omitting `prediction` from the signature tells the streaming evaluator not
    to compute the empirical mean for an acquisition decision. The mean remains
    the prediction returned for every evaluation item at every label budget.
    """
    if context is None:
        raise ValueError("BLE acquisition requires streaming context")
    if context["labels_remaining"] <= 0:
        return False
    if context["items_remaining"] <= context["labels_remaining"]:
        return True
    probability = _ble_prediction(input, labeled if labeled is not None else [])
    return uncertainty_policy(
        input, prediction=probability, labeled=labeled, context=context,
    )
