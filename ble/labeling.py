"""Streaming uncertainty acquisition using the prediction already computed.

Return True to purchase this candidate's response, or False to skip it.
The predictor received the same previously acquired evidence in `labeled`.
"""


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
