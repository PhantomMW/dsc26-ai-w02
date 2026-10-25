"""Decision metrics for a mule-risk score."""
from __future__ import annotations

import math
import numpy as np


def _validate_inputs(y_true, scores):
    try:
        y = np.asarray(y_true)
        s = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid labels or scores") from exc

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(y) == 0 or len(s) == 0:
        raise ValueError("Inputs must not be empty")

    if len(y) != len(s):
        raise ValueError("Labels and scores must have equal lengths")

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Labels must be 0 or 1")

    if np.isnan(s).any():
        raise ValueError("Scores must not contain NaN")

    return y.astype(int), s


def confusion_at(y_true, scores, threshold: float) -> dict:
    y, s = _validate_inputs(y_true, scores)
    flagged = s >= threshold

    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    tn = int(np.sum(~flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at(
    y_true, scores, threshold: float
) -> tuple[float, float]:
    counts = confusion_at(y_true, scores, threshold)

    flagged = counts["tp"] + counts["fp"]
    positives = counts["tp"] + counts["fn"]

    precision = counts["tp"] / flagged if flagged else 0.0
    recall = counts["tp"] / positives if positives else 0.0

    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    y, s = _validate_inputs(y_true, scores)

    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, len(y))

    # Stable sorting preserves the original order when scores tie.
    order = np.argsort(-s, kind="stable")
    return float(np.mean(y[order[:k]]))


def expected_cost(
    y_true,
    scores,
    threshold: float,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    counts = confusion_at(y_true, scores, threshold)

    return float(
        counts["fp"] * cost_fp + counts["fn"] * cost_fn
    )


def best_threshold(
    y_true,
    scores,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    _, s = _validate_inputs(y_true, scores)

    candidates = list(np.unique(s)) + [math.inf]

    # Highest threshold wins when expected costs are tied.
    return float(
        min(
            candidates,
            key=lambda threshold: (
                expected_cost(
                    y_true,
                    s,
                    threshold,
                    cost_fp=cost_fp,
                    cost_fn=cost_fn,
                ),
                -threshold,
            ),
        )
    )


def threshold_for_precision(
    y_true, scores, min_precision: float
) -> float | None:
    _, s = _validate_inputs(y_true, scores)

    candidates = np.unique(s)

    valid = [
        float(threshold)
        for threshold in candidates
        if precision_recall_at(
            y_true, s, threshold
        )[0] >= min_precision
    ]

    return min(valid) if valid else None
