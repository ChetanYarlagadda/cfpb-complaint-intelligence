"""Classification metrics and resampling statistics (paired bootstrap, Holm and Benjamini-Hochberg)."""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion counts, rows = true class, columns = predicted class."""
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """F1 per class: 2 TP / (2 TP + FP + FN); 0 when the class never occurs and is never predicted."""
    cm = confusion(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    return np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1 over classes present in y_true."""
    present = np.bincount(y_true, minlength=n_classes) > 0
    return float(per_class_f1(y_true, y_pred, n_classes)[present].mean())


def bootstrap_macro_f1_diff(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_classes: int, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap over test documents for macro-F1(b) - macro-F1(a).

    Returns the observed difference, a 95% percentile CI, and the two-sided bootstrap
    p-value (twice the share of resamples on the other side of zero).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = macro_f1(y_true[idx], pred_b[idx], n_classes) - macro_f1(y_true[idx], pred_a[idx], n_classes)
    observed = macro_f1(y_true, pred_b, n_classes) - macro_f1(y_true, pred_a, n_classes)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": observed,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(min(1.0, max(p, 1 / n_boot))),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Mean of per-item differences b - a with a 95% bootstrap CI and sign-flip randomisation p-value."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    boots = rng.choice(d, size=(n_boot, d.size)).mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, d.size))
    null = np.abs((signs * d).mean(axis=1))
    p = (1 + np.sum(null >= abs(d.mean()) - 1e-12)) / (n_boot + 1)
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_value": float(p),
    }


def holm(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (false discovery rate control)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out
