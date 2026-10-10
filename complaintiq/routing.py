"""Complaint routing: predict the CFPB issue (the routing queue) from the narrative.

Splits are temporal: models learn from older complaints and are scored on later ones, the way
a deployed router would be. Near-duplicate groups never straddle a split.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from complaintiq.stats import macro_f1


@dataclass
class RoutingSplits:
    """Deduplicated train / validation / test complaints with integer issue labels."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    test_all: pd.DataFrame  # every test-period complaint incl. template copies (label-filtered)
    labels: list[str]
    dropped: pd.DataFrame  # issues dropped for having too few training examples


def build_splits(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    val_frac: float = 0.1,
    min_train_per_class: int = 100,
) -> RoutingSplits:
    """Temporal train/val/test split with one complaint per near-duplicate group.

    The validation set is the latest `val_frac` of the training period. Test complaints whose
    near-duplicate group already appeared before the test period are removed (template leakage).
    """
    in_train = (df.date >= train_start) & (df.date <= train_end)
    in_test = (df.date >= test_start) & (df.date <= test_end)
    train = df[in_train].drop_duplicates("group").sort_values(["date", "complaint_id"])
    seen = set(df.loc[df.date < test_start, "group"])
    test_all = df[in_test & ~df.group.isin(seen)]
    test = test_all.drop_duplicates("group")

    counts = train.issue.value_counts()
    keep = sorted(counts[counts >= min_train_per_class].index)
    dropped = counts[counts < min_train_per_class].rename("train_count").to_frame()
    label_id = {lab: i for i, lab in enumerate(keep)}

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame[frame.issue.isin(label_id)].copy()
        frame["y"] = frame.issue.map(label_id).astype(int)
        return frame.reset_index(drop=True)

    train = encode(train)
    n_val = int(round(len(train) * val_frac))
    return RoutingSplits(
        train=train.iloc[: len(train) - n_val].reset_index(drop=True),
        val=train.iloc[len(train) - n_val :].reset_index(drop=True),
        test=encode(test),
        test_all=encode(test_all),
        labels=keep,
        dropped=dropped,
    )


def tfidf_lr(C: float, seed: int = 0) -> Pipeline:
    """Word uni+bigram TF-IDF (sublinear tf) into a class-balanced multinomial logistic regression."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=200_000)),
            ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)),
        ]
    )


def tune_tfidf_lr(
    train_texts: list[str], y_train: np.ndarray, val_texts: list[str], y_val: np.ndarray, Cs: list[float], n_classes: int
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit one model per C, pick the best validation macro-F1, return it with the tuning table."""
    rows, best, best_score = [], None, -1.0
    for C in Cs:
        t0 = time.perf_counter()
        model = tfidf_lr(C).fit(train_texts, y_train)
        score = macro_f1(y_val, model.predict(val_texts), n_classes)
        rows.append({"C": C, "val_macro_f1": score, "fit_s": time.perf_counter() - t0})
        if score > best_score:
            best, best_score = model, score
    return best, pd.DataFrame(rows)


def iid_cv_macro_f1(texts: list[str], y: np.ndarray, C: float, n_classes: int, folds: int = 5, seed: int = 0) -> list[float]:
    """Macro-F1 of random stratified K-fold CV inside the training period (the optimistic, non-temporal view)."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    texts_arr = np.asarray(texts, dtype=object)
    scores = []
    for tr, te in skf.split(texts_arr, y):
        model = tfidf_lr(C).fit(list(texts_arr[tr]), y[tr])
        scores.append(macro_f1(y[te], model.predict(list(texts_arr[te])), n_classes))
    return scores


def risk_coverage(y_true: np.ndarray, proba: np.ndarray, coverages: np.ndarray | None = None) -> pd.DataFrame:
    """Accuracy on the most confident fraction of complaints (auto-route those, send the rest to triage)."""
    if coverages is None:
        coverages = np.linspace(0.05, 1.0, 20)
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    order = np.argsort(-conf, kind="stable")
    rows = []
    for c in coverages:
        n = max(1, int(round(c * len(order))))
        top = order[:n]
        rows.append({"coverage": float(c), "accuracy": float(correct[top].mean()), "min_confidence": float(conf[top].min())})
    return pd.DataFrame(rows)


def top_confusions(cm: np.ndarray, labels: list[str], n: int = 8) -> pd.DataFrame:
    """Largest off-diagonal cells as a share of the true class."""
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    rows = [
        {"true": labels[i], "predicted": labels[j], "count": int(cm[i, j]), "share_of_true": float(share[i, j])}
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j
    ]
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(n).reset_index(drop=True)
