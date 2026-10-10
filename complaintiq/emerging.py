"""Emerging-issue detection: embedding topics, c-TF-IDF labels and monthly burst tests.

Topics come from k-means on sentence embeddings (one embedding per near-duplicate group).
Each topic's monthly share of all complaints is compared with its median monthly share over
the previous `window` months. A month is a burst when the excess is significant under a
binomial model (Benjamini-Hochberg across every topic-month), the share at least `min_ratio`
times the baseline, and the count at least `min_count`.

The baseline is a median of monthly shares, not the pooled share of the window: one flood
month (January 2025 brought 41,618 complaints, 82% of them template copies) would otherwise
dominate the pooled share and make every other topic look like a burst for six months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

from complaintiq.stats import benjamini_hochberg


def kmeans_topics(emb: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Topic id per embedding (mini-batch k-means on L2-normalised vectors, i.e. spherical-ish)."""
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=4096, max_iter=200)
    return km.fit_predict(emb)


def ctfidf_terms(texts: list[str], topics: np.ndarray, n_terms: int = 6, min_df: int = 5) -> dict[int, list[str]]:
    """Top terms per topic by class-based TF-IDF (BERTopic's weighting).

    W(t, c) = tf(t, c) / |c| * log(1 + A / f(t)), with |c| the words in topic c, f(t) the
    frequency of t over all topics and A the average number of words per topic.
    """
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    counts = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    topic_ids = np.unique(topics)
    tf = np.vstack([np.asarray(counts[topics == t].sum(axis=0)).ravel() for t in topic_ids]).astype(float)
    words_per_topic = tf.sum(axis=1, keepdims=True).clip(min=1)
    avg_words = words_per_topic.mean()
    weight = tf / words_per_topic * np.log(1 + avg_words / tf.sum(axis=0).clip(min=1))
    return {int(t): list(vocab[np.argsort(-weight[i])[:n_terms]]) for i, t in enumerate(topic_ids)}


def monthly_matrix(months: pd.Series, topics: np.ndarray, all_months: pd.PeriodIndex) -> pd.DataFrame:
    """Complaint counts, rows = topic, columns = month (zeros filled)."""
    table = pd.crosstab(pd.Series(topics, name="topic"), pd.Series(months.to_numpy(), name="month"))
    return table.reindex(columns=all_months, fill_value=0)


def detect_bursts(
    counts: pd.DataFrame,
    window: int = 6,
    min_ratio: float = 2.0,
    min_count: int = 30,
    alpha: float = 0.01,
    prior: float = 0.5,
    baseline: str = "median",
) -> pd.DataFrame:
    """Score every topic-month that has a full trailing window; returns one long table.

    expected = N_t * p_hat, with p_hat the median over the previous `window` months of the
    topic's monthly share (each share smoothed with a small `prior` count so an empty topic
    doesn't get p_hat = 0). p-value = P(X >= observed) for X ~ Binomial(N_t, p_hat).
    baseline="pooled" uses the window's pooled share instead (kept for the ablation).
    """
    if baseline not in ("median", "pooled"):
        raise ValueError("baseline must be 'median' or 'pooled'")
    x = counts.to_numpy(dtype=float)
    n_topics, n_months = x.shape
    totals = x.sum(axis=0)
    shares = (x + prior) / (totals + prior * n_topics).clip(min=1e-12)
    t = np.array([m for m in range(window, n_months) if totals[m] > 0 and (totals[m - window : m] > 0).all()], dtype=int)
    if baseline == "median":
        p_hat = np.stack([np.median(shares[:, m - window : m], axis=1) for m in t], axis=1) if len(t) else np.zeros((n_topics, 0))
    else:
        base = np.stack([x[:, m - window : m].sum(axis=1) for m in t], axis=1) if len(t) else np.zeros((n_topics, 0))
        p_hat = (base + prior) / (base.sum(axis=0) + prior * n_topics)
    expected = totals[t] * p_hat
    observed = x[:, t]
    p = binom.sf(observed - 1, totals[t], p_hat)
    out = pd.DataFrame(
        {
            "topic": np.repeat(counts.index.to_numpy(), len(t)),
            "month": np.tile(counts.columns[t].to_numpy(), n_topics),
            "count": observed.ravel().astype(int),
            "expected": expected.ravel(),
            "ratio": (observed / expected).ravel(),
            "p_value": p.ravel(),
        }
    )
    out["q_value"] = benjamini_hochberg(out.p_value.to_numpy())
    out["burst"] = (out.q_value < alpha) & (out.ratio >= min_ratio) & (out["count"] >= min_count)
    return out


def injection_recovery(
    counts: pd.DataFrame,
    n_cells: int,
    multipliers: list[float],
    eval_months: pd.PeriodIndex,
    seed: int = 0,
    **detector_kwargs,
) -> pd.DataFrame:
    """Add synthetic surges of `multiplier x expected` complaints to random quiet topic-months and test recovery.

    A surge of +m complaints in topic c, month t also raises that month's total. Cells that
    are already bursts in the real data are never used.
    """
    rng = np.random.default_rng(seed)
    base = detect_bursts(counts, **detector_kwargs)
    quiet = base[~base.burst & base.month.isin(eval_months) & (base.expected >= 5)].reset_index(drop=True)
    cells = quiet.iloc[rng.choice(len(quiet), size=min(n_cells, len(quiet)), replace=False)]
    rows = []
    for cell in cells.itertuples():
        for mult in multipliers:
            m = int(round(mult * cell.expected))
            injected = counts.copy()
            injected.loc[cell.topic, cell.month] += m
            res = detect_bursts(injected, **detector_kwargs)
            hit = res[(res.topic == cell.topic) & (res.month == cell.month)].iloc[0]
            rows.append(
                {
                    "topic": cell.topic,
                    "month": str(cell.month),
                    "expected": cell.expected,
                    "multiplier": mult,
                    "injected": m,
                    "detected": bool(hit.burst),
                    "other_new_bursts": int(res.burst.sum() - base.burst.sum() - int(hit.burst)),
                }
            )
    return pd.DataFrame(rows)
