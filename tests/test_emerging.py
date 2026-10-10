import numpy as np
import pandas as pd

from complaintiq.emerging import ctfidf_terms, detect_bursts, injection_recovery, monthly_matrix


def _flat_counts(n_topics: int = 5, n_months: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shares = np.array([0.4, 0.25, 0.15, 0.12, 0.08])[:n_topics] if n_topics <= 5 else np.full(n_topics, 1 / n_topics)
    months = pd.period_range("2022-01", periods=n_months, freq="M")
    counts = np.vstack([rng.multinomial(2000, shares / shares.sum()) for _ in months]).T
    return pd.DataFrame(counts, index=range(n_topics), columns=months)


def test_no_bursts_when_shares_are_stable():
    assert not detect_bursts(_flat_counts()).burst.any()


def test_spike_is_flagged_in_the_right_cell_only():
    counts = _flat_counts()
    month = counts.columns[15]
    counts.loc[4, month] += 600  # topic 4 jumps from ~160 to ~760 complaints
    res = detect_bursts(counts)
    flagged = res[res.burst]
    assert list(zip(flagged.topic, flagged.month)) == [(4, month)]
    assert flagged.ratio.iloc[0] > 3


def test_injection_recovery_finds_large_surges_not_small_ones():
    counts = _flat_counts(n_topics=10)  # 10% each: a 3x surge triples the share
    res = injection_recovery(counts, n_cells=10, multipliers=[0.2, 3.0], eval_months=counts.columns[6:], seed=0)
    by_mult = res.groupby("multiplier").detected.mean()
    assert by_mult[3.0] == 1.0
    assert by_mult[0.2] == 0.0


def test_monthly_matrix_fills_empty_months():
    months = pd.period_range("2024-01", periods=3, freq="M")
    m = monthly_matrix(pd.Series(pd.PeriodIndex(["2024-01", "2024-03", "2024-03"], freq="M")), np.array([0, 1, 1]), months)
    assert m.loc[1].tolist() == [0, 0, 2]
    assert m.loc[0].tolist() == [1, 0, 0]


def test_ctfidf_terms_are_topic_specific():
    texts = ["zelle scam payment zelle"] * 6 + ["crypto wallet bitcoin crypto"] * 6
    topics = np.array([0] * 6 + [1] * 6)
    terms = ctfidf_terms(texts, topics, n_terms=2, min_df=1)
    assert "zelle" in terms[0] and "crypto" in terms[1]


def test_one_flood_month_does_not_make_later_months_bursts():
    counts = _flat_counts(n_topics=5, n_months=24)
    counts.loc[0, counts.columns[12]] += 40_000  # a template campaign floods topic 0 for one month
    res = detect_bursts(counts)
    flagged = res[res.burst]
    assert list(zip(flagged.topic, flagged.month)) == [(0, counts.columns[12])]
    # the pooled baseline is dragged down by the flood and flags the other topics afterwards
    pooled = detect_bursts(counts, baseline="pooled")
    assert pooled[pooled.burst & (pooled.topic != 0)].month.min() > counts.columns[12]
