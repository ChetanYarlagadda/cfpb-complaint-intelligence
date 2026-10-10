import numpy as np
import pandas as pd

from complaintiq.routing import build_splits, risk_coverage, top_confusions


def _toy_complaints() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, 600), unit="D")
    issues = rng.choice(["Fraud or scam", "Other transaction problem", "Rare issue"], p=[0.6, 0.38, 0.02], size=600)
    df = pd.DataFrame({"complaint_id": [str(i) for i in range(600)], "date": dates, "issue": issues, "text": "t"})
    df["group"] = np.arange(600)
    # a template group that starts in training and continues into the test period
    df.loc[[0, 1, 2], "date"] = pd.to_datetime(["2023-03-01", "2025-02-01", "2025-03-01"])
    df.loc[[0, 1, 2], "group"] = 10_000
    return df.sort_values("date").reset_index(drop=True)


def test_temporal_split_without_group_leakage():
    sp = build_splits(_toy_complaints(), "2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31", val_frac=0.1, min_train_per_class=20)
    assert sp.train.date.max() <= sp.val.date.min()  # validation is the latest slice of training
    assert sp.val.date.max() < sp.test.date.min()
    assert not set(sp.test.group) & set(sp.train.group)  # template seen in training is removed from test
    assert sp.labels == ["Fraud or scam", "Other transaction problem"]  # rare issue dropped
    assert "Rare issue" in sp.dropped.index
    assert set(sp.test.y) <= {0, 1}


def test_risk_coverage_full_coverage_is_plain_accuracy():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.55, 0.45]])
    rc = risk_coverage(y, proba, coverages=np.array([0.25, 1.0]))
    assert rc.accuracy.tolist() == [1.0, 0.75]


def test_top_confusions_ignore_diagonal():
    cm = np.array([[50, 7], [3, 40]])
    out = top_confusions(cm, ["a", "b"], n=2)
    assert out.iloc[0].to_dict()["true"] == "a" and out.iloc[0]["count"] == 7
    assert len(out) == 2
