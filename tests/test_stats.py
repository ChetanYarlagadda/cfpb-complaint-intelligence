import numpy as np
import pytest
from sklearn.metrics import f1_score

from complaintiq.stats import benjamini_hochberg, bootstrap_macro_f1_diff, holm, macro_f1, paired_bootstrap


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    pred = np.where(rng.random(500) < 0.6, y, rng.integers(0, 5, 500))
    assert macro_f1(y, pred, 5) == pytest.approx(f1_score(y, pred, average="macro"))


def test_bootstrap_detects_better_classifier():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 2000)
    weak = np.where(rng.random(2000) < 0.5, y, rng.integers(0, 4, 2000))
    strong = np.where(rng.random(2000) < 0.8, y, rng.integers(0, 4, 2000))
    res = bootstrap_macro_f1_diff(y, weak, strong, 4, n_boot=300)
    assert res["diff"] > 0.2 and res["ci_low"] > 0 and res["p_value"] < 0.01


def test_paired_bootstrap_identical_systems():
    a = np.random.default_rng(2).random(300)
    res = paired_bootstrap(a, a.copy(), n_boot=2000)
    assert res["mean_diff"] == 0.0 and res["p_value"] > 0.9


def test_holm_and_bh_hand_computed():
    # sorted p: 0.01 (x3 = 0.03), 0.02 (x2 = 0.04), 0.04 (x1 = 0.04)
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    # BH: 0.01*3/1 = 0.03, 0.02*3/2 = 0.03, 0.04*3/3 = 0.04
    assert benjamini_hochberg(np.array([0.04, 0.01, 0.02])) == pytest.approx([0.04, 0.03, 0.03])
