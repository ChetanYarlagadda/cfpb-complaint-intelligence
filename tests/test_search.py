import numpy as np
import pytest

from complaintiq.search import dense_search, precision_at_k, random_precision, tfidf_search


def test_precision_at_k_counts_same_label_share():
    corpus_labels = np.array([0, 0, 1, 1, 2])
    cand = np.array([[0, 2, 1], [4, 3, 2]])
    p = precision_at_k(cand, np.array([0, 1]), corpus_labels, [1, 3])
    assert p[1].tolist() == [1.0, 0.0]
    assert p[3] == pytest.approx([2 / 3, 2 / 3])


def test_random_precision_is_corpus_share_of_query_label():
    corpus_labels = np.array([0, 0, 0, 1])
    assert random_precision(np.array([0, 1]), corpus_labels).tolist() == [0.75, 0.25]


def test_tfidf_search_finds_matching_complaint():
    corpus = [
        "zelle payment sent to a scammer selling a puppy online",
        "zelle scammer took my payment",
        "coinbase froze my crypto wallet and support never answers",
        "crypto wallet locked by coinbase support",
        "western union international transfer never arrived",
        "international transfer through western union is missing",
    ]
    idx, _ = tfidf_search(corpus, ["my crypto wallet on coinbase is frozen"], k=2)
    assert set(idx[0].tolist()) == {2, 3}


def test_dense_search_orders_by_cosine():
    corpus = np.eye(3, dtype=np.float32)
    query = np.array([[0.1, 0.9, 0.0]], dtype=np.float32)
    idx, scores = dense_search(corpus, query, 2)
    assert idx[0].tolist() == [1, 0] and scores[0, 0] == pytest.approx(0.9)
