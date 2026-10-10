"""Similar-case search: TF-IDF, dense (FAISS) and cross-encoder reranking, scored by same-issue precision@k."""

from __future__ import annotations

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from complaintiq.embeddings import Reranker


def dense_search(corpus_emb: np.ndarray, query_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product (cosine on normalised vectors) top-k with a FAISS flat index."""
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(corpus_emb)
    scores, idx = index.search(query_emb, k)
    return idx, scores


def tfidf_search(corpus_texts: list[str], query_texts: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine top-k over word uni+bigram TF-IDF vectors fitted on the corpus."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english")
    corpus = vec.fit_transform(corpus_texts)
    queries = vec.transform(query_texts)
    idx = np.zeros((len(query_texts), k), dtype=np.int64)
    scores = np.zeros((len(query_texts), k), dtype=np.float32)
    for s in range(0, len(query_texts), 256):
        sims = (queries[s : s + 256] @ corpus.T).toarray()
        top = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
        idx[s : s + 256] = np.take_along_axis(top, order, axis=1)
        scores[s : s + 256] = np.take_along_axis(sims, idx[s : s + 256], axis=1)
    return idx, scores


def rerank(reranker: Reranker, query_texts: list[str], corpus_texts: list[str], cand_idx: np.ndarray, depth: int) -> np.ndarray:
    """Reorder each query's first `depth` candidates by cross-encoder score; the tail keeps its order."""
    out = cand_idx.copy()
    for q, text in enumerate(query_texts):
        head = cand_idx[q, :depth]
        s = reranker.score(text, [corpus_texts[i] for i in head])
        out[q, :depth] = head[np.argsort(-s, kind="stable")]
    return out


def precision_at_k(cand_idx: np.ndarray, query_labels: np.ndarray, corpus_labels: np.ndarray, ks: list[int]) -> dict[int, np.ndarray]:
    """Per-query share of the top-k retrieved complaints that carry the query's label."""
    hits = corpus_labels[cand_idx] == query_labels[:, None]
    return {k: hits[:, :k].mean(axis=1) for k in ks}


def random_precision(query_labels: np.ndarray, corpus_labels: np.ndarray) -> np.ndarray:
    """Expected precision@k of random retrieval for each query: the corpus share of its label."""
    n_labels = int(max(query_labels.max(), corpus_labels.max())) + 1
    share = np.bincount(corpus_labels, minlength=n_labels) / len(corpus_labels)
    return share[query_labels]
