"""Near-duplicate grouping with MinHash + LSH over word shingles.

Complaint databases receive template campaigns: thousands of near-identical narratives
filed within days. Grouping them matters twice. For model evaluation, a template seen in
training makes test complaints trivially easy. For emerging-issue monitoring, the number of
distinct groups separates organic growth from a coordinated campaign.

Jaccard similarity of two shingle sets equals the probability that their minimum hash under
a random permutation agrees, so the share of agreeing signature slots estimates it.

Shingle hashes are polynomial in vocabulary ids and the permutations are affine mod 2^31 - 1.
That is accurate on natural text (`minhash_check` compares estimates with exact Jaccard on
real complaint pairs), but it is not a strong mixing hash: on synthetic text where every
token is new, the shingle hashes form an arithmetic progression and the estimate is biased.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_P = np.uint64(2_147_483_647)  # Mersenne prime 2^31 - 1: products of two residues fit in uint64
_BASE = np.uint64(1_000_003)
_TOKEN = re.compile(r"[a-z0-9]+")
_REDACTED = re.compile(r"x{2,}")


def tokens(text: str) -> list[str]:
    """Lower-case word tokens with redaction marks (XXXX) removed."""
    return [t for t in _TOKEN.findall(text.lower()) if not _REDACTED.fullmatch(t)]


def shingle_hashes(toks: list[str], vocab: dict[str, int], k: int = 5) -> np.ndarray:
    """Hashes (mod 2^31 - 1) of the word k-shingles of one document; a short doc is one shingle."""
    if not toks:
        toks = ["<empty>"]
    ids = np.fromiter((vocab.setdefault(t, len(vocab) + 1) for t in toks), dtype=np.uint64, count=len(toks))
    ids %= _P
    width = min(k, len(ids))
    powers = np.array([pow(int(_BASE), j, int(_P)) for j in range(width)], dtype=np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(ids, width)
    return np.unique(((windows * powers) % _P).sum(axis=1) % _P)


def minhash_signatures(texts: list[str], num_perm: int = 128, k: int = 5, seed: int = 0) -> np.ndarray:
    """(n_docs, num_perm) MinHash signatures under random affine hash functions mod 2^31 - 1."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(_P), size=num_perm, dtype=np.uint64)[:, None]
    b = rng.integers(0, int(_P), size=num_perm, dtype=np.uint64)[:, None]
    vocab: dict[str, int] = {}
    sig = np.empty((len(texts), num_perm), dtype=np.uint64)
    for i, text in enumerate(texts):
        h = shingle_hashes(tokens(text), vocab, k)[None, :]
        sig[i] = ((a * h + b) % _P).min(axis=1)
    return sig


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray:
    """Share of agreeing signature slots (broadcasts over rows)."""
    return (sig_a == sig_b).mean(axis=-1)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def near_duplicate_groups(sig: np.ndarray, bands: int = 32, threshold: float = 0.5) -> np.ndarray:
    """Group ids (0..G-1, in order of first appearance) linking documents with estimated Jaccard >= threshold.

    LSH: documents that agree on every slot of at least one band become candidates. Each
    bucket is verified against its first member (a star, not all pairs) so a 15,000-member
    template bucket costs 15,000 comparisons rather than 10^8.
    """
    n, num_perm = sig.shape
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands
    uf = _UnionFind(n)
    for b in range(bands):
        _, inverse = np.unique(sig[:, b * rows : (b + 1) * rows], axis=0, return_inverse=True)
        inverse = inverse.ravel()
        order = np.argsort(inverse, kind="stable")
        starts = np.flatnonzero(np.r_[True, np.diff(inverse[order]) != 0])
        ends = np.r_[starts[1:], n]
        multi = (ends - starts) >= 2
        for s, e in zip(starts[multi], ends[multi]):
            members = order[s:e]
            anchor = members[0]
            similar = members[1:][estimated_jaccard(sig[members[1:]], sig[anchor]) >= threshold]
            for m in similar:
                uf.union(int(anchor), int(m))
    roots = np.array([uf.find(i) for i in range(n)])
    _, first_idx, labels = np.unique(roots, return_index=True, return_inverse=True)
    # relabel so that group ids follow the order of each group's first document
    rank = np.empty_like(first_idx)
    rank[np.argsort(first_idx)] = np.arange(len(first_idx))
    return rank[labels.ravel()]


def exact_jaccard(text_a: str, text_b: str, k: int = 5) -> float:
    """Exact Jaccard similarity of two documents' word k-shingle sets."""

    def shingle_set(text: str) -> set[tuple[str, ...]]:
        toks = tokens(text) or ["<empty>"]
        return {tuple(toks[i : i + k]) for i in range(max(1, len(toks) - k + 1))}

    a, b = shingle_set(text_a), shingle_set(text_b)
    return len(a & b) / len(a | b)


def minhash_check(df: pd.DataFrame, sig: np.ndarray, n_pairs: int = 800, seed: int = 0) -> pd.DataFrame:
    """Estimated vs exact Jaccard for random pairs inside near-duplicate groups and for random
    pairs of different groups that share company and month (the hardest negatives)."""
    rng = np.random.default_rng(seed)
    texts = df["narrative"].to_numpy()
    groups = df["group"].to_numpy()
    multi = [v for v in df.groupby("group").indices.values() if len(v) >= 2]
    cells = [v for v in df.groupby(["company", "month"]).indices.values() if len(v) >= 20]
    pairs = [("same group", *rng.choice(multi[g], 2, replace=False)) for g in rng.choice(len(multi), min(n_pairs, len(multi)), replace=False)]
    while len(pairs) < 2 * n_pairs and cells:
        i, j = rng.choice(cells[rng.integers(len(cells))], 2, replace=False)
        if groups[i] != groups[j]:
            pairs.append(("different groups, same company and month", i, j))
    return pd.DataFrame(
        [{"pair": kind, "exact": exact_jaccard(texts[i], texts[j]), "estimated": float(estimated_jaccard(sig[i], sig[j]))} for kind, i, j in pairs]
    )


def group_summary(df: pd.DataFrame, group_col: str = "group") -> pd.DataFrame:
    """One row per near-duplicate group with at least two complaints, largest first."""
    g = df.groupby(group_col)
    out = pd.DataFrame(
        {
            "n_complaints": g.size(),
            "first_date": g["date"].min().dt.strftime("%Y-%m-%d"),
            "last_date": g["date"].max().dt.strftime("%Y-%m-%d"),
            "top_company": g["company"].agg(lambda s: s.value_counts().index[0]),
            "top_issue": g["issue"].agg(lambda s: s.value_counts().index[0]),
            "share_top_issue": g["issue"].agg(lambda s: s.value_counts(normalize=True).iloc[0]),
            "example_start": g["text"].first().str[:160],
        }
    )
    return out[out.n_complaints >= 2].sort_values("n_complaints", ascending=False)
