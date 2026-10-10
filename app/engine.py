"""Inference engine behind the API: issue routing plus similar-case search over past complaints.

Loads the artifacts `scripts/run_pipeline.py` writes to data/cache/api/: the TF-IDF + LR
router, the 2023-24 complaint corpus with its BGE-small embeddings, and (optionally) the
fine-tuned DistilBERT router and the cross-encoder.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from complaintiq.data import clean_text

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "data" / "cache" / "api"


@dataclass
class IssueScore:
    issue: str
    probability: float


@dataclass
class Case:
    complaint_id: str
    date: str
    issue: str
    sub_product: str
    company: str
    snippet: str
    score: float


class Engine(Protocol):
    router_name: str

    def route(self, text: str, k: int) -> list[IssueScore]: ...

    def similar(self, text: str, k: int, rerank: bool) -> list[Case]: ...


class ComplaintEngine:
    """Routes a narrative to CFPB issues and finds the most similar past complaints."""

    def __init__(self, api_dir: Path = API_DIR, router: str = "tfidf", reranker: bool = True, rerank_depth: int = 20) -> None:
        import faiss
        import torch

        from complaintiq.embeddings import EMBEDDERS, Embedder, Reranker

        torch.set_num_threads(2)
        config = json.loads((api_dir / "config.json").read_text())
        bundle = joblib.load(api_dir / "tfidf_lr.joblib")
        self.labels: list[str] = bundle["labels"]
        self.router_name = router
        if router == "distilbert":
            from complaintiq.finetune import load_finetuned

            self._bert, self._tok = load_finetuned(ROOT / config["distilbert_dir"])
            self._max_length = config["distilbert_max_length"]
        else:
            self._tfidf = bundle["model"]
        self.corpus = pd.read_parquet(api_dir / "corpus.parquet")
        self.embedder = Embedder(EMBEDDERS[config["embedder"]])
        emb = np.load(api_dir / f"corpus_{config['embedder']}.npy")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.reranker = Reranker() if reranker else None
        self.rerank_depth = rerank_depth

    def route(self, text: str, k: int = 3) -> list[IssueScore]:
        """Top-k issues with probabilities."""
        text = clean_text(text)
        if self.router_name == "distilbert":
            from complaintiq.finetune import predict_proba

            proba = predict_proba(self._bert, self._tok, [text], self._max_length)[0]
        else:
            proba = self._tfidf.predict_proba([text])[0]
        top = np.argsort(-proba)[:k]
        return [IssueScore(self.labels[i], float(proba[i])) for i in top]

    def similar(self, text: str, k: int = 5, rerank: bool = False) -> list[Case]:
        """Nearest 2023-24 complaints by embedding cosine, optionally reordered by the cross-encoder."""
        text = clean_text(text)
        q = self.embedder.encode([text])
        depth = max(k, self.rerank_depth) if rerank and self.reranker else k
        scores, idx = self.index.search(q, depth)
        ranked = list(zip(idx[0].tolist(), scores[0].tolist()))
        if rerank and self.reranker is not None:
            ce = self.reranker.score(text, [self.corpus.text.iloc[i] for i, _ in ranked])
            ranked = [(ranked[j][0], float(ce[j])) for j in np.argsort(-ce, kind="stable")]
        out = []
        for i, s in ranked[:k]:
            r = self.corpus.iloc[i]
            out.append(Case(r.complaint_id, f"{r.date:%Y-%m-%d}", r.issue, r.sub_product, r.company, r.text[:300], float(s)))
        return out


def build_default_engine() -> ComplaintEngine:
    """Engine configured by env vars COMPLAINTIQ_ROUTER (tfidf | distilbert) and COMPLAINTIQ_RERANKER (1 | 0)."""
    return ComplaintEngine(
        router=os.environ.get("COMPLAINTIQ_ROUTER", "tfidf"),
        reranker=os.environ.get("COMPLAINTIQ_RERANKER", "1") == "1",
    )
