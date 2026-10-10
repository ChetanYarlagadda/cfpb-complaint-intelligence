"""Sentence Transformers bi-encoders and a cross-encoder, with embeddings cached on disk."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass(frozen=True)
class EmbedderSpec:
    key: str
    label: str
    model_id: str
    max_seq_length: int = 256


# Complaint-to-complaint similarity is symmetric, so no query instruction/prefix is used.
EMBEDDERS: dict[str, EmbedderSpec] = {
    s.key: s
    for s in [
        EmbedderSpec("minilm", "MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2"),
        EmbedderSpec("bge-small", "BGE-small", "BAAI/bge-small-en-v1.5"),
    ]
}
CROSS_ENCODER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def ids_key(ids: list[str]) -> str:
    """Short stable hash of an ordered id list (part of every cache file name)."""
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """Encodes texts with an EmbedderSpec (fp16 on the GPU); caches by spec, precision and ids."""

    def __init__(self, spec: EmbedderSpec, cache_dir: Path | None = None, batch_size: int = 128) -> None:
        self.spec = spec
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        device = _device()
        self.precision = "fp16" if device == "cuda" else "fp32"
        kwargs = {"dtype": torch.float16} if device == "cuda" else {}
        self.model = SentenceTransformer(spec.model_id, device=device, model_kwargs=kwargs)
        self.model.max_seq_length = spec.max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        """L2-normalised float32 embeddings."""
        emb = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return emb.astype(np.float32)

    def encode_cached(self, texts: list[str], ids: list[str]) -> np.ndarray:
        """Encode, or load the cached matrix for exactly these ids."""
        path = None
        if self.cache_dir is not None:
            name = f"{self.spec.key}__{self.spec.max_seq_length}__{self.precision}__{ids_key(ids)}.npy"
            path = self.cache_dir / name
            if path.exists():
                return np.load(path)
        emb = self.encode(texts)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb


class Reranker:
    """MS MARCO cross-encoder scoring (query complaint, candidate complaint) pairs; raw logits."""

    def __init__(self, model_id: str = CROSS_ENCODER_ID, max_length: int = 512) -> None:
        device = _device()
        kwargs = {"dtype": torch.float16} if device == "cuda" else {}
        self.model = CrossEncoder(model_id, max_length=max_length, device=device, model_kwargs=kwargs)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(
            self.model.predict(
                [(query, d) for d in docs], batch_size=len(docs), show_progress_bar=False, activation_fn=torch.nn.Identity()
            ),
            dtype=np.float32,
        )
