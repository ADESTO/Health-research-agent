"""Embedding backends.

The embedder is infrastructure used by the Discovery tools, not an agent.

- FastEmbedEmbedder: real sentence-embedding model run on CPU via ONNX (no torch/GPU needed).
  Default BAAI/bge-small-en-v1.5 (384-d) — fast enough to embed a ~200k-paper subset on a laptop.
- HashingEmbedder: deterministic bag-of-words feature hashing. No downloads, no semantics beyond
  word overlap. Used by the test-suite and as an offline fallback — never for real runs.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np

from research_agent.config import settings

_TOKEN = re.compile(r"[a-z0-9]+")

# bge models expect this prefix on *queries* (not on documents) for retrieval
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        grams = tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)


class FastEmbedEmbedder:
    def __init__(self, model_name: str, dim: int, batch_size: int = 64):
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self.model = TextEmbedding(model_name=model_name)
        self.dim = dim
        self.name = model_name
        self.batch_size = batch_size
        self._query_prefix = _BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vecs = list(self.model.embed(list(texts), batch_size=self.batch_size))
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self.model.embed([self._query_prefix + text]))), dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if settings.embedder == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
    raise ValueError(f"Unknown EMBEDDER={settings.embedder!r}")


def paper_text_for_embedding(title: str, abstract: str) -> str:
    return f"{title.strip()}. {abstract.strip()}"
