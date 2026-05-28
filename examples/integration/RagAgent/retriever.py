"""Embedding-based retriever for the RAG agent demo.

Wraps a sentence-transformers encoder and a FAISS inner-product index over the
Wikipedia corpus. Embeddings are cached on disk so the first run pays the
encoding cost once.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path(__file__).parent
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _fingerprint(docs: List[dict], model_name: str) -> str:
    """Stable hash of (corpus, model) — invalidates cache when either changes."""
    h = hashlib.sha1()
    h.update(model_name.encode())
    for d in docs:
        h.update(d["id"].encode())
        h.update(d["text"].encode())
    return h.hexdigest()[:12]


class Retriever:
    """Encodes a corpus with a sentence-transformer and searches via FAISS."""

    def __init__(
        self,
        docs: List[dict],
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path = CACHE_DIR,
    ):
        self.docs = docs
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self._fp = _fingerprint(docs, model_name)
        self._emb_path = cache_dir / f"embeddings_{self._fp}.npy"
        self._index_path = cache_dir / f"faiss_{self._fp}.index"
        self.index, self.embeddings = self._build_or_load()

    def _build_or_load(self):
        if self._emb_path.exists() and self._index_path.exists():
            embeddings = np.load(self._emb_path)
            index = faiss.read_index(str(self._index_path))
            return index, embeddings

        texts = [d["text"] for d in self.docs]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # so inner product = cosine similarity
            convert_to_numpy=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        np.save(self._emb_path, embeddings)
        faiss.write_index(index, str(self._index_path))
        return index, embeddings

    def search(self, query: str, k: int = 3) -> List[dict]:
        """Return the top-k passages by cosine similarity to ``query``."""
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores, idx = self.index.search(q, k)
        out = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            doc = dict(self.docs[i])
            doc["score"] = float(score)
            out.append(doc)
        return out


if __name__ == "__main__":
    import sys
    from corpus import load_corpus

    docs = load_corpus()
    print(f"Loaded {len(docs)} passages")

    retriever = Retriever(docs)
    print(f"Index built with {retriever.index.ntotal} vectors")

    queries = [
        "When did Voyager 1 enter interstellar space?",
        "Who first mapped the Atlantic Ocean floor?",
        "What is the smallest living relative of the giraffe?",
        "How do tardigrades survive in space?",
    ]
    for q in queries:
        print(f"\n>>> {q}")
        for hit in retriever.search(q, k=2):
            print(f"  [{hit['score']:.3f}] {hit['title']:25s}  {hit['text'][:120]}...")
