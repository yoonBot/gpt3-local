"""Load a rag/build_index.py index and run nearest-neighbor search over it.

Uses plain numpy cosine similarity (embeddings are pre-normalized, so it's
just a dot product) rather than a library like FAISS -- for the corpus
sizes a single-GPU research project deals with (thousands to a few hundred
thousand chunks) a brute-force matmul is simple, dependency-free, and fast
enough.
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_dir: str):
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "chunks.json")) as f:
            self.chunks = json.load(f)
        with open(os.path.join(index_dir, "meta.json")) as f:
            meta = json.load(f)
        self.model = SentenceTransformer(meta["embed_model"])

    def search(self, query: str, k: int = 5):
        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.embeddings @ q_emb
        top_idx = np.argsort(-scores)[:k]
        return [
            {"score": float(scores[i]), "source": self.chunks[i]["source"], "text": self.chunks[i]["text"]}
            for i in top_idx
        ]
