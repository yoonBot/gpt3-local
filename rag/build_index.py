"""
Build a retrieval index from a directory of .txt documents: chunk each
file, embed the chunks with a small sentence-embedding model, and store the
embeddings + chunk text to disk. This index is entirely separate from the
GPT-3 model's weights -- retrieval never touches or changes them.

    python rag/build_index.py --docs_dir mydocs/ --out_dir rag_index/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunk import chunk_directory

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--chunk_tokens", type=int, default=256)
    p.add_argument("--overlap_tokens", type=int, default=32)
    p.add_argument("--embed_model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    sources, texts = [], []
    for source, text in chunk_directory(args.docs_dir, args.chunk_tokens, args.overlap_tokens):
        sources.append(source)
        texts.append(text)

    if not texts:
        raise ValueError(f"no .txt files (or no text in them) found under {args.docs_dir}")
    print(f"chunked {len(texts)} passages from {len(set(sources))} files")

    model = SentenceTransformer(args.embed_model)
    embeddings = model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(args.out_dir, "chunks.json"), "w") as f:
        json.dump([{"source": s, "text": t} for s, t in zip(sources, texts)], f)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"embed_model": args.embed_model, "n_chunks": len(texts)}, f)

    print(f"wrote index ({len(texts)} chunks, dim={embeddings.shape[1]}) to {args.out_dir}")


if __name__ == "__main__":
    main()
