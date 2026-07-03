"""Split text files into overlapping token-length chunks for retrieval indexing."""

import os

import tiktoken

_enc = tiktoken.get_encoding("gpt2")


def chunk_text(text: str, chunk_tokens: int = 256, overlap_tokens: int = 32):
    ids = _enc.encode_ordinary(text)
    if not ids:
        return []
    stride = max(1, chunk_tokens - overlap_tokens)
    chunks = []
    for start in range(0, len(ids), stride):
        piece = ids[start:start + chunk_tokens]
        if not piece:
            break
        chunks.append(_enc.decode(piece))
        if start + chunk_tokens >= len(ids):
            break
    return chunks


def chunk_directory(docs_dir: str, chunk_tokens: int = 256, overlap_tokens: int = 32):
    """Yields (source_path, chunk_text) for every .txt file under docs_dir."""
    for root, _, files in os.walk(docs_dir):
        for fname in sorted(files):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(root, fname)
            with open(path, "r", errors="ignore") as f:
                text = f.read()
            for chunk in chunk_text(text, chunk_tokens, overlap_tokens):
                yield path, chunk
