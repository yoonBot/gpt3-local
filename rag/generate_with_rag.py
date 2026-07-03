"""
Retrieval-augmented generation: retrieve the top-k most relevant chunks for
a query from a rag/build_index.py index, stuff them into the prompt as
context, and let the trained GPT-3-style model generate an answer
conditioned on that context.

This does not change the model's weights or require retraining -- the
model just attends to retrieved text placed in its input context window,
same as it would attend to any other prompt text.

    python rag/generate_with_rag.py --ckpt out/ckpt.pt --index_dir rag_index/ --query "..."
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
import torch

from configs import GPT3Config
from model import GPT3
from rag.retriever import Retriever

PROMPT_TEMPLATE = """Context:
{context}

Question: {query}
Answer:"""


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--index_dir", type=str, required=True)
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--top_k_chunks", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    retriever = Retriever(args.index_dir)
    hits = retriever.search(args.query, k=args.top_k_chunks)
    print("--- retrieved context ---")
    for h in hits:
        print(f"[{h['score']:.3f}] {h['source']}: {h['text'][:100]}...")
    print()

    context = "\n\n".join(h["text"] for h in hits)
    prompt = PROMPT_TEMPLATE.format(context=context, query=args.query)

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = GPT3Config(**checkpoint["config"])
    model = GPT3(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(prompt)
    if len(ids) > cfg.block_size:
        # keep the tail (closest to the question) if context overflows the
        # model's context window
        ids = ids[-cfg.block_size:]
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        out = model.generate(
            idx, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
    print("--- answer ---")
    print(enc.decode(out[0, len(ids):].tolist()))


if __name__ == "__main__":
    main()
