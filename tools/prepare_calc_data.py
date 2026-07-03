"""
Generate synthetic arithmetic Q&A examples that teach the model to emit
<calc>expr</calc> around arithmetic it needs computed, then state the true
result. This is what actually gives the model reliable math: the ability
isn't "more arithmetic in the weights," it's learning to delegate to the
calculator tool at inference time (see generate_with_calc.py).

Produces data/calc/train.bin and data/calc/val.bin, tokenized with the
extended tokenizer from tools/tokenizer.py (vocab_size=50259). Fine-tune a
model on this (optionally after first pretraining it on general text with
tools/resize_embeddings.py to adapt an existing checkpoint's vocab) to teach
the tool-use behavior.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.calculator import format_result, safe_eval
from tools.tokenizer import CALC_CLOSE, CALC_OPEN, get_tool_tokenizer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "calc")

TEMPLATES = [
    "Q: What is {a} {op_word} {b}?\nA: {open}{expr}{close} = {result}\n\n",
    "Q: Compute {a} {sym} {b}.\nA: {open}{expr}{close} = {result}\n\n",
    "Q: {a} {sym} {b} = ?\nA: {open}{expr}{close} = {result}\n\n",
]

OPS = [
    ("+", "plus", -1000, 1000),
    ("-", "minus", -1000, 1000),
    ("*", "times", -100, 100),
    ("//", "divided by (floor)", 1, 100),
    ("%", "mod", 1, 100),
]


def gen_example(rng: random.Random) -> str:
    sym, op_word, lo, hi = rng.choice(OPS)
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    if sym in ("//", "%") and b == 0:
        b = 1
    expr = f"{a}{sym}{b}"
    result = format_result(safe_eval(expr))
    template = rng.choice(TEMPLATES)
    return template.format(
        a=a, b=b, sym=sym, op_word=op_word, expr=expr,
        open=CALC_OPEN, close=CALC_CLOSE, result=result,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=200_000)
    p.add_argument("--val_fraction", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    rng = random.Random(args.seed)
    enc = get_tool_tokenizer()

    os.makedirs(OUT_DIR, exist_ok=True)
    n_val = int(args.n_examples * args.val_fraction)
    n_train = args.n_examples - n_val

    for split, n in [("train", n_train), ("val", n_val)]:
        ids = []
        for _ in range(n):
            text = gen_example(rng)
            ids.extend(enc.encode(text, allowed_special="all"))
        arr = np.array(ids, dtype=np.uint16)
        out_path = os.path.join(OUT_DIR, f"{split}.bin")
        arr.tofile(out_path)
        print(f"{split}: {n:,} examples, {len(ids):,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
