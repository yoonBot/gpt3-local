"""
Adapt an already-trained checkpoint (vocab_size=50257, plain GPT-2 BPE) to
the extended tool-use vocab (vocab_size=50259, adds <calc>/</calc>) so you
can continue fine-tuning it on tools/prepare_calc_data.py's synthetic data
instead of training the tool-use behavior from scratch.

Copies over all existing rows of the token embedding / lm_head weight
(they're tied, so it's one tensor) and randomly initializes the 2 new rows
with the same std the model used at initialization time.

    python tools/resize_embeddings.py --in_ckpt out/ckpt.pt --out_ckpt out/ckpt_tool.pt
"""

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs import GPT3Config
from tools.tokenizer import VOCAB_SIZE as TOOL_VOCAB_SIZE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_ckpt", type=str, required=True)
    p.add_argument("--out_ckpt", type=str, required=True)
    p.add_argument("--init_std", type=float, default=0.02)
    args = p.parse_args()

    ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=True)
    cfg = GPT3Config(**ckpt["config"])
    old_vocab = cfg.vocab_size

    if old_vocab >= TOOL_VOCAB_SIZE:
        raise ValueError(
            f"checkpoint vocab_size={old_vocab} is already >= tool vocab_size="
            f"{TOOL_VOCAB_SIZE}; nothing to resize"
        )

    state = ckpt["model"]
    old_wte = state["transformer.wte.weight"]
    n_embd = old_wte.shape[1]

    new_wte = torch.empty(TOOL_VOCAB_SIZE, n_embd, dtype=old_wte.dtype)
    torch.nn.init.normal_(new_wte, mean=0.0, std=args.init_std)
    new_wte[:old_vocab] = old_wte
    # wte and lm_head share one Parameter at the Python level (tied
    # weights), but state_dict() saves it under both names separately, so
    # both keys need updating to the resized tensor.
    state["transformer.wte.weight"] = new_wte
    state["lm_head.weight"] = new_wte

    cfg.vocab_size = TOOL_VOCAB_SIZE
    ckpt["config"] = dataclasses.asdict(cfg)
    # optimizer state and iter count don't carry over cleanly across a vocab
    # resize (Adam moments for the embedding rows changed shape); start a
    # fresh optimizer/schedule for the fine-tuning run.
    ckpt.pop("optimizer", None)
    ckpt["iter"] = 0
    ckpt["best_val_loss"] = float("inf")

    torch.save(ckpt, args.out_ckpt)
    print(f"resized vocab {old_vocab} -> {TOOL_VOCAB_SIZE}, wrote {args.out_ckpt}")


if __name__ == "__main__":
    main()
