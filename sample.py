"""
Generate text from a trained checkpoint.

    python sample.py --ckpt out/ckpt.pt --prompt "Once upon a time"
"""

import argparse

import tiktoken
import torch

from configs import GPT3Config
from model import GPT3


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = GPT3Config(**checkpoint["config"])
    model = GPT3(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(args.prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        for i in range(args.num_samples):
            y = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            print(f"--- sample {i + 1} ---")
            print(enc.decode(y[0].tolist()))
            print()


if __name__ == "__main__":
    main()
