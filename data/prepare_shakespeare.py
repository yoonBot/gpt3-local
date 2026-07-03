"""
Quick smoke-test dataset: tiny-shakespeare, tokenized with GPT-2 BPE.
Produces data/shakespeare/train.bin and data/shakespeare/val.bin.

Use this to sanity-check the model/training loop end-to-end in minutes
before committing to a large real dataset.
"""

import os
import urllib.request

import numpy as np
import tiktoken

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_DIR = os.path.join(os.path.dirname(__file__), "shakespeare")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "input.txt")
    if not os.path.exists(raw_path):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, raw_path)

    with open(raw_path, "r") as f:
        data = f.read()

    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens, val has {len(val_ids):,} tokens")

    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(OUT_DIR, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(OUT_DIR, "val.bin"))
    print(f"wrote train.bin / val.bin to {OUT_DIR}")


if __name__ == "__main__":
    main()
