"""
Real pretraining dataset: OpenWebText (an open reproduction of the WebText
corpus GPT-2/GPT-3 were originally trained on), tokenized with GPT-2 BPE.

Uses the "Skylion007/openwebtext" mirror rather than the canonical
"openwebtext" repo on the Hub -- the canonical one is a legacy,
non-namespaced, loading-script-based dataset, and recent huggingface_hub
versions raise `HfUriError: Repository id must be 'namespace/name'` trying
to resolve it (a real incompatibility, not a transient error -- retrying
or waiting doesn't help). Skylion007's mirror is the same corpus, hosted
as plain parquet files under a proper namespaced repo, which sidesteps
that entirely.

Requires: pip install datasets tiktoken tqdm
Produces data/openwebtext/train.bin and data/openwebtext/val.bin.

Warning: the full dataset is ~54GB of text / ~9B tokens and will take a
while to download and tokenize. Set NUM_PROC to your CPU core count to
parallelize tokenization.
"""

import os

import numpy as np
import tiktoken
from datasets import load_dataset

NUM_PROC = max(1, os.cpu_count() // 2)
OUT_DIR = os.path.join(os.path.dirname(__file__), "openwebtext")
enc = tiktoken.get_encoding("gpt2")


def tokenize(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset = load_dataset("Skylion007/openwebtext", num_proc=NUM_PROC)

    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    tokenized = split_dataset.map(
        tokenize,
        remove_columns=["text"],
        desc="tokenizing",
        num_proc=NUM_PROC,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(OUT_DIR, f"{split}.bin")
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))

        total_batches = 1024
        idx = 0
        for batch_idx in range(total_batches):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"wrote {filename} ({arr_len:,} tokens)")


if __name__ == "__main__":
    main()
