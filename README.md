# gpt3-local

A from-scratch, locally trainable implementation of the GPT-3 architecture,
built by extending the public GPT-2 transformer design with the changes
described in the GPT-3 paper (Brown et al., 2020):

- model sizes from Table 2.1 of the paper (125M up to 175B, see `configs.py`)
- attention layers alternate between full dense causal attention and
  locally banded sparse attention (`model.py`), following the Sparse
  Transformer pattern GPT-3 uses
- GPT-2 BPE tokenizer (via `tiktoken`), learned positional embeddings,
  pre-norm transformer blocks, tied input/output embeddings — same as GPT-2

**This is not, and cannot be, the actual GPT-3.** OpenAI has not released
GPT-3's weights, training data, or exact hyperparameters. What you get here
is the published architecture, which you train yourself from scratch on
your own data and GPU. Treat it as a research/education tool for studying
GPT-3-style models at a scale your hardware can actually handle, not as a
way to reproduce OpenAI's model.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare data

Quick smoke test (a few MB, tokenizes in seconds):
```bash
python data/prepare_shakespeare.py
```

Real pretraining corpus (OpenWebText, ~9B tokens, requires real time/disk):
```bash
python data/prepare_openwebtext.py
```

Either script writes `train.bin` / `val.bin` (uint16 GPT-2 token ids) into
its own subdirectory under `data/`.

## 2. Train

```bash
python train.py --config gpt3-small --data_dir data/shakespeare --block_size 512 \
    --batch_size 12 --grad_accum_steps 4 --max_iters 5000
```

Multi-GPU (DDP) on one node:
```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --config gpt3-medium --data_dir data/openwebtext
```

Resume from the last checkpoint:
```bash
python train.py --config gpt3-small --data_dir data/shakespeare --resume
```

### Choosing a config for a single consumer GPU (8-24GB)

| config       | params | notes |
|--------------|-------:|-------|
| gpt3-small   | 125M   | comfortably fits on 8GB, fast iteration |
| gpt3-medium  | 350M   | fits on 12GB+ |
| gpt3-large   | 760M   | fits on 16-24GB, use `--gradient_checkpointing` |
| gpt3-xl      | 1.3B   | needs 24GB + `--gradient_checkpointing`, small batch size |
| 2.7B+        | —      | not realistic on a single consumer GPU; needs multi-GPU/offloading |

Key flags for fitting bigger configs into limited VRAM:
- `--gradient_checkpointing` — recompute activations in the backward pass
  instead of storing them (saves memory, costs ~20-30% more compute time)
- `--dtype bfloat16` (default) — halves activation memory vs float32
- lower `--batch_size` and raise `--grad_accum_steps` to keep the same
  effective batch size while using less memory per step
- lower `--block_size` (context length) — memory scales roughly with T²
  in the dense attention layers

`--no_sparse_attention` disables the alternating sparse layers and makes
every layer dense full attention (closer to plain GPT-2, simpler/faster per
step but higher memory at long context).

## 3. Sample / generate text

```bash
python sample.py --ckpt out/ckpt.pt --prompt "The meaning of life is" --max_new_tokens 200
```

## Files

- `configs.py` — `GPT3Config` dataclass and the official GPT-3 size table
- `model.py` — the transformer: attention (dense + locally banded sparse),
  MLP, blocks, and the full `GPT3` model with weight init, optimizer
  grouping, and a `generate()` method (top-k / top-p sampling)
- `train.py` — training loop: AMP mixed precision, gradient accumulation,
  gradient checkpointing, cosine LR schedule with warmup, DDP, checkpointing
- `sample.py` — load a checkpoint and generate text
- `data/prepare_shakespeare.py` — tiny smoke-test dataset
- `data/prepare_openwebtext.py` — full-scale open pretraining corpus
