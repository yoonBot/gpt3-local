# gpt3-local

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoonBot/gpt3-local/blob/main/gpt3_local_colab.ipynb)

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

### No local GPU? Run on Colab

Click the badge above, or open [gpt3_local_colab.ipynb](gpt3_local_colab.ipynb) directly in
[Colab](https://colab.research.google.com/github/yoonBot/gpt3-local/blob/main/gpt3_local_colab.ipynb).
It clones this repo, installs dependencies, and walks through data prep, training, sampling, RAG,
and calculator tool-use, cell by cell. Free-tier T4 GPUs (16GB) comfortably fit `gpt3-small`; mount
Google Drive (a cell in the notebook does this) so checkpoints survive a session disconnect and you
can `--resume` — Colab wipes local disk and disconnects after ~90 min idle / ~12h max.

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

### Progress bar and live loss plot

Training shows a `tqdm` progress bar (current iter, loss, lr, it/s) and tracks a
train/val loss line plot (`plotting.py`), redrawn every `--plot_interval` iters
(default: same as `--log_interval`). Where the plot renders depends on how you
run it:
- **In a notebook** (Colab, Jupyter), if `train.py` is imported and run
  in-process (`import train; train.main()` — see `gpt3_local_colab.ipynb`),
  the chart updates live, in place, in the cell output.
- **As a plain script** (`python train.py ...`, including via `!python train.py`
  in a notebook), there's no notebook frontend to draw into, so it instead
  (re)writes `{out_dir}/loss_curve.png` on every update — open that file to
  watch progress.

Pass `--no_plot` to skip this entirely.

## 3. Sample / generate text

```bash
python sample.py --ckpt out/ckpt.pt --prompt "The meaning of life is" --max_new_tokens 200
```

## 4. Retrieval-augmented generation (RAG)

The model's weights only ever store statistical patterns from next-token
prediction — they don't store a document store. RAG keeps your knowledge
in a separate, searchable index and injects the relevant text into the
prompt at inference time; the model's weights are untouched, no retraining
needed.

```bash
# 1. index a directory of .txt files
python rag/build_index.py --docs_dir mydocs/ --out_dir rag_index/

# 2. ask a question against that index
python rag/generate_with_rag.py --ckpt out/ckpt.pt --index_dir rag_index/ \
    --query "Where is the Eiffel Tower?"
```

`build_index.py` chunks each `.txt` file (`--chunk_tokens`, default 256, with
`--overlap_tokens` overlap) and embeds each chunk with a small sentence
embedding model (`all-MiniLM-L6-v2` by default). `generate_with_rag.py`
embeds your query, does a cosine-similarity nearest-neighbor search over
the index (plain numpy — fine for up to roughly hundreds of thousands of
chunks, no FAISS dependency needed at this scale), and feeds the top
matches into the model's context window ahead of the question.

## 5. Tool-use for math

Transformers don't have a calculator built into their weights — arithmetic
"ability" is pattern-matched from training examples and degrades fast on
larger numbers. Instead of hoping the model gets math right, you can teach
it to *delegate*: emit `<calc>expr</calc>` around an expression it wants
computed, and a wrapper around generation intercepts that span, evaluates
it for real (no `eval()`/`exec()` — `tools/calculator.py` walks a Python AST
with a whitelist of numeric operators only), and splices the true result
back into the output.

This needs a model whose vocab includes the extended tokens
(`vocab_size=50262` vs. the base 50257 — see `tools/tokenizer.py`) and
that's seen examples of the behavior during training:

```bash
# fine-tune from scratch with the extended vocab:
python tools/prepare_calc_data.py --n_examples 200000
python train.py --config gpt3-small --vocab_size 50262 --data_dir data/calc \
    --out_dir out_calc --max_iters 5000

# OR adapt a checkpoint you already pretrained on general text:
python tools/resize_embeddings.py --in_ckpt out/ckpt.pt --out_ckpt out/ckpt_tool.pt
python train.py --config gpt3-small --vocab_size 50262 --data_dir data/calc \
    --out_dir out_calc --init_from_ckpt out/ckpt_tool.pt --max_iters 5000

# generate with the calculator wired in:
python tools/generate_with_calc.py --ckpt out_calc/ckpt.pt \
    --prompt $'Q: What is 47 + 89?\nA:'
```

`--init_from_ckpt` loads model weights only (fresh optimizer/iter count) —
use it for fine-tuning from an existing checkpoint, as opposed to `--resume`
which continues an interrupted run of the *same* training job.

## 6. Chat interface

A raw next-token-prediction model doesn't inherently know how to hold a
conversation — that turn-taking behavior needs to be taught with
chat-formatted fine-tuning data, same as the calculator tool-use above (in
fact it shares the same extended tokenizer and vocab, so one fine-tuned
checkpoint can chat *and* use the calculator mid-conversation).

```bash
# 1. get chat data — either or both:
python chat/prepare_chat_synthetic.py --n_examples 100000   # quick smoke test: greetings + arithmetic, no download
python chat/prepare_chat_alpaca.py                            # real: ~52K Stanford Alpaca instruction examples

# 2. fine-tune. Two starting points, pick one:

# (a) haven't done tool-use fine-tuning yet -- resize the base checkpoint first:
python tools/resize_embeddings.py --in_ckpt out/ckpt.pt --out_ckpt out/ckpt_tool.pt
python train.py --config gpt3-small --vocab_size 50262 --data_dir data/chat_alpaca \
    --out_dir out_chat --init_from_ckpt out/ckpt_tool.pt --max_iters 5000

# (b) already have a calculator-tuned checkpoint (out_calc/ckpt.pt) -- build on it directly,
#     no resize needed (it already has the extended vocab), so the chat model keeps the
#     calculator behavior instead of re-learning it from a second cold fine-tune:
python train.py --config gpt3-small --vocab_size 50262 --data_dir data/chat_alpaca \
    --out_dir out_chat --init_from_ckpt out_calc/ckpt.pt --max_iters 5000

# 3. launch the web chat UI:
python chat/gradio_app.py --ckpt out_chat/ckpt.pt
# with retrieval-augmented answers from a rag/build_index.py index:
python chat/gradio_app.py --ckpt out_chat/ckpt.pt --rag_index_dir rag_index/
# on Colab (no localhost access), get a public link instead:
python chat/gradio_app.py --ckpt out_chat/ckpt.pt --share
```

`chat/format.py` defines the shared `<|user|>`/`<|assistant|>`/`<|end|>`
template used by both the data prep scripts and `chat/gradio_app.py`, so
they always agree on the exact same conversation format. Generation stops
at `<|end|>` (via `tools/generate_with_calc.py`'s shared `generate_with_tools`
loop, which also still intercepts `<calc>` tool calls mid-reply) instead of
running to `--max_new_tokens` on every message.

Caveat worth restating: `chat/prepare_chat_synthetic.py` alone only teaches
turn-taking format and calculator delegation — a few hundred templated
patterns, not real dialogue variety. For a model that's actually broadly
conversational, use the Alpaca data (or your own dataset in the same
format).

## Files

- `configs.py` — `GPT3Config` dataclass and the official GPT-3 size table
- `model.py` — the transformer: attention (dense + locally banded sparse),
  MLP, blocks, and the full `GPT3` model with weight init, optimizer
  grouping, and a `generate()` method (top-k / top-p sampling)
- `train.py` — training loop: AMP mixed precision, gradient accumulation,
  gradient checkpointing, cosine LR schedule with warmup, DDP, checkpointing,
  resume, fine-tuning from another checkpoint (`--init_from_ckpt`), a `tqdm`
  progress bar, and a live/saved loss-curve plot
- `plotting.py` — the loss-curve plotter (live inline in a notebook, saved
  PNG otherwise)
- `sample.py` — load a checkpoint and generate text
- `data/prepare_shakespeare.py` — tiny smoke-test dataset
- `data/prepare_openwebtext.py` — full-scale open pretraining corpus
- `rag/chunk.py`, `rag/build_index.py`, `rag/retriever.py`,
  `rag/generate_with_rag.py` — retrieval-augmented generation
- `tools/tokenizer.py` — GPT-2 BPE extended with `<calc>`/`</calc>` and chat
  turn (`<|user|>`/`<|assistant|>`/`<|end|>`) special tokens
- `tools/calculator.py` — safe (no `eval()`) arithmetic expression evaluator
- `tools/prepare_calc_data.py` — synthetic arithmetic fine-tuning data generator
- `tools/resize_embeddings.py` — adapt a checkpoint's vocab to the extended tokenizer
- `tools/generate_with_calc.py` — generation loop that intercepts `<calc>` spans
  and (optionally) stops at a given token id; shared by the CLI script and
  `chat/gradio_app.py`
- `chat/format.py` — the shared `<|user|>`/`<|assistant|>`/`<|end|>` chat template
- `chat/prepare_chat_synthetic.py` — quick smoke-test chat data (greetings + arithmetic)
- `chat/prepare_chat_alpaca.py` — real ~52K-example instruction-tuning dataset
- `chat/gradio_app.py` — ChatGPT-style web chat UI, with optional RAG and
  calculator tool-use wired in
