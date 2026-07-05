"""
Generate text from a checkpoint fine-tuned on tools/prepare_calc_data.py,
intercepting <calc>expr</calc> spans as the model emits them: once a
</calc> tag closes, the expression between the tags is evaluated for real
(tools/calculator.py, no eval()/exec()) and the true result's tokens are
spliced into the sequence, instead of letting the model guess the answer
from its own (unreliable) weights.

    python tools/generate_with_calc.py --ckpt out/ckpt_calc.pt --prompt "Q: What is 47 + 89?\nA:"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from configs import GPT3Config
from model import GPT3
from tools.calculator import CalcError, format_result, safe_eval
from tools.tokenizer import CALC_CLOSE, CALC_OPEN, get_extended_tokenizer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


@torch.no_grad()
def generate_with_tools(model, enc, idx, max_new_tokens, temperature, top_k, device, stop_token_id=None):
    """Shared generation loop: intercepts <calc>...</calc> spans (evaluates
    them for real and splices in the true result), and optionally stops
    early at `stop_token_id` (used by chat/gradio_app.py to stop at <|end|>
    instead of running to max_new_tokens on every reply).
    """
    open_id = enc.encode(CALC_OPEN, allowed_special="all")[0]
    close_id = enc.encode(CALC_CLOSE, allowed_special="all")[0]

    tokens = idx[0].tolist()
    calc_start = None  # index into `tokens` right after the most recent <calc>

    steps = 0
    while steps < max_new_tokens:
        steps += 1
        cur = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
        cur = cur if cur.size(1) <= model.config.block_size else cur[:, -model.config.block_size:]
        logits, _ = model(cur)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_id)

        if stop_token_id is not None and next_id == stop_token_id:
            break

        if next_id == open_id:
            calc_start = len(tokens)  # expression begins after this token
        elif next_id == close_id and calc_start is not None:
            expr = enc.decode(tokens[calc_start:-1])  # exclude the </calc> token itself
            try:
                result = format_result(safe_eval(expr))
                injected = f" = {result}"
            except CalcError as e:
                injected = f" [calc error: {e}]"
            tokens.extend(enc.encode(injected, allowed_special="all"))
            calc_start = None

    return tokens


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

    enc = get_extended_tokenizer()
    if cfg.vocab_size != enc.n_vocab:
        raise ValueError(
            f"checkpoint vocab_size={cfg.vocab_size} doesn't match the extended "
            f"tokenizer's vocab_size={enc.n_vocab} -- was this checkpoint fine-tuned "
            f"with tools/prepare_calc_data.py + tools/resize_embeddings.py?"
        )

    ids = enc.encode(args.prompt, allowed_special="all")
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    out_tokens = generate_with_tools(
        model, enc, idx, args.max_new_tokens, args.temperature, args.top_k, device
    )
    print(enc.decode(out_tokens))


if __name__ == "__main__":
    main()
