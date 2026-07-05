"""
Quick smoke-test chat data: synthetic templates covering turn-taking
(greetings/small talk), arithmetic via the calculator tool, and a couple of
templated multi-turn exchanges -- enough to check the whole chat pipeline
(format, training, stopping at <|end|>, tool interception) works end to
end in minutes, on no more than a CPU.

This alone will NOT make the model broadly conversational or
knowledgeable -- it's a few hundred templated patterns, not real dialogue
data. For that, use chat/prepare_chat_alpaca.py's ~52K real instruction
examples (or bring your own dataset in the same <|user|>/<|assistant|>
format -- see chat/format.py).

Produces data/chat_synthetic/train.bin and val.bin, tokenized with the
extended tokenizer (vocab_size=50262).
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from chat.format import format_turn
from tools.calculator import format_result, safe_eval
from tools.tokenizer import get_extended_tokenizer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chat_synthetic")

GREETING_TEMPLATES = [
    ("Hi!", "Hello! How can I help you today?"),
    ("Hello", "Hi there! What can I do for you?"),
    ("Hey", "Hey! What's up?"),
    ("Good morning", "Good morning! How can I help?"),
    ("How are you?", "I'm doing well, thanks for asking! How can I help you?"),
    ("What's your name?", "I'm a small GPT-3-style model trained from scratch for research."),
    ("What can you do?", "I can chat, answer simple questions, and do arithmetic using a calculator tool."),
    ("Thanks!", "You're welcome!"),
    ("Thank you", "Happy to help!"),
    ("Bye", "Goodbye! Have a great day."),
    ("Goodnight", "Goodnight! Talk soon."),
]

ARITH_OPS = [
    ("+", "plus", -1000, 1000),
    ("-", "minus", -1000, 1000),
    ("*", "times", -100, 100),
    ("//", "divided by", 1, 100),
]

ARITH_TEMPLATES = [
    "What is {a} {op_word} {b}?",
    "Can you compute {a} {sym} {b}?",
    "{a} {sym} {b} = ?",
]


def gen_greeting(rng: random.Random) -> str:
    user_msg, assistant_msg = rng.choice(GREETING_TEMPLATES)
    return format_turn(user_msg, assistant_msg)


def gen_arith(rng: random.Random) -> str:
    sym, op_word, lo, hi = rng.choice(ARITH_OPS)
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    if sym == "//" and b == 0:
        b = 1
    expr = f"{a}{sym}{b}"
    result = format_result(safe_eval(expr))
    user_msg = rng.choice(ARITH_TEMPLATES).format(a=a, b=b, sym=sym, op_word=op_word)
    assistant_msg = f"<calc>{expr}</calc> = {result}"
    return format_turn(user_msg, assistant_msg)


def gen_example(rng: random.Random) -> str:
    # occasionally chain 2 turns so the model sees multi-turn context
    n_turns = rng.choice([1, 1, 1, 2])
    turns = []
    for _ in range(n_turns):
        turns.append(gen_arith(rng) if rng.random() < 0.6 else gen_greeting(rng))
    return "".join(turns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=100_000)
    p.add_argument("--val_fraction", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    rng = random.Random(args.seed)
    enc = get_extended_tokenizer()

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
