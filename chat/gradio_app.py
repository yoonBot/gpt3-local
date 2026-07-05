"""
ChatGPT-style web chat UI for a gpt3-local checkpoint fine-tuned with
chat/prepare_chat_synthetic.py and/or chat/prepare_chat_alpaca.py
(vocab_size=50262). Optionally retrieves context from a rag/build_index.py
index before each reply, and always intercepts <calc>...</calc> tool calls
during generation via tools/generate_with_calc.py's shared generation loop.

    python chat/gradio_app.py --ckpt out_chat/ckpt.pt
    python chat/gradio_app.py --ckpt out_chat/ckpt.pt --rag_index_dir rag_index/
    python chat/gradio_app.py --ckpt out_chat/ckpt.pt --share   # public URL, needed on Colab
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch

from chat.format import format_prompt
from configs import GPT3Config
from model import GPT3
from tools.generate_with_calc import generate_with_tools
from tools.tokenizer import END_TURN, get_extended_tokenizer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--rag_index_dir", type=str, default=None, help="optional rag/build_index.py output dir")
    p.add_argument("--rag_top_k", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--max_history_turns", type=int, default=6, help="how many past turns to keep in context")
    p.add_argument("--share", action="store_true", help="get a public gradio.live URL (needed on Colab)")
    p.add_argument("--server_port", type=int, default=7860)
    return p.parse_args()


def history_to_pairs(history: list[dict]) -> list[tuple[str, str]]:
    """Convert Gradio's openai-style [{'role':..,'content':..}, ...] history
    into the (user_msg, assistant_msg) pairs chat/format.py expects."""
    pairs = []
    pending_user = None
    for msg in history:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant" and pending_user is not None:
            pairs.append((pending_user, msg["content"]))
            pending_user = None
    return pairs


def main():
    args = get_args()
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
            f"with chat/prepare_chat_*.py + tools/resize_embeddings.py?"
        )
    end_id = enc.encode(END_TURN, allowed_special="all")[0]

    retriever = None
    if args.rag_index_dir:
        from rag.retriever import Retriever
        retriever = Retriever(args.rag_index_dir)

    def respond(message: str, history: list[dict]) -> str:
        past_turns = history_to_pairs(history)[-args.max_history_turns:]

        context = None
        if retriever is not None:
            hits = retriever.search(message, k=args.rag_top_k)
            context = "\n\n".join(h["text"] for h in hits)

        prompt = format_prompt(past_turns, message, context=context)
        ids = enc.encode(prompt, allowed_special="all")
        if len(ids) > cfg.block_size:
            ids = ids[-cfg.block_size:]  # keep the tail (most recent turns) if it overflows
        idx = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

        out_tokens = generate_with_tools(
            model, enc, idx, args.max_new_tokens, args.temperature, args.top_k,
            device, stop_token_id=end_id,
        )
        reply_ids = out_tokens[len(ids):]
        if reply_ids and reply_ids[-1] == end_id:
            reply_ids = reply_ids[:-1]
        return enc.decode(reply_ids)

    demo = gr.ChatInterface(
        respond,
        title="gpt3-local chat",
        description=(
            "A from-scratch GPT-3-style model, fine-tuned for chat. "
            "Not the real GPT-3/ChatGPT -- see the repo README for what that means."
        ),
    )
    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
