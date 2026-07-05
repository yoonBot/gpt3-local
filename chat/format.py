"""
Chat prompt formatting shared by training data prep (chat/prepare_chat_*.py)
and inference (chat/gradio_app.py), so both sides agree on the exact same
template.

A turn looks like:
    <|user|>{message}<|end|><|assistant|>{reply}<|end|>

For inference, the prompt ends with an open "<|assistant|>" (no closing
<|end|>) for the model to complete; generation stops at <|end|> (see
tools/generate_with_calc.py's `stop_token_id` argument).
"""

from __future__ import annotations

from tools.tokenizer import ASSISTANT_TURN, END_TURN, USER_TURN


def format_turn(user_msg: str, assistant_msg: str) -> str:
    return f"{USER_TURN}{user_msg}{END_TURN}{ASSISTANT_TURN}{assistant_msg}{END_TURN}"


def format_prompt(history: list[tuple[str, str]], user_msg: str, context: str | None = None) -> str:
    """history: list of (past_user_msg, past_assistant_msg) pairs, oldest first.
    context: optional RAG-retrieved text, prepended inside the current user turn.
    """
    prompt = "".join(format_turn(u, a) for u, a in history)
    current = f"Context:\n{context}\n\nQuestion: {user_msg}" if context else user_msg
    prompt += f"{USER_TURN}{current}{END_TURN}{ASSISTANT_TURN}"
    return prompt
