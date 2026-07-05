"""
GPT-2 BPE tokenizer extended with special tokens for two things:

  - calculator tool-use:
      <calc>  -- opens a literal arithmetic expression the model wants evaluated
      </calc> -- closes it
    A model can emit "<calc>47+89</calc>" during generation; whoever's
    generating (generate_with_calc.py, chat/gradio_app.py) watches for the
    closing tag, evaluates the expression for real (calculator.py), and
    splices the true result back into the token stream instead of letting
    the model guess it.

  - chat turn-taking:
      <|user|>      -- opens a user message
      <|assistant|> -- opens the assistant's reply
      <|end|>       -- closes either one
    A generation loop stops at <|end|> instead of rambling on, and the chat
    template (chat/format.py) uses these to mark whose turn is whose.

Both live in one shared vocab (rather than two separate extended
tokenizers) so a single fine-tuned checkpoint can use the calculator from
inside a chat conversation.

vocab_size for this tokenizer is 50262 (GPT-2's 50257 + these 5 tokens), vs.
plain GPT-2 BPE's 50257 used elsewhere in this repo. A model must be
built/resized with the matching vocab_size -- see resize_embeddings.py to
adapt a checkpoint already trained with the base 50257 tokenizer.
"""

import tiktoken

CALC_OPEN = "<calc>"
CALC_CLOSE = "</calc>"
USER_TURN = "<|user|>"
ASSISTANT_TURN = "<|assistant|>"
END_TURN = "<|end|>"

_NEW_SPECIAL_TOKENS = [CALC_OPEN, CALC_CLOSE, USER_TURN, ASSISTANT_TURN, END_TURN]
VOCAB_SIZE = 50257 + len(_NEW_SPECIAL_TOKENS)


def get_extended_tokenizer() -> tiktoken.Encoding:
    base = tiktoken.get_encoding("gpt2")
    special_tokens = dict(base._special_tokens)
    for i, tok in enumerate(_NEW_SPECIAL_TOKENS):
        special_tokens[tok] = 50257 + i
    return tiktoken.Encoding(
        name="gpt2_extended",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )


def check_vocab_or_raise(ckpt_vocab_size: int, ckpt_path: str) -> None:
    """New special tokens can be (and have been) added to this tokenizer
    over time -- e.g. the chat turn tokens were added after the calculator
    tokens, bumping VOCAB_SIZE from 50259 to 50262. A checkpoint saved
    before such a bump still has its old, smaller vocab_size baked into its
    config, which then mismatches a freshly pulled copy of this repo. That
    isn't a broken checkpoint, just a stale one -- resize_embeddings.py
    already supports incrementally growing any vocab_size up to the
    current VOCAB_SIZE, since new tokens are always appended at the end
    rather than inserted in the middle (so nothing already learned shifts
    position). Give a specific fix instead of a bare mismatch error.
    """
    if ckpt_vocab_size == VOCAB_SIZE:
        return
    if ckpt_vocab_size < VOCAB_SIZE:
        raise ValueError(
            f"{ckpt_path} has vocab_size={ckpt_vocab_size}, but this repo's tokenizer "
            f"is now {VOCAB_SIZE} (new special tokens were added since this checkpoint "
            f"was made -- e.g. by a `git pull`/re-clone after training). This is fixable: "
            f"upgrade the checkpoint in place with\n"
            f"    python tools/resize_embeddings.py --in_ckpt {ckpt_path} --out_ckpt {ckpt_path}\n"
            f"(safe to run again even if you've resized this checkpoint before -- it copies "
            f"every existing row unchanged and only adds fresh rows for the new tokens), "
            f"then retry."
        )
    raise ValueError(
        f"{ckpt_path} has vocab_size={ckpt_vocab_size}, larger than this repo's current "
        f"tokenizer vocab_size={VOCAB_SIZE}. This isn't fixable by resizing -- check you're "
        f"pointing at the right checkpoint and that your local repo is up to date."
    )
