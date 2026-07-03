"""
GPT-2 BPE tokenizer extended with two special tokens for calculator tool-use:

    <calc>  -- opens a literal arithmetic expression the model wants evaluated
    </calc> -- closes it

A model trained/fine-tuned with this tokenizer can emit e.g.
"<calc>47+89</calc>" during generation; generate_with_calc.py watches for
the closing tag, evaluates the expression for real (see calculator.py), and
splices the true result back into the token stream instead of letting the
model guess it.

vocab_size for this tokenizer is 50259 (GPT-2's 50257 + these 2 tokens),
vs. plain GPT-2 BPE's 50257 used elsewhere in this repo. A model must be
built/resized with the matching vocab_size -- see resize_embeddings.py to
adapt a checkpoint already trained with the base 50257 tokenizer.
"""

import tiktoken

CALC_OPEN = "<calc>"
CALC_CLOSE = "</calc>"

VOCAB_SIZE = 50257 + 2


def get_tool_tokenizer() -> tiktoken.Encoding:
    base = tiktoken.get_encoding("gpt2")
    special_tokens = dict(base._special_tokens)
    special_tokens[CALC_OPEN] = 50257
    special_tokens[CALC_CLOSE] = 50258
    return tiktoken.Encoding(
        name="gpt2_calc",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )
