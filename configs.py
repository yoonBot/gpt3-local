"""
Model size configurations matching Table 2.1 of the GPT-3 paper
(Brown et al., 2020, "Language Models are Few-Shot Learners").

Context length in the paper is 2048 for every size. That's kept as the
default here, but you can shrink `block_size` to fit your GPU's VRAM.
"""

from dataclasses import dataclass


@dataclass
class GPT3Config:
    name: str = "gpt3-small"
    vocab_size: int = 50257       # GPT-2 BPE vocab
    block_size: int = 2048        # context length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True             # bias in Linear/LayerNorm, like GPT-2/GPT-3

    # GPT-3 alternates dense full-attention layers with locally banded
    # sparse-attention layers (similar to Sparse Transformer, Child et al. 2019).
    use_sparse_attention: bool = True
    local_attn_window: int = 256  # window size for sparse (odd-indexed) layers


# Official GPT-3 paper sizes (name -> n_layer, n_head, n_embd, params)
GPT3_CONFIGS = {
    # params  : ~125M
    "gpt3-small": dict(n_layer=12, n_head=12, n_embd=768),
    # params  : ~350M
    "gpt3-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    # params  : ~760M
    "gpt3-large": dict(n_layer=24, n_head=16, n_embd=1536),
    # params  : ~1.3B
    "gpt3-xl": dict(n_layer=24, n_head=24, n_embd=2048),
    # params  : ~2.7B
    "gpt3-2.7B": dict(n_layer=32, n_head=32, n_embd=2560),
    # params  : ~6.7B
    "gpt3-6.7B": dict(n_layer=32, n_head=32, n_embd=4096),
    # params  : ~13B
    "gpt3-13B": dict(n_layer=40, n_head=40, n_embd=5140),
    # params  : ~175B (defined for completeness; not trainable on a single
    # consumer GPU -- would need a large multi-node cluster)
    "gpt3-175B": dict(n_layer=96, n_head=96, n_embd=12288),
}

# Rough guidance for a single consumer GPU (8-24GB), fp16/bf16, batch size 1,
# with gradient checkpointing enabled. Actual usage also depends on
# block_size, batch size, and optimizer state (AdamW keeps 2 extra copies).
VRAM_GUIDANCE_GB = {
    "gpt3-small": "~4-6GB train, ~1GB inference",
    "gpt3-medium": "~8-10GB train, ~2GB inference",
    "gpt3-large": "~12-16GB train, ~3GB inference",
    "gpt3-xl": "~20-24GB train w/ checkpointing, ~6GB inference",
    "gpt3-2.7B": "needs >24GB train even with checkpointing; inference only on 24GB",
    "gpt3-6.7B": "multi-GPU or offloading required",
    "gpt3-13B": "multi-GPU or offloading required",
    "gpt3-175B": "not feasible outside a large cluster",
}


def get_config(name: str, **overrides) -> GPT3Config:
    if name not in GPT3_CONFIGS:
        raise ValueError(
            f"Unknown config '{name}'. Choose from: {list(GPT3_CONFIGS.keys())}"
        )
    cfg = GPT3Config(name=name, **GPT3_CONFIGS[name])
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"GPT3Config has no field '{k}'")
        setattr(cfg, k, v)
    return cfg
