"""
GPT-3 style decoder-only transformer.

Architecturally this is GPT-2 (pre-norm transformer, learned positional
embeddings, GPT-2 BPE vocab) with the changes GPT-3's paper describes
relative to GPT-2:
  - layers alternate between dense (full causal) attention and locally
    banded sparse attention, following the Sparse Transformer
  - model is provided at the range of sizes from Table 2.1 of the paper
    (see configs.py), rather than only the four GPT-2 sizes

Weight initialization, tying of input/output embeddings, and the residual
scaling on output projections follow GPT-2/GPT-3 as described in the
respective papers and the original GPT-2 source.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from configs import GPT3Config


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, either full (dense) or locally
    banded sparse. Sparse attention restricts each query position to only
    attend to keys within `window` tokens behind it (plus itself), which is
    the "locally banded" pattern from the Sparse Transformer used by GPT-3
    for its odd-indexed layers.
    """

    def __init__(self, config: GPT3Config, sparse: bool):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.sparse = sparse
        self.window = config.local_attn_window

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")

        # Precompute a banded-causal mask for sparse layers. Boolean mask,
        # True = allowed to attend. Built lazily / cached per sequence length.
        self._sparse_mask_cache = {}

    def _get_sparse_mask(self, T: int, device, dtype):
        key = (T, device)
        if key in self._sparse_mask_cache:
            return self._sparse_mask_cache[key]
        idx = torch.arange(T, device=device)
        # allowed if 0 <= (query - key) < window  (causal + local band)
        diff = idx.view(T, 1) - idx.view(1, T)
        allowed = (diff >= 0) & (diff < self.window)
        mask = torch.zeros(T, T, device=device, dtype=dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        self._sparse_mask_cache[key] = mask
        return mask

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.sparse:
            attn_mask = self._get_sparse_mask(T, x.device, q.dtype)
            if self.flash:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att + attn_mask
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
        else:
            if self.flash:
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                causal_mask = torch.triu(
                    torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                )
                att = att.masked_fill(causal_mask, float("-inf"))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT3Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPT3Config, layer_idx: int):
        super().__init__()
        # Odd-indexed layers (1, 3, 5, ...) use locally banded sparse
        # attention; even-indexed layers use full dense attention. This
        # matches the alternating pattern described in the GPT-3 paper.
        sparse = config.use_sparse_attention and (layer_idx % 2 == 1)
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, sparse=sparse)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT3(nn.Module):
    def __init__(self, config: GPT3Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying, as in GPT-2/GPT-3
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        self.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"sequence length {t} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # named_parameters() dedupes tied weights (lm_head.weight == wte.weight)
        # by name automatically, so group by tensor shape instead of walking
        # modules: matrices (dim >= 2) get weight decay, vectors (biases,
        # LayerNorm gains) don't.
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        no_decay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                logits[mask] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
