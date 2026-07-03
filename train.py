"""
Training script for the local GPT-3-style model.

Single GPU:
    python train.py --config gpt3-small --data_dir data/shakespeare

Multi-GPU (DDP), e.g. 4 GPUs on one node:
    torchrun --standalone --nproc_per_node=4 train.py --config gpt3-medium --data_dir data/openwebtext

Run `python train.py --help` for all options.
"""

import argparse
import dataclasses
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from configs import get_config, VRAM_GUIDANCE_GB, GPT3_CONFIGS, GPT3Config
from model import GPT3


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="gpt3-small", choices=list(GPT3_CONFIGS.keys()))
    p.add_argument("--data_dir", type=str, required=True, help="dir with train.bin/val.bin")
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--block_size", type=int, default=1024, help="context length (<= 2048 in the paper)")
    p.add_argument("--vocab_size", type=int, default=None, help="override vocab size, e.g. 50259 for tools/tokenizer.py")

    p.add_argument("--batch_size", type=int, default=8, help="micro-batch size per GPU")
    p.add_argument("--grad_accum_steps", type=int, default=8, help="gradient accumulation steps")
    p.add_argument("--max_iters", type=int, default=20000)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--lr_decay_iters", type=int, default=20000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no_sparse_attention", action="store_true", help="disable alternating sparse attention layers")
    p.add_argument("--local_attn_window", type=int, default=256)

    p.add_argument("--gradient_checkpointing", action="store_true", help="trade compute for memory")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--compile", action="store_true", help="torch.compile the model (PyTorch 2.x)")

    p.add_argument("--resume", action="store_true", help="resume an interrupted run (out_dir/ckpt.pt), keeps optimizer state/iter count")
    p.add_argument("--init_from_ckpt", type=str, default=None, help="fine-tune: load model weights only from this checkpoint, fresh optimizer/schedule/iter count")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def get_batch(data_dir, split, block_size, batch_size, device):
    # memmap re-created each call, per nanoGPT, to avoid a slow memory leak
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it, args):
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    if it > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(model, args, device, ctx):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(args.data_dir, split, args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = get_args()

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_world_size = 1
        master_process = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + ddp_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"config={args.config}  VRAM guidance: {VRAM_GUIDANCE_GB.get(args.config)}")

    device_type = "cuda" if "cuda" in device.type else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))

    start_iter = 0
    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    pending_optimizer_state = None

    if args.resume and os.path.exists(ckpt_path):
        # continuing an interrupted run: reuse its exact config, optimizer
        # state, and iteration count.
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        cfg = GPT3Config(**checkpoint["config"])
        model = GPT3(cfg)
        model.load_state_dict(checkpoint["model"])
        pending_optimizer_state = checkpoint["optimizer"]
        start_iter = checkpoint["iter"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        if master_process:
            print(f"resumed from {ckpt_path} at iter {start_iter}")
    elif args.init_from_ckpt:
        # fine-tuning from another checkpoint's weights (e.g. after
        # tools/resize_embeddings.py): fresh optimizer/schedule/iter count,
        # config comes from that checkpoint so shapes match the loaded weights.
        checkpoint = torch.load(args.init_from_ckpt, map_location=device, weights_only=True)
        cfg = GPT3Config(**checkpoint["config"])
        model = GPT3(cfg)
        model.load_state_dict(checkpoint["model"])
        if master_process:
            print(f"initialized weights from {args.init_from_ckpt} (fresh optimizer/iter)")
    else:
        cfg_overrides = dict(
            block_size=args.block_size,
            dropout=args.dropout,
            use_sparse_attention=not args.no_sparse_attention,
            local_attn_window=args.local_attn_window,
        )
        if args.vocab_size is not None:
            cfg_overrides["vocab_size"] = args.vocab_size
        cfg = get_config(args.config, **cfg_overrides)
        model = GPT3(cfg)

    # cfg.block_size may come from a resumed/init checkpoint rather than
    # this invocation's --block_size flag; keep get_batch/estimate_loss (which
    # read args.block_size) consistent with the model actually being trained.
    args.block_size = cfg.block_size

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.to(device)

    if master_process:
        print(f"model params (non-embedding): {model.num_params():,}")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (args.beta1, args.beta2), device_type
    )
    if pending_optimizer_state is not None:
        optimizer.load_state_dict(pending_optimizer_state)

    if args.compile:
        model = torch.compile(model)

    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module

    x, y = get_batch(args.data_dir, "train", args.block_size, args.batch_size, device)
    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        lr = get_lr(it, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % args.eval_interval == 0 and master_process:
            losses = estimate_loss(model, args, device, ctx)
            print(f"iter {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": it,
                    "best_val_loss": best_val_loss,
                    "config": dataclasses.asdict(cfg),
                }, ckpt_path)
                print(f"saved checkpoint to {ckpt_path}")

        for micro_step in range(args.grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            x, y = get_batch(args.data_dir, "train", args.block_size, args.batch_size, device)
            scaler.scale(loss).backward()

        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if it % args.log_interval == 0 and master_process:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {it}: loss {loss.item() * args.grad_accum_steps:.4f}, lr {lr:.2e}, {dt*1000/args.log_interval:.1f}ms/iter")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
