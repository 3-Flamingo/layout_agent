"""Train the reward model on preference pairs.

Single GPU / CPU:
  python -m layout_agent.training.train \\
      --manifest data/evalsys/pairs_perturb/manifest.jsonl --out runs/reward_v0

8× A800 (recommended):
  torchrun --standalone --nproc_per_node=8 -m layout_agent.training.train \\
      --manifest data/evalsys/pairs_perturb/manifest.jsonl --out runs/reward_a800 \\
      --config src/layout_agent/training/configs/a800_8x.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from layout_agent.training.data import PairDataset
from layout_agent.training.losses import bt_loss
from layout_agent.training.model import RewardModel

DEFAULTS: Dict[str, Any] = {
    "img_size": 256,
    "channels": 3,
    "batch_size": 16,  # per-GPU
    "epochs": 60,
    "lr": 3.0e-4,  # base LR for 1 GPU; scaled by world_size when scale_lr=True
    "weight_decay": 1.0e-2,
    "freeze_stages": 2,
    "pretrained": False,
    "patience": 10,
    "num_workers": 0,
    "seed": 7,
    "amp": "fp16",  # off | fp16 | bf16  (bf16 preferred on A800)
    "scale_lr": True,  # linear LR scaling with world_size
    "tf32": True,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "find_unused_parameters": False,
}


def _collate(batch):
    pos = torch.from_numpy(np.stack([b["pos"] for b in batch]))
    neg = torch.from_numpy(np.stack([b["neg"] for b in batch]))
    levels = [b["level"] for b in batch]
    return pos, neg, levels


def _dist_info() -> tuple[bool, int, int, int]:
    """Return (distributed, rank, local_rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        return world > 1 or "LOCAL_RANK" in os.environ, rank, local, max(world, 1)
    return False, 0, 0, 1


def _setup_dist(local_rank: int) -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def _cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _amp_dtype(name: str) -> Optional[torch.dtype]:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


@torch.no_grad()
def validate(model, loader, device, amp_dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    for pos, neg, levels in loader:
        x = torch.cat([pos, neg]).to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp_dtype is not None, dtype=amp_dtype or torch.float16):
            s = _unwrap(model)(x)
        sp, sn = s[: len(pos)], s[len(pos) :]
        ok = (sp > sn).detach().float().cpu().numpy()
        for o, lv in zip(ok, levels):
            total["all"] += 1
            total[lv or "?"] += 1
            if o:
                correct["all"] += 1
                correct[lv or "?"] += 1
    return {k: round(correct[k] / total[k], 4) for k in total}


def _save_ckpt(path: Path, model: torch.nn.Module, cfg: Dict[str, Any], epoch: int) -> None:
    torch.save(
        {
            "model": _unwrap(model).state_dict(),
            "cfg": cfg,
            "epoch": epoch,
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="yaml overriding the built-in defaults")
    ap.add_argument("--device", default=None, help="cuda | cpu | cuda:N (ignored under torchrun)")
    for k, v in DEFAULTS.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(flag, type=type(v), default=None)
    args = ap.parse_args()

    distributed, rank, local_rank, world_size = _dist_info()
    if distributed:
        _setup_dist(local_rank)

    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {})
    for k in DEFAULTS:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    # device
    if distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.get("tf32") and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    seed = int(cfg["seed"]) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    out = Path(args.out)
    if _is_main(rank):
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            **cfg,
            "world_size": world_size,
            "global_batch_size": int(cfg["batch_size"]) * world_size,
            "device": str(device),
        }
        (out / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    manifest = Path(args.manifest)
    # cache only when single-process DataLoader (multi-worker duplicates RAM)
    use_cache = int(cfg["num_workers"]) == 0
    ds_tr = PairDataset(manifest, "train", cfg["img_size"], cfg["channels"], cache=use_cache)
    ds_va = PairDataset(manifest, "val", cfg["img_size"], cfg["channels"], cache=use_cache)

    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(ds_tr, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg["seed"])

    nw = int(cfg["num_workers"])
    loader_kw = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": nw,
        "collate_fn": _collate,
        "pin_memory": bool(cfg["pin_memory"]) and device.type == "cuda",
    }
    if nw > 0:
        loader_kw["persistent_workers"] = bool(cfg["persistent_workers"])
        loader_kw["prefetch_factor"] = int(cfg["prefetch_factor"])

    dl_tr = DataLoader(
        ds_tr,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=distributed and len(ds_tr) >= world_size * int(cfg["batch_size"]),
        **loader_kw,
    )
    # validate on rank 0 only (full val set, no sampler)
    dl_va = None
    if _is_main(rank) and len(ds_va):
        dl_va = DataLoader(ds_va, shuffle=False, **loader_kw)

    if _is_main(rank):
        print(
            f"train pairs: {len(ds_tr)}, val pairs: {len(ds_va)}, "
            f"device: {device}, world_size: {world_size}, "
            f"batch/gpu: {cfg['batch_size']}, global_batch: {int(cfg['batch_size']) * world_size}, "
            f"amp: {cfg['amp']}"
        )

    model = RewardModel(cfg["channels"], cfg["freeze_stages"], cfg["pretrained"]).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(cfg["find_unused_parameters"]),
        )

    base_lr = float(cfg["lr"])
    lr = base_lr * world_size if cfg.get("scale_lr", True) else base_lr
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(cfg["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(cfg["epochs"]))

    amp_dtype = _amp_dtype(str(cfg["amp"]).lower()) if device.type == "cuda" else None
    # GradScaler only needed for fp16; bf16 on Ampere does not need it
    use_scaler = amp_dtype is torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    best_acc = -1.0
    best_epoch = -1
    history = []
    epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.perf_counter()
        losses = []
        for pos, neg, _ in dl_tr:
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)
            x = torch.cat([pos, neg], dim=0)
            with torch.amp.autocast(device.type, enabled=amp_dtype is not None, dtype=amp_dtype or torch.float16):
                s = model(x)
                loss = bt_loss(s[: pos.size(0)], s[pos.size(0) :])
            optim.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            losses.append(float(loss.detach()))
        sched.step()

        # average train loss across ranks for logging
        mean_loss = sum(losses) / max(1, len(losses)) if losses else 0.0
        if distributed:
            t = torch.tensor([mean_loss, float(len(losses))], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mean_loss = float(t[0] / max(t[1], 1.0))

        val = {"all": 0.0}
        if _is_main(rank) and dl_va is not None:
            val = validate(model, dl_va, device, amp_dtype)

        dt = time.perf_counter() - t0
        if _is_main(rank):
            row = {
                "epoch": epoch,
                "loss": round(mean_loss, 4),
                "val": val,
                "lr": round(optim.param_groups[0]["lr"], 8),
                "sec": round(dt, 1),
            }
            history.append(row)
            print(f"epoch {epoch:3d}  loss {row['loss']:.4f}  val {val}  lr {row['lr']:.2e}  ({dt:.1f}s)")

            _save_ckpt(out / "last.pt", model, cfg, epoch)
            if val.get("all", 0.0) > best_acc:
                best_acc = val["all"]
                best_epoch = epoch
                _save_ckpt(out / "best.pt", model, cfg, epoch)
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        # early stop: broadcast decision from rank 0
        stop = False
        if _is_main(rank) and best_epoch > 0 and epoch - best_epoch >= patience:
            print(f"early stop at {epoch} (best val {best_acc:.4f} @ {best_epoch})")
            stop = True
        if distributed:
            flag = torch.tensor([1 if stop else 0], device=device)
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop:
            break

    if _is_main(rank):
        print(f"done. best val agreement: {best_acc:.4f} (epoch {best_epoch}) -> {out / 'best.pt'}")

    _cleanup_dist()


if __name__ == "__main__":
    main()
