# train_simcft.py
# Train the SimCFT model.

import os
import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from simcft_dataset import SimCFTDataset, pad_batch, odd_even_views
from simcft_model import SimCFT
from parameters import (
    train_dir, valid_dir, model_dir,
    embedding_dim, num_heads, max_len,
    learning_rate, temperature,
)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def sec(self) -> float:
        return time.time() - self.t0

    @staticmethod
    def fmt(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"
        if seconds < 3600:
            m = int(seconds // 60)
            s = seconds - 60 * m
            return f"{m}m{s:.1f}s"
        h = int(seconds // 3600)
        m = int((seconds - 3600 * h) // 60)
        s = seconds - 3600 * h - 60 * m
        return f"{h}h{m}m{s:.1f}s"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_ckpt(path, model, opt, epoch, best_score, scaler=None):
    ensure_dir(path)
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def info_nce_paper(z_a, z_b, tau=0.1, clamp_logits=50.0):
    """
    InfoNCE loss in the odd-even training setting:
      odd-view as anchors
      even-view in the same batch as positives/negatives
    """
    logits = (z_a @ z_b.t()) / tau
    if clamp_logits is not None:
        logits = logits.clamp(-float(clamp_logits), float(clamp_logits))
    logprob = torch.log_softmax(logits, dim=1)
    return -torch.diag(logprob).mean()


@torch.no_grad()
def eval_loss(model, loader, device, max_len, tau, clamp_logits, amp_enabled, amp_dtype, max_batches=None):
    model.eval()
    total = 0.0
    n = 0

    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    for bi, (points, grids, lens, mask) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        points = points.to(device, non_blocking=True)
        grids = grids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        (p1, g1, l1, m1), (p2, g2, l2, m2) = odd_even_views(points, grids, lens, mask, max_len)

        with torch.cuda.amp.autocast(enabled=(amp_enabled and device.type == "cuda"), dtype=dtype):
            z1 = model(p1, g1, l1, m1)
            z2 = model(p2, g2, l2, m2)
            loss = info_nce_paper(z1, z2, tau=tau, clamp_logits=clamp_logits)

        if torch.isfinite(loss):
            total += float(loss.item())
            n += 1

    model.train()
    return total / max(1, n)


def get_lambda_value(model):
    if not hasattr(model, "lamb"):
        return None
    return float(torch.sigmoid(model.lamb).detach().cpu())


def main():
    ap = argparse.ArgumentParser("Train SimCFT")

    ap.add_argument("--train_json", type=str, default=os.path.join(train_dir, "simcft_train.json"))
    ap.add_argument("--valid_json", type=str, default=os.path.join(valid_dir, "simcft_valid.json"))
    ap.add_argument("--cell_emb", type=str, default=os.path.join(model_dir, "cell_emb_128.npy"))

    ap.add_argument("--max_len", type=int, default=max_len)
    ap.add_argument("--dim", type=int, default=embedding_dim)
    ap.add_argument("--heads", type=int, default=num_heads)

    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=learning_rate)

    ap.add_argument("--tau", type=float, default=temperature)
    ap.add_argument("--clamp_logits", type=float, default=50.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--pin_memory", action="store_true", default=True)
    ap.add_argument("--no_pin_memory", action="store_true")

    ap.add_argument("--persistent_workers", action="store_true", default=True)
    ap.add_argument("--no_persistent_workers", action="store_true")

    ap.add_argument("--prefetch_factor", type=int, default=4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=200)

    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--normalize_time", action="store_true", default=True)
    ap.add_argument("--no_normalize_time", action="store_true")

    ap.add_argument("--strict_grid_check", action="store_true", default=False)

    ap.add_argument("--valid_every", type=int, default=1)
    ap.add_argument("--valid_max_batches", type=int, default=None)

    ap.add_argument("--save_best", type=str, default=os.path.join(model_dir, "simcft_best.pth"))
    ap.add_argument("--save_latest", type=str, default=os.path.join(model_dir, "simcft_latest.pth"))

    args = ap.parse_args()

    for p in [args.train_json, args.valid_json, args.cell_emb]:
        if p and not os.path.exists(p):
            raise FileNotFoundError(f"not found: {p}")

    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[Info] train device={device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_enabled = bool(args.amp) and (not args.no_amp) and (device.type == "cuda")
    normalize_time = bool(args.normalize_time) and (not args.no_normalize_time)

    if amp_enabled:
        print(f"[Info] AMP enabled (dtype={args.amp_dtype})")

    cell_emb = np.load(args.cell_emb).astype(np.float32)
    vocab_size = int(cell_emb.shape[0])
    emb_dim = int(cell_emb.shape[1])

    if emb_dim != args.dim:
        raise ValueError(f"cell_emb dim={emb_dim} != --dim={args.dim}")

    print(f"[Info] cell_emb loaded: shape={cell_emb.shape}, vocab_size={vocab_size}")

    model = SimCFT(
        vocab_size=vocab_size,
        d=args.dim,
        heads=args.heads,
        max_len=args.max_len,
        cell_emb=cell_emb,
        normalize_time=normalize_time,
        strict_grid_check=args.strict_grid_check,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and args.amp_dtype == "fp16"))

    train_ds = SimCFTDataset(args.train_json, max_len=args.max_len)
    valid_ds = SimCFTDataset(args.valid_json, max_len=args.max_len) if args.valid_json else None

    pin = bool(args.pin_memory) and (not args.no_pin_memory)
    persistent = bool(args.persistent_workers) and (not args.no_persistent_workers) and args.num_workers > 0
    prefetch = int(args.prefetch_factor) if args.num_workers > 0 else None

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        collate_fn=lambda b: pad_batch(b, args.max_len, strict_grid_check=args.strict_grid_check),
    )
    if prefetch is not None:
        loader_kwargs["prefetch_factor"] = prefetch

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )

    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch,
            shuffle=False,
            drop_last=False,
            **loader_kwargs
        )

    print(f"[Info] train={len(train_ds)} valid={len(valid_ds) if valid_ds else 0}")
    print(
        f"[Info] batch={args.batch} num_workers={args.num_workers} "
        f"pin_memory={pin} persistent_workers={persistent} prefetch_factor={prefetch}"
    )
    print(f"[Info] tau={args.tau} clamp_logits={args.clamp_logits} grad_clip={args.grad_clip}")
    print(f"[Info] normalize_time={normalize_time} strict_grid_check={args.strict_grid_check}")

    total_timer = Timer()
    global_step = 0
    skipped = 0
    best_valid = float("inf")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    for ep in range(args.epochs):
        ep_timer = Timer()
        model.train()

        running = 0.0
        steps = 0

        last_log_time = time.time()
        last_log_step = global_step

        for points, grids, lens, mask in train_loader:
            points = points.to(device, non_blocking=True)
            grids = grids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if not torch.isfinite(points).all():
                skipped += 1
                continue

            (p1, g1, l1, m1), (p2, g2, l2, m2) = odd_even_views(points, grids, lens, mask, args.max_len)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(amp_enabled and device.type == "cuda"), dtype=amp_dtype):
                z1 = model(p1, g1, l1, m1)
                z2 = model(p2, g2, l2, m2)
                loss = info_nce_paper(z1, z2, tau=args.tau, clamp_logits=args.clamp_logits)

            if not torch.isfinite(loss):
                skipped += 1
                continue

            if amp_enabled and args.amp_dtype == "fp16":
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                opt.step()

            running += float(loss.item())
            steps += 1
            global_step += 1

            if args.log_every > 0 and global_step % args.log_every == 0:
                now = time.time()
                dt = now - last_log_time
                ds = global_step - last_log_step
                sps = ds / max(dt, 1e-6)

                lam_val = get_lambda_value(model)
                avg_so_far = running / max(1, steps)

                print(
                    f"[Train] epoch={ep} step={global_step} avg_loss={avg_so_far:.6f} "
                    f"lambda={lam_val:.4f} ep_t={Timer.fmt(ep_timer.sec())} "
                    f"tot_t={Timer.fmt(total_timer.sec())} dt={Timer.fmt(dt)} "
                    f"sps={sps:.1f} skipped={skipped}"
                )

                last_log_time = now
                last_log_step = global_step

        train_loss = running / max(1, steps)
        lam_val = get_lambda_value(model)
        print(
            f"[EpochEnd] epoch={ep} train_loss={train_loss:.6f} "
            f"lambda={lam_val:.4f} time={Timer.fmt(ep_timer.sec())} skipped={skipped}"
        )

        save_ckpt(
            args.save_latest,
            model,
            opt,
            ep,
            best_score=-train_loss,
            scaler=scaler if (amp_enabled and args.amp_dtype == "fp16") else None
        )

        if valid_loader is not None and args.valid_every > 0 and (ep % args.valid_every == 0):
            vt = Timer()
            vloss = eval_loss(
                model, valid_loader, device, args.max_len, args.tau,
                clamp_logits=args.clamp_logits,
                amp_enabled=amp_enabled,
                amp_dtype=args.amp_dtype,
                max_batches=args.valid_max_batches
            )
            print(f"[Valid] epoch={ep} valid_loss={vloss:.6f} time={Timer.fmt(vt.sec())}")

            if vloss < best_valid:
                best_valid = vloss
                save_ckpt(
                    args.save_best,
                    model,
                    opt,
                    ep,
                    best_score=-best_valid,
                    scaler=scaler if (amp_enabled and args.amp_dtype == "fp16") else None
                )
                print(f"[Save] new best valid_loss={best_valid:.6f} -> {args.save_best}")
        else:
            if ep == 0:
                save_ckpt(
                    args.save_best,
                    model,
                    opt,
                    ep,
                    best_score=-train_loss,
                    scaler=scaler if (amp_enabled and args.amp_dtype == "fp16") else None
                )

    print(f"\n[Done] training finished. total_time={Timer.fmt(total_timer.sec())}")
    print(f"[Done] best checkpoint: {args.save_best}")
    print(f"[Done] latest checkpoint: {args.save_latest}")


if __name__ == "__main__":
    main()