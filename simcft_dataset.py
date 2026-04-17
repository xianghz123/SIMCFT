# simcft_dataset.py
# Shared dataset, collate, and odd-even view utilities for SimCFT.

import json
import numpy as np
import torch
from torch.utils.data import Dataset


class SimCFTDataset(Dataset):
    """
    Dataset for SimCFT JSON files.

    Each JSON file is expected to contain:
    {
        "samples": [
            {
                "points": [(lon, lat, time), ...],
                "grids": [g1, g2, ...]
            },
            ...
        ]
    }
    """

    def __init__(self, json_path, max_len=256, min_points=2):
        self.max_len = int(max_len)
        self.min_points = int(min_points)

        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if "samples" not in obj:
            raise KeyError(f"{json_path} missing key: 'samples'")

        self.data = obj["samples"]
        self.json_path = json_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if "points" not in item:
            raise KeyError(f"{self.json_path} sample idx={idx} missing key: 'points'")
        if "grids" not in item:
            raise KeyError(f"{self.json_path} sample idx={idx} missing key: 'grids'")

        pts = np.asarray(item["points"], dtype=np.float32)   # [L, 3]
        grids = np.asarray(item["grids"], dtype=np.int64)    # [L]

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"{self.json_path} sample idx={idx} points shape must be [L,3], got {pts.shape}"
            )

        if grids.ndim != 1:
            raise ValueError(
                f"{self.json_path} sample idx={idx} grids must be 1D, got {grids.shape}"
            )

        if len(pts) != len(grids):
            raise ValueError(
                f"{self.json_path} sample idx={idx} length mismatch: "
                f"len(points)={len(pts)} vs len(grids)={len(grids)}"
            )

        if len(pts) < self.min_points:
            raise ValueError(
                f"{self.json_path} sample idx={idx} too short: "
                f"len={len(pts)} < min_points={self.min_points}"
            )

        L = min(len(pts), self.max_len)
        return pts[:L], grids[:L], L


def pad_batch(batch, max_len, strict_grid_check=False):
    """
    Pad all trajectories in a batch to a fixed length max_len.

    Returns:
        points: [B, Lmax, 3]
        grids : [B, Lmax]
        lens  : [B]
        mask  : [B, Lmax], where True indicates padding
    """
    B = len(batch)
    Lmax = int(max_len)

    points = torch.zeros((B, Lmax, 3), dtype=torch.float32)
    grids = torch.zeros((B, Lmax), dtype=torch.long)   # padding grid id = 0
    mask = torch.ones((B, Lmax), dtype=torch.bool)     # True = padding
    lens = torch.zeros((B,), dtype=torch.long)

    for i, (pts, g, L) in enumerate(batch):
        if len(pts) != len(g):
            raise ValueError(
                f"Batch sample idx={i} length mismatch before padding: "
                f"len(points)={len(pts)} vs len(grids)={len(g)}"
            )

        L = min(int(L), Lmax)

        if strict_grid_check:
            gg = np.asarray(g[:L])
            if (gg < 0).any():
                bad = int(gg.min())
                raise ValueError(
                    f"Found negative grid id={bad} in batch sample idx={i} (valid part)."
                )

        points[i, :L, :] = torch.from_numpy(pts[:L])
        grids[i, :L] = torch.from_numpy(g[:L])
        mask[i, :L] = False
        lens[i] = L

    return points, grids, lens, mask


def odd_even_views(points, grids, lens, mask, max_len):
    """
    Split each trajectory into two non-overlapping views:
      - odd view : indices 0, 2, 4, ...
      - even view: indices 1, 3, 5, ...

    Since the original trajectory length is capped by max_len, each view only
    needs about half that length. Therefore, the padded view length is fixed to
    ceil(max_len / 2) to reduce unnecessary computation in downstream modules.

    Inputs:
        points: [B, L, 3]
        grids : [B, L]
        lens  : [B]
        mask  : [B, L], where True indicates padding

    Returns:
        ((p1, g1, l1, m1), (p2, g2, l2, m2))
    """
    B, L, _ = points.shape
    Lmax = int((max_len + 1) // 2)

    def select(idx_start: int):
        idx = torch.arange(idx_start, L, 2, device=points.device)
        pts = points.index_select(1, idx)
        g = grids.index_select(1, idx)
        m = mask.index_select(1, idx)

        ln = (~m).sum(dim=1).to(torch.long)
        L2 = pts.size(1)

        if L2 < Lmax:
            pad_n = Lmax - L2
            pts = torch.cat(
                [pts, torch.zeros((B, pad_n, 3), device=points.device)],
                dim=1
            )
            g = torch.cat(
                [g, torch.zeros((B, pad_n), dtype=torch.long, device=points.device)],
                dim=1
            )
            m = torch.cat(
                [m, torch.ones((B, pad_n), dtype=torch.bool, device=points.device)],
                dim=1
            )
        else:
            pts = pts[:, :Lmax]
            g = g[:, :Lmax]
            m = m[:, :Lmax]
            ln = ln.clamp_max(Lmax)

        return pts, g, ln, m

    view_odd = select(0)
    view_even = select(1)
    return view_odd, view_even