# build_cell_graph.py
# Build the weighted cell transition graph from the training split.

import os
import json
import argparse
import ast
import time
import glob
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from parameters import (
    min_lon, min_lat, max_lon, max_lat,
    split_dir, grid_dir, train_csv_name,
    polyline_col,
    row_num as default_row_num,
    column_num as default_column_num,
    grid_lower_bound,
)

_EDGE_EPS = 1e-12


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def sec(self):
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


def safe_parse_polyline(polyline_str):
    if pd.isna(polyline_str):
        return None
    if not isinstance(polyline_str, str):
        polyline_str = str(polyline_str)

    polyline_str = polyline_str.strip()
    if polyline_str == "" or polyline_str == "[]":
        return None

    try:
        pts = json.loads(polyline_str)
    except Exception:
        return None

    if not isinstance(pts, list):
        return None

    return pts


def load_grid2idx(path: str):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    grid2idx = {}
    for k, v in d.items():
        cell = ast.literal_eval(k)
        grid2idx[(int(cell[0]), int(cell[1]))] = int(v)
    return grid2idx


def load_meta_if_exists(grid2idx_path: str, meta_path: Optional[str]):
    if meta_path is None:
        cand = grid2idx_path.replace(".json", ".meta.json")
        meta_path = cand if os.path.exists(cand) else None

    if meta_path is None:
        return None, None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    row_num = int(meta["row_num"])
    col_num = int(meta["column_num"])
    return meta, (row_num, col_num)


def build_lookup_table(grid2idx: dict, row_num: int, col_num: int) -> np.ndarray:
    lookup = np.full((col_num, row_num), -1, dtype=np.int32)
    for (gx, gy), idx in grid2idx.items():
        if 0 <= gx < col_num and 0 <= gy < row_num:
            lookup[gx, gy] = int(idx)
    return lookup


def polyline_to_grid_ids(pts, row_num: int, col_num: int, lookup: np.ndarray):
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0,), dtype=np.int32)

    lon = arr[:, 0]
    lat = arr[:, 1]

    lon = np.minimum(lon, max_lon - _EDGE_EPS)
    lat = np.minimum(lat, max_lat - _EDGE_EPS)

    h = (max_lat - min_lat) / row_num
    l = (max_lon - min_lon) / col_num

    gx = ((lon - min_lon) // l).astype(np.int32)
    gy = ((lat - min_lat) // h).astype(np.int32)

    valid = (gx >= 0) & (gx < col_num) & (gy >= 0) & (gy < row_num)
    if not np.any(valid):
        return np.empty((0,), dtype=np.int32)

    gx = gx[valid]
    gy = gy[valid]

    grid_ids = lookup[gx, gy]
    grid_ids = grid_ids[grid_ids >= 0]
    return grid_ids.astype(np.int32, copy=False)


def auto_find_grid2idx(grid_dir_: str, row_num: int, col_num: int, lower_bound: int) -> str:
    """
    Automatically find a grid2idx file like:
      str_grid2idx_{row_num}x{col_num}_*_lb{lower_bound}.json
    """
    pattern = os.path.join(
        grid_dir_,
        f"str_grid2idx_{row_num}x{col_num}_*_lb{lower_bound}.json"
    )
    cands = sorted(glob.glob(pattern))
    if len(cands) == 0:
        raise FileNotFoundError(
            f"No grid2idx JSON found under pattern: {pattern}\n"
            f"Please run generate_grid2idx.py first, or pass --grid2idx manually."
        )
    if len(cands) > 1:
        print("[Warn] Multiple grid2idx files found. Using the latest one:")
        for p in cands:
            print("   ", p)
    return cands[-1]


def main():
    ap = argparse.ArgumentParser("Build the weighted cell transition graph from the training split")

    ap.add_argument(
        "--input_csv",
        type=str,
        default=os.path.join(split_dir, train_csv_name),
        help="Path to the training split CSV file",
    )
    ap.add_argument(
        "--grid2idx",
        type=str,
        default=None,
        help="Path to the grid2idx JSON file; if omitted, it will be auto-detected under grid_dir",
    )
    ap.add_argument(
        "--grid2idx_meta",
        type=str,
        default=None,
        help="Optional path to the grid2idx meta JSON; if omitted, <grid2idx>.meta.json will be tried automatically",
    )
    ap.add_argument(
        "--out_edges",
        type=str,
        default=os.path.join(grid_dir, "cell_edges.json"),
        help="Output path for the edge JSON file",
    )

    ap.add_argument("--chunksize", type=int, default=100000)
    ap.add_argument("--encoding", type=str, default="utf-8")
    ap.add_argument(
        "--dedup_consecutive",
        action="store_true",
        help="Remove consecutive duplicate grid ids within each trajectory before counting transitions",
    )

    ap.add_argument(
        "--directed",
        action="store_true",
        default=True,
        help="Keep directed transitions only (recommended and default)",
    )

    ap.add_argument("--log_every", type=int, default=10)

    ap.add_argument("--row_num", type=int, default=default_row_num)
    ap.add_argument("--column_num", type=int, default=default_column_num)
    ap.add_argument("--lower_bound", type=int, default=grid_lower_bound)

    args = ap.parse_args()

    if args.grid2idx is None:
        args.grid2idx = auto_find_grid2idx(
            grid_dir_=grid_dir,
            row_num=args.row_num,
            col_num=args.column_num,
            lower_bound=args.lower_bound,
        )

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"--input_csv not found: {args.input_csv}")
    if not os.path.exists(args.grid2idx):
        raise FileNotFoundError(f"--grid2idx not found: {args.grid2idx}")

    out_dir = os.path.dirname(args.out_edges)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[Info] input_csv    : {args.input_csv}")
    print(f"[Info] grid2idx     : {args.grid2idx}")
    print(f"[Info] out_edges    : {args.out_edges}")
    print(f"[Info] directed     : {args.directed}")
    print(f"[Info] dedup        : {args.dedup_consecutive}")
    print("[Info] Recommended protocol: use the training split only and build a directed graph.")

    total_timer = Timer()

    grid2idx = load_grid2idx(args.grid2idx)
    _meta, rc = load_meta_if_exists(args.grid2idx, args.grid2idx_meta)

    if rc is not None:
        row_num, col_num = rc
    else:
        row_num, col_num = args.row_num, args.column_num
        print(f"[Warn] grid2idx meta not found. Fallback to row={row_num}, col={col_num}.")

    lookup = build_lookup_table(grid2idx, row_num=row_num, col_num=col_num)

    reader = pd.read_csv(
        args.input_csv,
        usecols=[polyline_col],
        chunksize=args.chunksize,
        encoding=args.encoding,
    )

    t = Timer()
    edge_counter = defaultdict(int)

    total_rows = 0
    valid_traj = 0
    total_edges = 0

    for chunk_idx, df in enumerate(reader):
        total_rows += len(df)

        for polyline_str in df[polyline_col]:
            pts = safe_parse_polyline(polyline_str)
            if pts is None or len(pts) < 2:
                continue

            grid_ids = polyline_to_grid_ids(
                pts, row_num=row_num, col_num=col_num, lookup=lookup
            )

            if len(grid_ids) < 2:
                continue

            if args.dedup_consecutive:
                keep = np.empty(len(grid_ids), dtype=bool)
                keep[0] = True
                keep[1:] = grid_ids[1:] != grid_ids[:-1]
                grid_ids = grid_ids[keep]
                if len(grid_ids) < 2:
                    continue

            valid_traj += 1

            for i in range(len(grid_ids) - 1):
                u = int(grid_ids[i])
                v = int(grid_ids[i + 1])

                if u == v:
                    continue

                edge_counter[(u, v)] += 1
                total_edges += 1

                if not args.directed:
                    edge_counter[(v, u)] += 1
                    total_edges += 1

        if args.log_every > 0 and chunk_idx % args.log_every == 0:
            print(
                f"[chunk {chunk_idx}] rows={total_rows} "
                f"valid_traj={valid_traj} raw_edges={total_edges} "
                f"unique_edges={len(edge_counter)} elapsed={Timer.fmt(t.sec())}"
            )

    edges = [
        {"src": int(u), "dst": int(v), "weight": int(w)}
        for (u, v), w in edge_counter.items()
    ]
    edges.sort(key=lambda x: (x["src"], x["dst"]))

    with open(args.out_edges, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {args.out_edges}")

    meta = {
        "input_csv": args.input_csv,
        "grid2idx": args.grid2idx,
        "grid2idx_meta": args.grid2idx_meta,
        "row_num": int(row_num),
        "column_num": int(col_num),
        "num_total_rows": int(total_rows),
        "num_valid_traj": int(valid_traj),
        "num_raw_edges": int(total_edges),
        "num_unique_edges": int(len(edge_counter)),
        "directed": bool(args.directed),
        "dedup_consecutive": bool(args.dedup_consecutive),
        "polyline_col": polyline_col,
    }

    meta_path = args.out_edges.replace(".json", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {meta_path}")
    print(f"\n[Done] total_time={Timer.fmt(total_timer.sec())}")


if __name__ == "__main__":
    main()