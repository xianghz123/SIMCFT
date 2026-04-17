# generate_simcft_dataset.py
# Convert split CSV files into SimCFT JSON datasets.

import os
import json
import argparse
import time
import ast
import glob
from typing import Optional

import numpy as np
import pandas as pd

from traj2grid import Traj2Grid
from parameters import (
    min_lon, min_lat, max_lon, max_lat,
    row_num as default_row_num,
    column_num as default_column_num,
    split_dir, grid_dir, train_dir, valid_dir, test_dir,
    train_csv_name, valid_csv_name, test_csv_name,
    trip_id_col, timestamp_col, polyline_col,
    sample_interval as default_sample_interval,
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


def load_grid2idx(grid2idx_path: str):
    """Load grid2idx JSON in the format {'(gx,gy)': idx}."""
    with open(grid2idx_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    grid2idx = {}
    for k, v in d.items():
        cell = ast.literal_eval(k)
        grid2idx[(int(cell[0]), int(cell[1]))] = int(v)

    vocab_size = (max(grid2idx.values()) + 1) if len(grid2idx) else 0
    return grid2idx, vocab_size


def try_load_meta(grid2idx_path: str, meta_path: Optional[str]):
    """Load the meta file generated together with grid2idx, if available."""
    if meta_path is None:
        cand = grid2idx_path.replace(".json", ".meta.json")
        meta_path = cand if os.path.exists(cand) else None
    if meta_path is None:
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def build_grid_lookup(grid2idx: dict, row_num: int, col_num: int):
    """Build lookup[gx, gy] = idx or -1."""
    lookup = np.full((col_num, row_num), -1, dtype=np.int32)
    for (gx, gy), idx in grid2idx.items():
        if 0 <= gx < col_num and 0 <= gy < row_num:
            lookup[gx, gy] = int(idx)
    return lookup


def grids_vectorized(lon: np.ndarray, lat: np.ndarray, t2g: Traj2Grid, lookup: np.ndarray):
    """Convert arrays of lon/lat to grid indices. Invalid points get -1."""
    lon = np.minimum(lon, t2g.max_lon - _EDGE_EPS)
    lat = np.minimum(lat, t2g.max_lat - _EDGE_EPS)

    row_num = t2g.row_num
    col_num = t2g.column_num
    h = (t2g.max_lat - t2g.min_lat) / row_num
    l = (t2g.max_lon - t2g.min_lon) / col_num

    gx = ((lon - t2g.min_lon) // l).astype(np.int32)
    gy = ((lat - t2g.min_lat) // h).astype(np.int32)

    valid = (gx >= 0) & (gx < col_num) & (gy >= 0) & (gy < row_num)
    out = np.full(lon.shape[0], -1, dtype=np.int32)

    idxs = np.where(valid)[0]
    if idxs.size > 0:
        out[idxs] = lookup[gx[idxs], gy[idxs]]

    return out


def safe_parse_polyline(polyline_str):
    """Parse a POLYLINE string into a list of [lon, lat]."""
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


def convert_one_trip(
    tid,
    start_ts,
    polyline_pts,
    t2g: Traj2Grid,
    lookup: np.ndarray,
    min_points: int,
    dedup_consecutive: bool,
    keep_id: bool,
    sample_interval: int,
):
    """
    Convert one row with:
      trip id, start timestamp, and POLYLINE
    into:
      {"points":[(lon,lat,t),...], "grids":[...], "tid":...}
    """
    if polyline_pts is None or len(polyline_pts) < min_points:
        return None

    arr = np.asarray(polyline_pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None

    lon = arr[:, 0]
    lat = arr[:, 1]

    tt = np.asarray(
        [int(start_ts) + i * int(sample_interval) for i in range(len(arr))],
        dtype=np.int64
    )

    grid_idx = grids_vectorized(lon, lat, t2g, lookup)
    keep = grid_idx >= 0
    if int(keep.sum()) < min_points:
        return None

    lon = lon[keep].tolist()
    lat = lat[keep].tolist()
    tt = tt[keep].tolist()
    grid_idx = grid_idx[keep].astype(int).tolist()

    if dedup_consecutive and len(grid_idx) > 1:
        new_lon, new_lat, new_t, new_g = [lon[0]], [lat[0]], [tt[0]], [grid_idx[0]]
        for i in range(1, len(grid_idx)):
            if grid_idx[i] != grid_idx[i - 1]:
                new_lon.append(lon[i])
                new_lat.append(lat[i])
                new_t.append(tt[i])
                new_g.append(grid_idx[i])
        lon, lat, tt, grid_idx = new_lon, new_lat, new_t, new_g

    if len(grid_idx) < min_points:
        return None

    sample = {
        "points": list(zip(lon, lat, tt)),
        "grids": grid_idx,
    }
    if keep_id:
        sample["tid"] = str(tid)

    return sample


def process_one_file(
    csv_path: str,
    out_json: str,
    t2g: Traj2Grid,
    lookup: np.ndarray,
    min_points: int,
    dedup_consecutive: bool,
    keep_id: bool,
    sample_interval: int,
    encoding: Optional[str] = "utf-8",
    log_every_rows: int = 100000,
):
    """Convert one split CSV file into a SimCFT JSON dataset."""
    file_timer = Timer()

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path, encoding=encoding)
    total_rows = len(df)
    samples = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        try:
            tid = getattr(row, trip_id_col)
            start_ts = int(getattr(row, timestamp_col))
            polyline_str = getattr(row, polyline_col)
        except AttributeError as e:
            raise KeyError(
                f"CSV missing required columns. "
                f"Expected: {trip_id_col}, {timestamp_col}, {polyline_col}"
            ) from e

        polyline_pts = safe_parse_polyline(polyline_str)
        sample = convert_one_trip(
            tid=tid,
            start_ts=start_ts,
            polyline_pts=polyline_pts,
            t2g=t2g,
            lookup=lookup,
            min_points=min_points,
            dedup_consecutive=dedup_consecutive,
            keep_id=keep_id,
            sample_interval=sample_interval,
        )

        if sample is not None:
            samples.append(sample)

        if log_every_rows > 0 and i % log_every_rows == 0:
            print(
                f"[{os.path.basename(csv_path)}] rows={i}/{total_rows} "
                f"saved_trajs={len(samples)} elapsed={Timer.fmt(file_timer.sec())}"
            )

    save_timer = Timer()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False)

    print(
        f"[Done] {os.path.basename(csv_path)} -> {out_json} "
        f"(trajs={len(samples)}) save_time={Timer.fmt(save_timer.sec())} "
        f"total_time={Timer.fmt(file_timer.sec())}"
    )

    return len(samples)


def main():
    ap = argparse.ArgumentParser(
        "Generate SimCFT JSON datasets (train/valid/test). "
        "Recommended: keep raw points and keep trajectory ids."
    )

    ap.add_argument("--grid2idx", type=str, default=None)
    ap.add_argument("--grid2idx_meta", type=str, default=None)

    ap.add_argument("--train_csv", type=str, default=os.path.join(split_dir, train_csv_name))
    ap.add_argument("--valid_csv", type=str, default=os.path.join(split_dir, valid_csv_name))
    ap.add_argument("--test_csv",  type=str, default=os.path.join(split_dir, test_csv_name))

    ap.add_argument("--out_train", type=str, default=os.path.join(train_dir, "simcft_train.json"))
    ap.add_argument("--out_valid", type=str, default=os.path.join(valid_dir, "simcft_valid.json"))
    ap.add_argument("--out_test",  type=str, default=os.path.join(test_dir, "simcft_test.json"))

    ap.add_argument("--row_num", type=int, default=default_row_num)
    ap.add_argument("--column_num", type=int, default=default_column_num)

    ap.add_argument("--min_points", type=int, default=5)
    ap.add_argument(
        "--dedup_consecutive",
        action="store_true",
        help="Remove consecutive duplicate grid ids. This is OFF by default for the standard protocol.",
    )
    ap.add_argument(
        "--no_keep_id",
        action="store_true",
        help="Do not save the original trajectory id",
    )
    ap.add_argument("--encoding", type=str, default="utf-8")
    ap.add_argument("--sample_interval", type=int, default=default_sample_interval)
    ap.add_argument("--log_every_rows", type=int, default=100000)
    ap.add_argument("--lower_bound", type=int, default=grid_lower_bound)

    args = ap.parse_args()

    if args.grid2idx is None:
        args.grid2idx = auto_find_grid2idx(
            grid_dir_=grid_dir,
            row_num=args.row_num,
            col_num=args.column_num,
            lower_bound=args.lower_bound,
        )

    keep_id = not args.no_keep_id
    total_timer = Timer()

    for p in [args.train_csv, args.valid_csv, args.test_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {p}")
    if not os.path.exists(args.grid2idx):
        raise FileNotFoundError(f"--grid2idx not found: {args.grid2idx}")

    print(f"[Info] grid2idx   : {args.grid2idx}")
    print(f"[Info] keep_id    : {keep_id}")
    print(f"[Info] dedup      : {args.dedup_consecutive}")
    print(f"[Info] interval   : {args.sample_interval}")
    print("[Info] Recommended protocol: keep raw points and do not deduplicate consecutive grids.")

    t = Timer()
    grid2idx, vocab_size = load_grid2idx(args.grid2idx)
    print(f"[Info] loaded grid2idx vocab_size={vocab_size} | time={Timer.fmt(t.sec())}")

    meta = try_load_meta(args.grid2idx, args.grid2idx_meta)
    if meta is not None:
        if int(meta["row_num"]) != int(args.row_num) or int(meta["column_num"]) != int(args.column_num):
            raise ValueError(
                f"row/col mismatch: args=({args.row_num},{args.column_num}) "
                f"meta=({meta['row_num']},{meta['column_num']})"
            )
        if (
            abs(float(meta["min_lon"]) - float(min_lon)) > 1e-9
            or abs(float(meta["max_lon"]) - float(max_lon)) > 1e-9
            or abs(float(meta["min_lat"]) - float(min_lat)) > 1e-9
            or abs(float(meta["max_lat"]) - float(max_lat)) > 1e-9
        ):
            print("[Warn] Bounds in the meta file do not match those in parameters.py. Please check consistency.")

    t2g = Traj2Grid(
        args.row_num,
        args.column_num,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        grid2idx,
        out_of_bound="drop"
    )

    lookup = build_grid_lookup(grid2idx, row_num=args.row_num, col_num=args.column_num)

    print("\n=== Generate training dataset ===")
    process_one_file(
        args.train_csv, args.out_train, t2g, lookup,
        min_points=args.min_points,
        dedup_consecutive=args.dedup_consecutive,
        keep_id=keep_id,
        sample_interval=args.sample_interval,
        encoding=args.encoding,
        log_every_rows=args.log_every_rows,
    )

    print("\n=== Generate validation dataset ===")
    process_one_file(
        args.valid_csv, args.out_valid, t2g, lookup,
        min_points=args.min_points,
        dedup_consecutive=args.dedup_consecutive,
        keep_id=keep_id,
        sample_interval=args.sample_interval,
        encoding=args.encoding,
        log_every_rows=args.log_every_rows,
    )

    print("\n=== Generate test dataset ===")
    process_one_file(
        args.test_csv, args.out_test, t2g, lookup,
        min_points=args.min_points,
        dedup_consecutive=args.dedup_consecutive,
        keep_id=keep_id,
        sample_interval=args.sample_interval,
        encoding=args.encoding,
        log_every_rows=args.log_every_rows,
    )

    print(f"\n[Done] all datasets generated. total_time={Timer.fmt(total_timer.sec())}")


if __name__ == "__main__":
    main()