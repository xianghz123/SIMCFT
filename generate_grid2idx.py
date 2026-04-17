# generate_grid2idx.py
# Build grid vocabulary from the training split.

import os
import json
import argparse
import numpy as np
import pandas as pd

from parameters import (
    min_lon, min_lat, max_lon, max_lat,
    split_dir, grid_dir, train_csv_name,
    row_num as default_row_num,
    column_num as default_column_num,
    grid_lower_bound,
    polyline_col,
)

_EDGE_EPS = 1e-12


class Timer:
    def __init__(self):
        import time
        self._time = time
        self.t0 = time.time()

    def reset(self):
        self.t0 = self._time.time()

    def sec(self):
        return self._time.time() - self.t0

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
    """
    Parse POLYLINE safely.
    Return a list of [lon, lat] or None.
    """
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


def main():
    ap = argparse.ArgumentParser("Generate grid2idx vocabulary from the training split")

    ap.add_argument(
        "--input_csv",
        type=str,
        default=os.path.join(split_dir, train_csv_name),
        help="Path to the training split CSV file",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=grid_dir,
        help="Output directory for grid2idx JSON files",
    )
    ap.add_argument("--row_num", type=int, default=default_row_num)
    ap.add_argument("--column_num", type=int, default=default_column_num)
    ap.add_argument(
        "--lower_bound",
        type=int,
        default=grid_lower_bound,
        help="Minimum point-count threshold for keeping a grid cell",
    )
    ap.add_argument("--chunksize", type=int, default=100000)
    ap.add_argument("--encoding", type=str, default="utf-8")
    ap.add_argument(
        "--sort_mode",
        type=str,
        default="grid",
        choices=["grid", "count_desc"],
        help="'grid' sorts by (gx, gy); 'count_desc' sorts by count descending and then by (gx, gy)",
    )

    args = ap.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    if args.row_num <= 0 or args.column_num <= 0:
        raise ValueError("--row_num and --column_num must be positive")

    if max_lat <= min_lat or max_lon <= min_lon:
        raise ValueError("Invalid geographic bounds in parameters.py")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Info] input_csv    : {args.input_csv}")
    print(f"[Info] out_dir      : {args.out_dir}")
    print(f"[Info] row_num      : {args.row_num}")
    print(f"[Info] column_num   : {args.column_num}")
    print(f"[Info] lower_bound  : {args.lower_bound}")
    print(f"[Info] sort_mode    : {args.sort_mode}")
    print("[Info] Recommended protocol: build the grid vocabulary from the training split only.")

    total_timer = Timer()

    row_num = int(args.row_num)
    col_num = int(args.column_num)

    h = (max_lat - min_lat) / row_num
    l = (max_lon - min_lon) / col_num

    if h <= 0 or l <= 0:
        raise ValueError("Computed grid size is non-positive")

    counts = np.zeros(row_num * col_num, dtype=np.int64)

    reader = pd.read_csv(
        args.input_csv,
        usecols=[polyline_col],
        chunksize=args.chunksize,
        encoding=args.encoding,
    )

    t = Timer()
    total_rows = 0
    total_points = 0
    kept_points = 0

    for chunk_idx, df in enumerate(reader):
        total_rows += len(df)

        for polyline_str in df[polyline_col]:
            pts = safe_parse_polyline(polyline_str)
            if pts is None or len(pts) == 0:
                continue

            arr = np.asarray(pts, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue

            lon = arr[:, 0]
            lat = arr[:, 1]
            total_points += len(lon)

            lon = np.minimum(lon, max_lon - _EDGE_EPS)
            lat = np.minimum(lat, max_lat - _EDGE_EPS)

            gx = ((lon - min_lon) // l).astype(np.int32)
            gy = ((lat - min_lat) // h).astype(np.int32)

            valid = (gx >= 0) & (gx < col_num) & (gy >= 0) & (gy < row_num)
            gx = gx[valid]
            gy = gy[valid]

            if gx.size == 0:
                continue

            kept_points += gx.size

            key = gx * row_num + gy
            counts += np.bincount(key, minlength=row_num * col_num)

        if chunk_idx % 10 == 0:
            print(
                f"[chunk {chunk_idx}] rows={total_rows} total_points={total_points} "
                f"kept_points={kept_points} elapsed={Timer.fmt(t.sec())}"
            )

    nonzero_idx = np.nonzero(counts)[0]
    value_counts = {}
    for k in nonzero_idx:
        gx = int(k // row_num)
        gy = int(k % row_num)
        value_counts[(gx, gy)] = int(counts[k])

    filtered = [(g, c) for g, c in value_counts.items() if c >= args.lower_bound]
    if len(filtered) == 0:
        raise RuntimeError(
            f"No grids remain after lower_bound={args.lower_bound}. "
            f"Try a smaller lower_bound or check the geographic bounds."
        )

    if args.sort_mode == "grid":
        filtered.sort(key=lambda x: (x[0][0], x[0][1]))
    else:
        filtered.sort(key=lambda x: (-x[1], x[0][0], x[0][1]))

    grid2idx = {g: i for i, (g, _c) in enumerate(filtered)}

    print(
        f"\n[build_vocab] remain {len(grid2idx)}/{len(value_counts)} grids, "
        f"filtered {round(100 - 100 * len(grid2idx) / max(1, len(value_counts)), 2)}%"
    )

    out_json = os.path.join(
        args.out_dir,
        f"str_grid2idx_{row_num}x{col_num}_{len(grid2idx)}_lb{args.lower_bound}.json"
    )
    str_grid2idx = {f"({g[0]},{g[1]})": int(idx) for g, idx in grid2idx.items()}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(str_grid2idx, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {out_json}")

    meta = {
        "input_csv": args.input_csv,
        "row_num": row_num,
        "column_num": col_num,
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
        "lower_bound": int(args.lower_bound),
        "sort_mode": args.sort_mode,
        "grid_size_h": float(h),
        "grid_size_l": float(l),
        "num_rows": int(total_rows),
        "num_total_points": int(total_points),
        "num_kept_points": int(kept_points),
        "num_all_nonzero_grids": int(len(value_counts)),
        "num_vocab_grids": int(len(grid2idx)),
        "polyline_col": polyline_col,
    }

    out_meta = out_json.replace(".json", ".meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {out_meta}")
    print(f"\n[Done] total_time={Timer.fmt(total_timer.sec())}")


if __name__ == "__main__":
    main()