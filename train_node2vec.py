# train_node2vec.py
# Train Node2Vec embeddings on the cell transition graph.

import os
import json
import argparse
import random
import time
import glob
from collections import defaultdict

import numpy as np

from parameters import (
    dataset_name,
    grid_dir,
    model_dir,
    row_num as default_row_num,
    column_num as default_column_num,
    grid_lower_bound,
    embedding_dim as default_dim,
)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def reset(self):
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


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


def load_vocab_size_from_grid2idx(grid2idx_path: str) -> int:
    with open(grid2idx_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if len(d) == 0:
        raise ValueError(f"grid2idx is empty: {grid2idx_path}")
    max_idx = max(int(v) for v in d.values())
    return max_idx + 1


def build_adj_and_nbrset(edges, directed: bool):
    """
    Build adjacency list:
      adj[u] = [(v, w), ...]
    and neighbor set:
      nbr_set[u] = set(neighbors)
    """
    adj = defaultdict(list)
    nbr_set = defaultdict(set)

    for e in edges:
        u, v, w = int(e["src"]), int(e["dst"]), int(e["weight"])
        if w <= 0:
            continue

        adj[u].append((v, w))
        nbr_set[u].add(v)

        if not directed:
            adj[v].append((u, w))
            nbr_set[v].add(u)

    return adj, nbr_set


def alias_setup(probs: np.ndarray):
    K = len(probs)
    q = np.zeros(K, dtype=np.float64)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []

    for kk, p in enumerate(probs):
        q[kk] = K * p
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J: np.ndarray, q: np.ndarray) -> int:
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    return int(J[kk])


def preprocess_transition_probs(adj, nbr_set, p: float, q: float):
    """
    Precompute alias tables:
      alias_nodes[u] = (vs, J, q) for the first step
      alias_edges[(t,u)] = (vs, J, q) for the second-order transition
    """
    alias_nodes = {}
    for u, neis in adj.items():
        vs = np.array([x[0] for x in neis], dtype=np.int32)
        ws = np.array([x[1] for x in neis], dtype=np.float64)
        s = ws.sum()
        if s <= 0 or len(vs) == 0:
            continue
        probs = ws / s
        J, qq = alias_setup(probs)
        alias_nodes[u] = (vs, J, qq)

    alias_edges = {}
    p = max(float(p), 1e-12)
    q = max(float(q), 1e-12)

    for t in adj.keys():
        for u, _w in adj[t]:
            if u not in adj:
                continue

            neis = adj[u]
            vs = np.array([x[0] for x in neis], dtype=np.int32)
            ws = np.array([x[1] for x in neis], dtype=np.float64)

            bias = np.ones_like(ws, dtype=np.float64)
            t_nbrs = nbr_set.get(t, set())

            for i, x in enumerate(vs):
                if x == t:
                    bias[i] = 1.0 / p
                elif x in t_nbrs:
                    bias[i] = 1.0
                else:
                    bias[i] = 1.0 / q

            unnorm = ws * bias
            s = unnorm.sum()
            if s <= 0:
                continue

            probs = unnorm / s
            J, qq = alias_setup(probs)
            alias_edges[(t, u)] = (vs, J, qq)

    return alias_nodes, alias_edges


def node2vec_walk(adj, alias_nodes, alias_edges, walk_len: int, start: int):
    walk = [start]
    while len(walk) < walk_len:
        cur = walk[-1]
        if cur not in alias_nodes:
            break

        if len(walk) == 1:
            vs, J, qq = alias_nodes[cur]
            nxt = vs[alias_draw(J, qq)]
            walk.append(int(nxt))
        else:
            prev = walk[-2]
            key = (prev, cur)
            if key in alias_edges:
                vs, J, qq = alias_edges[key]
                nxt = vs[alias_draw(J, qq)]
                walk.append(int(nxt))
            else:
                vs, J, qq = alias_nodes[cur]
                nxt = vs[alias_draw(J, qq)]
                walk.append(int(nxt))
    return walk


class WalksCorpus:
    """
    Memory-efficient iterable over random walks for gensim Word2Vec.
    """

    def __init__(self, adj, alias_nodes, alias_edges,
                 vocab_size: int,
                 walk_len: int,
                 num_walks: int,
                 min_walk_len: int = 2,
                 log_every: int = 2000):
        self.adj = adj
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        self.vocab_size = int(vocab_size)
        self.walk_len = int(walk_len)
        self.num_walks = int(num_walks)
        self.min_walk_len = int(min_walk_len)
        self.log_every = int(log_every)

    def __iter__(self):
        t = Timer()
        total_nodes = self.vocab_size

        for node in range(total_nodes):
            if self.log_every > 0 and node > 0 and node % self.log_every == 0:
                done_walks = node * self.num_walks
                elapsed = t.sec()
                wps = done_walks / max(elapsed, 1e-6)
                eta = ((total_nodes * self.num_walks) - done_walks) / max(wps, 1e-6)
                print(
                    f"[Walks] node={node}/{total_nodes} walks={done_walks}/{total_nodes*self.num_walks} "
                    f"speed={wps:.1f} walks/s elapsed={Timer.fmt(elapsed)} ETA={Timer.fmt(eta)}"
                )

            if node not in self.alias_nodes:
                continue

            for _ in range(self.num_walks):
                w = node2vec_walk(
                    self.adj,
                    self.alias_nodes,
                    self.alias_edges,
                    self.walk_len,
                    start=node
                )
                if len(w) >= self.min_walk_len:
                    yield [str(x) for x in w]


def main():
    ap = argparse.ArgumentParser("Train Node2Vec on the cell transition graph")

    ap.add_argument(
        "--edges_json",
        type=str,
        default=os.path.join(grid_dir, "cell_edges.json"),
        help="Path to the cell_edges.json file generated by build_cell_graph.py",
    )
    ap.add_argument(
        "--grid2idx",
        type=str,
        default=None,
        help="Path to the grid2idx JSON file; if omitted, it will be auto-detected under grid_dir",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=model_dir,
        help="Directory for saving Node2Vec embeddings",
    )
    ap.add_argument("--out_name", type=str, default=f"cell_emb_{default_dim}.npy")

    ap.add_argument("--dim", type=int, default=default_dim)
    ap.add_argument("--walk_len", type=int, default=40)
    ap.add_argument("--num_walks", type=int, default=10)
    ap.add_argument("--min_walk_len", type=int, default=2)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--negative", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--directed", action="store_true", default=True)

    ap.add_argument("--max_nodes", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=2000)

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

    if not os.path.exists(args.edges_json):
        raise FileNotFoundError(f"--edges_json not found: {args.edges_json}")
    if not os.path.exists(args.grid2idx):
        raise FileNotFoundError(f"--grid2idx not found: {args.grid2idx}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)

    print(f"[Info] edges_json : {args.edges_json}")
    print(f"[Info] grid2idx   : {args.grid2idx}")
    print(f"[Info] out_dir    : {args.out_dir}")
    print(f"[Info] out_path   : {out_path}")
    print(f"[Info] directed   : {args.directed}")
    print("[Info] Recommended protocol: keep directed=True to match build_cell_graph.py")

    set_seed(args.seed)
    total_timer = Timer()

    t = Timer()
    vocab_size = load_vocab_size_from_grid2idx(args.grid2idx)
    if args.max_nodes is not None:
        vocab_size = min(vocab_size, int(args.max_nodes))
    print(f"[Info] vocab_size={vocab_size} | time={Timer.fmt(t.sec())}")

    t = Timer()
    with open(args.edges_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        edges = data.get("edges", [])
    elif isinstance(data, list):
        edges = data
    else:
        raise ValueError(f"Unsupported edges JSON format: {type(data)}")

    adj, nbr_set = build_adj_and_nbrset(edges, directed=args.directed)
    print(f"[Step] load graph: edges={len(edges)}, nodes_with_adj={len(adj)} | time={Timer.fmt(t.sec())}")

    t = Timer()
    alias_nodes, alias_edges = preprocess_transition_probs(adj, nbr_set, p=args.p, q=args.q)
    print(f"[Step] preprocess alias: alias_nodes={len(alias_nodes)}, alias_edges={len(alias_edges)} | time={Timer.fmt(t.sec())}")

    print("[Step] training Word2Vec with streaming walks...")
    try:
        from gensim.models import Word2Vec
    except Exception as e:
        raise RuntimeError("gensim import failed. Please install it with: pip install gensim") from e

    corpus = WalksCorpus(
        adj=adj,
        alias_nodes=alias_nodes,
        alias_edges=alias_edges,
        vocab_size=vocab_size,
        walk_len=args.walk_len,
        num_walks=args.num_walks,
        min_walk_len=args.min_walk_len,
        log_every=args.log_every,
    )

    t = Timer()
    model = Word2Vec(
        vector_size=args.dim,
        window=args.window,
        sg=1,
        hs=0,
        negative=args.negative,
        min_count=0,
        workers=args.workers,
        seed=args.seed,
    )
    model.build_vocab(corpus_iterable=corpus)
    model.train(corpus_iterable=corpus, total_examples=model.corpus_count, epochs=args.epochs)
    print(f"[Step] Word2Vec training done | time={Timer.fmt(t.sec())}")

    t = Timer()
    emb = np.zeros((vocab_size, args.dim), dtype=np.float32)
    miss = 0
    for i in range(vocab_size):
        key = str(i)
        if key in model.wv:
            emb[i] = model.wv[key]
        else:
            miss += 1
            emb[i] = np.random.normal(0, 0.01, size=(args.dim,)).astype(np.float32)

    np.save(out_path, emb)
    print(f"[Step] save emb: shape={emb.shape}, missing={miss} -> {out_path} | time={Timer.fmt(t.sec())}")

    meta = {
        "dataset": dataset_name,
        "vocab_size": int(vocab_size),
        "dim": int(args.dim),
        "walk_len": int(args.walk_len),
        "num_walks": int(args.num_walks),
        "min_walk_len": int(args.min_walk_len),
        "window": int(args.window),
        "epochs": int(args.epochs),
        "negative": int(args.negative),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "p": float(args.p),
        "q": float(args.q),
        "directed": bool(args.directed),
        "edges_json": args.edges_json,
        "grid2idx": args.grid2idx,
        "out_path": out_path,
        "nodes_with_adj": int(len(adj)),
        "alias_nodes": int(len(alias_nodes)),
        "alias_edges": int(len(alias_edges)),
    }

    meta_path = out_path.replace(".npy", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {meta_path}")
    print(f"\n[Done] total time = {Timer.fmt(total_timer.sec())}")


if __name__ == "__main__":
    main()