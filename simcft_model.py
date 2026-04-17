# simcft_model.py
# Paper-aligned SimCFT model for trajectory similarity learning.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in kilometers using the Haversine formula.

    Inputs are in degrees.
    """
    R = 6371.0
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a + 1e-12), torch.sqrt(1 - a + 1e-12))
    return R * c


def bearing_rad(lat1, lon1, lat2, lon2):
    """
    Initial great-circle bearing in radians.

    Inputs are in degrees.
    """
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    return torch.atan2(y, x)


def build_status_features(points, mask, lens, normalize_time=True):
    """
    Build status features for the status channel.

    Args:
        points: [B, L, 3] = (lon, lat, time)
        mask:   [B, L], where True indicates padding
        lens:   [B], valid trajectory lengths
        normalize_time: if True, use relative timestamps for numerical stability

    Returns:
        s: [B, L, 6] = (lon, lat, t, l, v, r)

    Notes:
        - normalize_time=True uses relative time (t_i - t_1).
        - Endpoint motion features are computed using each trajectory's true
          valid endpoints rather than padded positions.
    """
    lon = points[..., 0]
    lat = points[..., 1]
    t = points[..., 2]

    if normalize_time:
        t0 = t[:, :1]
        t = t - t0

    B, L = lon.shape
    device = points.device

    d = torch.zeros((B, L), device=device)
    if L >= 2:
        valid_pair = (~mask[:, :-1]) & (~mask[:, 1:])
        d_ = haversine_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        d[:, :-1] = torch.where(valid_pair, d_, torch.zeros_like(d_))

    l_move = torch.zeros((B, L), device=device)
    v = torch.zeros((B, L), device=device)
    r = torch.zeros((B, L), device=device)

    lens_t = torch.as_tensor(lens, device=device, dtype=torch.long).clamp_min(1)

    if L >= 3:
        l_move[:, 1:-1] = d[:, :-2] + d[:, 1:-1]
        denom_mid = (t[:, 2:] - t[:, :-2]).abs().clamp_min(EPS)
        v[:, 1:-1] = l_move[:, 1:-1] / denom_mid

    if L >= 2:
        br = bearing_rad(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        valid_pair = (~mask[:, :-1]) & (~mask[:, 1:])
        r[:, :-1] = torch.where(valid_pair, br, torch.zeros_like(br))

    for b in range(B):
        n = int(lens_t[b].item())

        if n <= 0:
            continue

        if n == 1:
            l_move[b, 0] = 0.0
            v[b, 0] = 0.0
            r[b, 0] = 0.0
            continue

        d_first = haversine_km(
            lat[b, 0], lon[b, 0],
            lat[b, 1], lon[b, 1]
        )
        l_move[b, 0] = d_first
        denom0 = (t[b, 1] - t[b, 0]).abs().clamp_min(EPS)
        v[b, 0] = d_first / denom0
        r[b, 0] = bearing_rad(
            lat[b, 0], lon[b, 0],
            lat[b, 1], lon[b, 1]
        )

        last = n - 1
        d_last = haversine_km(
            lat[b, last], lon[b, last],
            lat[b, last - 1], lon[b, last - 1]
        )
        l_move[b, last] = d_last
        denomN = (t[b, last] - t[b, last - 1]).abs().clamp_min(EPS)
        v[b, last] = d_last / denomN
        r[b, last] = bearing_rad(
            lat[b, last], lon[b, last],
            lat[b, last - 1], lon[b, last - 1]
        )

    s = torch.stack([lon, lat, t, l_move, v, r], dim=-1)
    s = torch.where(mask.unsqueeze(-1), torch.zeros_like(s), s)
    return s


class DGRU(nn.Module):
    """
    Dynamic Gated Recurrent Unit used in the status channel.

    Paper-aligned formulation:
      f_i = W_f s_i + b_f
      alpha_i = sigmoid(W_alpha f_i + b_alpha)
      s_i^e = s_i + alpha_i * f_i
      g_i = sigmoid(W_g s_i^e + b_g)

      h_i = (1 - z_i) * h_{i-1} + z_i * (g_i * h_tilde_i)
    """

    def __init__(self, in_dim=6, hid_dim=128):
        super().__init__()
        self.hid_dim = hid_dim

        self.Wf = nn.Linear(in_dim, in_dim, bias=True)
        self.Wa = nn.Linear(in_dim, 1, bias=True)

        self.Wg = nn.Linear(in_dim, hid_dim, bias=True)

        self.Wz = nn.Linear(in_dim, hid_dim, bias=True)
        self.Uz = nn.Linear(hid_dim, hid_dim, bias=False)

        self.Wr = nn.Linear(in_dim, hid_dim, bias=True)
        self.Ur = nn.Linear(hid_dim, hid_dim, bias=False)

        self.Wh = nn.Linear(in_dim, hid_dim, bias=True)
        self.Uh = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, s, lens, mask):
        B, L, _ = s.shape
        device = s.device

        h = torch.zeros((B, self.hid_dim), device=device)
        hs = torch.zeros((B, L, self.hid_dim), device=device)

        for i in range(L):
            si = s[:, i, :]
            mi = mask[:, i]

            fi = self.Wf(si)
            alpha = torch.sigmoid(self.Wa(fi))
            se = si + alpha * fi

            gi = torch.sigmoid(self.Wg(se))

            zi = torch.sigmoid(self.Wz(se) + self.Uz(h))
            ri = torch.sigmoid(self.Wr(se) + self.Ur(h))
            h_tilde = torch.tanh(self.Wh(se) + self.Uh(ri * h))

            h_new = (1 - zi) * h + zi * (gi * h_tilde)

            h = torch.where(mi.unsqueeze(-1), h, h_new)
            hs[:, i, :] = h

        lens_t = torch.as_tensor(lens, device=device, dtype=torch.long).clamp_min(1)
        idx = (lens_t - 1).view(B, 1, 1).expand(B, 1, self.hid_dim)
        h_last = hs.gather(1, idx).squeeze(1)
        return F.normalize(h_last, p=2, dim=-1)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]


class NoiseRecognition(nn.Module):
    """
    Noise recognition layer used before Transformer encoding.
    """

    def __init__(self, d, max_len):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W1)

        self.w2 = nn.Parameter(torch.empty(d, 1))
        nn.init.xavier_uniform_(self.w2)

        self.B1 = nn.Parameter(torch.zeros(max_len, d))
        self.b2 = nn.Parameter(torch.zeros(max_len, 1))

    def forward(self, X, mask):
        """
        Args:
            X: [B, L, D]
            mask: [B, L], where True indicates padding
        """
        _, L, _ = X.shape
        H = torch.relu(X @ self.W1 + self.B1[:L, :].unsqueeze(0))
        w = torch.sigmoid(H @ self.w2 + self.b2[:L, :].unsqueeze(0))
        w = torch.where(mask.unsqueeze(-1), torch.zeros_like(w), w)
        return w * X, w


class TransNR(nn.Module):
    """
    Transformer with noise recognition for the spatial-context channel.

    Pipeline:
      Noise Recognition -> Positional Encoding -> Multi-Head Attention
      -> Feed-Forward Network -> Masked Average Pooling
    """

    def __init__(self, d=128, heads=8, ff=256, max_len=512):
        super().__init__()
        self.nr = NoiseRecognition(d, max_len=max_len)
        self.pe = PositionalEncoding(d, max_len=max_len)
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=heads, batch_first=True)
        self.ff1 = nn.Linear(d, ff, bias=True)
        self.ff2 = nn.Linear(ff, d, bias=True)

    def forward(self, X, mask, lens=None):
        Xp, _ = self.nr(X, mask)
        Xp = self.pe(Xp)

        A, _ = self.mha(Xp, Xp, Xp, key_padding_mask=mask, need_weights=False)
        Y = self.ff2(torch.relu(self.ff1(A)))

        valid = (~mask).float().unsqueeze(-1)
        Ysum = (Y * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        z_s = Ysum / denom
        return F.normalize(z_s, p=2, dim=-1)


class SimCFT(nn.Module):
    """
    Dual-channel trajectory encoder:
      - status channel: DGRU
      - spatial-context channel: TransNR
      - fusion: learnable lambda
    """

    def __init__(
        self,
        vocab_size,
        d=128,
        heads=8,
        max_len=512,
        cell_emb=None,
        normalize_time=True,
        strict_grid_check=False,
    ):
        """
        Args:
            vocab_size: size of the grid vocabulary
            d: embedding dimension
            heads: number of attention heads
            max_len: maximum trajectory length
            cell_emb: optional pretrained cell embedding matrix
            normalize_time: use relative timestamps for status features
            strict_grid_check: whether to check negative non-pad grid ids
        """
        super().__init__()
        self.lamb = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self.normalize_time = normalize_time
        self.strict_grid_check = strict_grid_check

        if cell_emb is not None:
            w = torch.as_tensor(cell_emb, dtype=torch.float32)
            if w.ndim != 2 or w.shape[1] != d:
                raise ValueError(
                    f"cell_emb shape must be [vocab_size, {d}], got {tuple(w.shape)}"
                )
            self.cell_emb = nn.Embedding.from_pretrained(w, freeze=False)
        else:
            self.cell_emb = nn.Embedding(vocab_size, d)

        self.dgru = DGRU(in_dim=6, hid_dim=d)
        self.transnr = TransNR(d=d, heads=heads, ff=2 * d, max_len=max_len)

    def forward(self, points, grids, lens, mask):
        if self.strict_grid_check:
            if mask is None:
                if not (grids >= 0).all():
                    raise ValueError("Found negative grid ids.")
            else:
                if not (grids[~mask] >= 0).all():
                    raise ValueError("Found negative grid ids in non-padding positions.")

        s = build_status_features(
            points=points,
            mask=mask,
            lens=lens,
            normalize_time=self.normalize_time,
        )
        z_t = self.dgru(s, lens, mask)

        X = self.cell_emb(grids)
        z_s = self.transnr(X, mask, lens)

        lam = torch.sigmoid(self.lamb)
        z = lam * z_s + (1.0 - lam) * z_t
        return F.normalize(z, p=2, dim=-1)