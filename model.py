# -*- coding: utf-8 -*-
"""
model.py
PMTA-Net v4.2 dual-path: 数据主路径 + 物理投影头。

- 宏观轨迹编码 ΔY(t)、三谱跨模态注意力、DSC 时序编码、时序油相谱池化 z_oil。
- y_data：双任务 decoder + 物理变换（主精度）。
- y_phys：动力学参数 + Arrhenius 温度因子 + 可微 ODE 积分（物理投影）。
- 训练以 L_data(y_data) 为主；L_physics_align / L_data_phys 弱耦合；推理 blend 融合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_io import ExperimentCurve, modality_lengths, TARGET_NAMES_DEFAULT
from kinetics import (
    TARGET_NAMES, R_GAS, empirical_transition_time, monotonicity_loss, dp_ode_loss,
    acid_ode_loss, coupling_rank_loss, dsc_anchor_loss,
    spectral_consistency_loss, initial_anchor_loss, arrhenius_order_loss,
    physics_align_loss, integrate_trajectory_physics, params_to_rates,
)


# --------------------------- 标准化器 ---------------------------

class Standardizer:
    def __init__(self, eps: float = 1e-8):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=float)
        self.mean_ = np.nanmean(x, axis=0).astype(np.float32)
        std = np.nanstd(x, axis=0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std[std < self.eps] = 1.0
        self.std_ = std.astype(np.float32)
        self.mean_ = np.nan_to_num(self.mean_, nan=0.0, posinf=0.0, neginf=0.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer 尚未 fit。")
        return ((np.asarray(x, dtype=float) - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer 尚未 fit。")
        return (np.asarray(x, dtype=float) * self.std_ + self.mean_).astype(np.float32)

    def state_dict(self) -> Dict:
        return {"mean": self.mean_, "std": self.std_, "eps": self.eps}

    def load_state_dict(self, d: Dict) -> "Standardizer":
        self.mean_ = np.asarray(d["mean"], dtype=np.float32)
        self.std_ = np.asarray(d["std"], dtype=np.float32)
        self.eps = float(d.get("eps", 1e-8))
        return self


def preprocess_spectrum_timeseries(arr: np.ndarray, srep: Optional[Dict]) -> np.ndarray:
    """(n_time, L) 原始谱段：可选沿波数求导、再相对 t=0 差谱/比谱。与训练样本构造一致。"""
    if srep is None:
        srep = {}
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return x
    order = int(srep.get("wavenumber_derivative_order", 0))
    if order >= 1 and x.shape[1] >= 2:
        for _ in range(order):
            gx = np.zeros_like(x, dtype=np.float32)
            for i in range(x.shape[0]):
                gx[i] = np.gradient(x[i].astype(np.float64)).astype(np.float32)
            x = gx
    if bool(srep.get("use_delta")) and x.shape[0] >= 1:
        mode = str(srep.get("delta_mode", "subtract")).lower()
        ref = x[0:1].astype(np.float32)
        if mode == "ratio":
            den = np.maximum(np.abs(ref), 1e-8)
            x = (x / den).astype(np.float32)
            x = np.clip(x, -30.0, 30.0)
        else:
            x = (x - ref).astype(np.float32)
    return x.astype(np.float32)


class SpectralNormalizer:
    """每个模态一个均值/方差，避免谱强度量纲差异支配训练。"""
    def __init__(self):
        self.stats: Dict[str, Tuple[float, float]] = {}

    def fit(self, experiments: Sequence[ExperimentCurve], indices: Sequence[int],
            spectral_repr: Optional[Dict] = None) -> "SpectralNormalizer":
        srep = spectral_repr or {}
        for m in ["ftir", "raman", "uv"]:
            vals = []
            for i in indices:
                arr = getattr(experiments[i], m)
                if arr is not None and arr.size:
                    proc = preprocess_spectrum_timeseries(arr.astype(np.float32), srep)
                    vals.append(proc.reshape(-1))
            if vals:
                x = np.concatenate(vals).astype(float)
                mu, sd = float(np.nanmean(x)), float(np.nanstd(x))
                if not np.isfinite(sd) or sd < 1e-8:
                    sd = 1.0
                self.stats[m] = (mu, sd)
            else:
                self.stats[m] = (0.0, 1.0)
        return self

    def transform(self, modality: str, x: np.ndarray) -> np.ndarray:
        mu, sd = self.stats.get(modality, (0.0, 1.0))
        return ((np.asarray(x, dtype=float) - mu) / sd).astype(np.float32)

    def state_dict(self) -> Dict:
        return {k: list(v) for k, v in self.stats.items()}

    def load_state_dict(self, d: Dict) -> "SpectralNormalizer":
        self.stats = {k: (float(v[0]), float(v[1])) for k, v in d.items()}
        return self


@dataclass
class DataArtifacts:
    target_names: List[str]
    modality_dims: Dict[str, int]
    dsc_dim: int
    init_scaler: Standardizer
    dsc_scaler: Standardizer
    target_scaler: Standardizer
    spectral_norm: SpectralNormalizer
    oil_vocab: List[str]
    max_time_day: float
    acid_sat: float
    train_latent_mean: Optional[torch.Tensor] = None
    train_latent_cov_inv: Optional[torch.Tensor] = None
    train_kernel_latents: Optional[torch.Tensor] = None
    train_kernel_k_inv: Optional[torch.Tensor] = None
    kernel_lengthscale: float = 1.0
    kernel_noise: float = 1e-3


# --------------------------- Dataset ---------------------------

class TrajectoryDataset(Dataset):
    def __init__(self, experiments: Sequence[ExperimentCurve], indices: Sequence[int], artifacts: DataArtifacts,
                 cfg: Dict, late_time_cutoff: Optional[float] = None, early_window_only: bool = False):
        self.experiments = list(experiments)
        self.indices = list(indices)
        self.artifacts = artifacts
        self.cfg = cfg
        self.late_time_cutoff = late_time_cutoff
        self.early_window_only = early_window_only

    def __len__(self):
        return len(self.indices)

    def _pad_modality(self, exp: ExperimentCurve, modality: str, length: int, keep_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(keep_mask.sum())
        if length <= 0:
            return np.zeros((n, 1), dtype=np.float32), np.zeros((n,), dtype=np.float32)
        arr = getattr(exp, modality)
        if arr is None or arr.ndim != 2:
            return np.zeros((n, length), dtype=np.float32), np.zeros((n,), dtype=np.float32)
        arr = arr[keep_mask]
        L = min(length, arr.shape[1])
        segment = arr[:, :L].astype(np.float32)
        segment = preprocess_spectrum_timeseries(segment, self.cfg.get("spectral_repr"))
        out = np.zeros((n, length), dtype=np.float32)
        out[:, : segment.shape[1]] = segment
        out = self.artifacts.spectral_norm.transform(modality, out)
        valid = np.ones((n,), dtype=np.float32)
        return out, valid

    def _dsc(self, exp: ExperimentCurve, keep_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(keep_mask.sum())
        d = self.artifacts.dsc_dim
        if d <= 0:
            return np.zeros((n, 0), dtype=np.float32), np.zeros((n,), dtype=np.float32)
        arr = exp.dsc_features
        if arr is None or arr.ndim != 2:
            return np.zeros((n, d), dtype=np.float32), np.zeros((n,), dtype=np.float32)
        arr = arr[keep_mask]
        out = np.zeros((n, d), dtype=np.float32)
        dd = min(d, arr.shape[1])
        out[:, :dd] = arr[:, :dd]
        out = self.artifacts.dsc_scaler.transform(out)
        return out.astype(np.float32), np.ones((n,), dtype=np.float32)

    def __getitem__(self, j):
        exp = self.experiments[self.indices[j]]
        t = exp.t_days.astype(np.float32)
        keep = np.ones_like(t, dtype=bool)
        if self.late_time_cutoff is not None:
            keep &= t <= float(self.late_time_cutoff)
        if self.early_window_only:
            keep &= t <= float(self.cfg.get("data", {}).get("early_window_days", 22.0))
        if keep.sum() < 2:
            keep = np.ones_like(t, dtype=bool)
        t = t[keep]
        y = exp.target_matrix(self.artifacts.target_names)[keep]
        y_std = self.artifacts.target_scaler.transform(y)
        init = exp.initial
        init_vec = np.array([
            init.acid0, init.resistivity0, init.loss_factor0, init.bdv0, init.dp0,
            0.0 if not np.isfinite(init.viscosity0) else init.viscosity0,
        ], dtype=np.float32)
        init_std = self.artifacts.init_scaler.transform(init_vec[None, :])[0]
        oil_id = self.artifacts.oil_vocab.index(exp.oil_type) if exp.oil_type in self.artifacts.oil_vocab else -1
        mods = {}
        mod_masks = {}
        for m in ["ftir", "raman", "uv"]:
            arr, vm = self._pad_modality(exp, m, self.artifacts.modality_dims.get(m, 0), keep)
            mods[m] = arr
            mod_masks[m] = vm
        dsc, dsc_mask = self._dsc(exp, keep)
        tc = empirical_transition_time(t, y)
        if t.max() > 0:
            tc_norm = np.clip(tc / max(self.artifacts.max_time_day, 1.0), 0.0, 1.0)
        else:
            tc_norm = 0.0
        y0 = y[0:1]
        delta = (y - y0).astype(np.float32)
        t_std = np.log1p(np.maximum(t, 0.0)) / max(np.log1p(self.artifacts.max_time_day), 1e-6)
        t_norm = t / max(float(self.artifacts.max_time_day), 1.0)
        std5 = self.artifacts.target_scaler.std_[: len(TARGET_NAMES)]
        std5 = np.where(std5 < 1e-8, 1.0, std5).astype(np.float32)
        delta_n = delta / std5
        macro_feat = np.concatenate(
            [delta_n, t_norm[:, None].astype(np.float32), t_std[:, None].astype(np.float32)],
            axis=1,
        ).astype(np.float32)
        return {
            "name": exp.name,
            "oil_type": exp.oil_type,
            "oil_id": oil_id,
            "T_C": np.float32(exp.temperature_C),
            "T_K": np.float32(exp.temperature_C + 273.15),
            "t_days": t.astype(np.float32),
            "init": init_vec.astype(np.float32),
            "init_std": init_std.astype(np.float32),
            "target": y.astype(np.float32),
            "target_std": y_std.astype(np.float32),
            "tc_target": np.float32(tc_norm),
            "ftir": mods["ftir"], "raman": mods["raman"], "uv": mods["uv"],
            "ftir_mask": mod_masks["ftir"], "raman_mask": mod_masks["raman"], "uv_mask": mod_masks["uv"],
            "dsc": dsc.astype(np.float32), "dsc_mask": dsc_mask.astype(np.float32),
            "macro_feat": macro_feat,
        }


def collate_trajectories(batch: List[Dict]) -> Dict:
    B = len(batch)
    max_len = max(len(b["t_days"]) for b in batch)
    names = [b["name"] for b in batch]
    oil_types = [b["oil_type"] for b in batch]
    def pad_2d(key, last_dim):
        out = torch.zeros(B, max_len, last_dim, dtype=torch.float32)
        for i, b in enumerate(batch):
            v = torch.as_tensor(b[key], dtype=torch.float32)
            out[i, :v.shape[0], :v.shape[1]] = v
            if v.shape[0] < max_len and v.shape[0] > 0:
                out[i, v.shape[0]:, :v.shape[1]] = v[-1:]
        return out
    def pad_1d(key):
        out = torch.zeros(B, max_len, dtype=torch.float32)
        for i, b in enumerate(batch):
            v = torch.as_tensor(b[key], dtype=torch.float32)
            out[i, :v.numel()] = v
            if v.numel() < max_len and v.numel() > 0:
                out[i, v.numel():] = v[-1]
        return out
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        mask[i, :len(b["t_days"])] = True
    d = {
        "name": names, "oil_type": oil_types,
        "oil_id": torch.tensor([b["oil_id"] for b in batch], dtype=torch.long),
        "T_C": torch.tensor([b["T_C"] for b in batch], dtype=torch.float32),
        "T_K": torch.tensor([b["T_K"] for b in batch], dtype=torch.float32),
        "t_days": pad_1d("t_days"),
        "mask": mask,
        "init": torch.stack([torch.as_tensor(b["init"], dtype=torch.float32) for b in batch]),
        "init_std": torch.stack([torch.as_tensor(b["init_std"], dtype=torch.float32) for b in batch]),
        "target": pad_2d("target", len(TARGET_NAMES)),
        "target_std": pad_2d("target_std", len(TARGET_NAMES)),
        "tc_target": torch.tensor([b["tc_target"] for b in batch], dtype=torch.float32),
        "dsc": pad_2d("dsc", batch[0]["dsc"].shape[1] if batch[0]["dsc"].ndim == 2 else 0),
        "dsc_mask": pad_1d("dsc_mask"),
        "macro_feat": pad_2d("macro_feat", batch[0]["macro_feat"].shape[1]),
    }
    for m in ["ftir", "raman", "uv"]:
        Lm = batch[0][m].shape[1]
        d[m] = pad_2d(m, Lm)
        d[f"{m}_mask"] = pad_1d(f"{m}_mask")
    return d


def build_artifacts(experiments: Sequence[ExperimentCurve], train_idx: Sequence[int], cfg: Dict, acid_sat: float) -> DataArtifacts:
    target_names = list(cfg.get("targets", TARGET_NAMES_DEFAULT))
    dims = modality_lengths(experiments, train_idx)
    init_mat, dsc_vals, targets = [], [], []
    for i in train_idx:
        e = experiments[i]
        init = e.initial
        init_mat.append([
            init.acid0, init.resistivity0, init.loss_factor0, init.bdv0, init.dp0,
            0.0 if not np.isfinite(init.viscosity0) else init.viscosity0,
        ])
        targets.append(e.target_matrix(target_names))
        if e.dsc_features is not None and e.dsc_features.size:
            d = e.dsc_features
            if dims.get("dsc", 0) > 0:
                tmp = np.zeros((d.shape[0], dims["dsc"]), dtype=np.float32)
                tmp[:, :min(dims["dsc"], d.shape[1])] = d[:, :min(dims["dsc"], d.shape[1])]
                dsc_vals.append(tmp)
    if not dsc_vals and dims.get("dsc", 0) > 0:
        dsc_vals.append(np.zeros((1, dims["dsc"]), dtype=np.float32))
    oil_vocab = sorted({experiments[i].oil_type for i in train_idx})
    max_time = max(float(np.max(experiments[i].t_days)) for i in train_idx)
    return DataArtifacts(
        target_names=target_names,
        modality_dims={k: int(dims.get(k, 0)) for k in ["ftir", "raman", "uv"]},
        dsc_dim=int(dims.get("dsc", 0)),
        init_scaler=Standardizer().fit(np.asarray(init_mat, dtype=np.float32)),
        dsc_scaler=Standardizer().fit(np.concatenate(dsc_vals, axis=0) if dsc_vals else np.zeros((1, 0), dtype=np.float32)),
        target_scaler=Standardizer().fit(np.concatenate(targets, axis=0)),
        spectral_norm=SpectralNormalizer().fit(experiments, train_idx, cfg.get("spectral_repr")),
        oil_vocab=oil_vocab,
        max_time_day=max(max_time, 1.0),
        acid_sat=float(acid_sat),
    )


# --------------------------- 网络模块 ---------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        groups = max(1, min(4, channels // 4))
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(groups, channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))



class SpectralAugmentation(nn.Module):
    """逐谱图光谱增强：噪声、基线漂移、强度缩放和轻微横向偏移。输入/输出均为 (B,L,N)。"""
    def __init__(self, noise_std: float = 0.0, baseline_scale: float = 0.0,
                 intensity_range=(1.0, 1.0), shift_max: int = 0):
        super().__init__()
        self.noise_std = float(noise_std)
        self.baseline_scale = float(baseline_scale)
        self.intensity_low = float(intensity_range[0])
        self.intensity_high = float(intensity_range[1])
        self.shift_max = int(shift_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or x.numel() == 0:
            return x
        B, L, N = x.shape
        y = x
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        if self.baseline_scale > 0 and N > 1:
            grid = torch.linspace(-1.0, 1.0, N, device=x.device, dtype=x.dtype).view(1, 1, N)
            coeff = torch.randn(B, L, 3, device=x.device, dtype=x.dtype) * self.baseline_scale
            baseline = (coeff[..., 0:1] * grid.pow(2) + coeff[..., 1:2] * grid + coeff[..., 2:3])
            y = y + baseline
        if abs(self.intensity_high - self.intensity_low) > 1e-12:
            scale = torch.rand(B, L, 1, device=x.device, dtype=x.dtype) * (self.intensity_high - self.intensity_low) + self.intensity_low
            y = y * scale
        if self.shift_max > 0 and N > 1:
            flat = y.reshape(B * L, N).clone()
            shifts = torch.randint(-self.shift_max, self.shift_max + 1, (B * L,), device=x.device)
            for i in range(B * L):
                sft = int(shifts[i].item())
                if sft == 0:
                    continue
                flat[i] = torch.roll(flat[i], shifts=sft, dims=-1)
                if sft > 0:
                    flat[i, :sft] = flat[i, sft]
                else:
                    flat[i, sft:] = flat[i, sft - 1]
            y = flat.view(B, L, N)
        return y


def _make_clamped_knots(num_basis: int, degree: int = 3, lo: float = -3.0, hi: float = 3.0,
                        device=None, dtype=None) -> torch.Tensor:
    n_internal = num_basis - degree + 1
    if n_internal < 2:
        n_internal = 2
    internal = torch.linspace(lo, hi, n_internal, device=device, dtype=dtype)
    return torch.cat([internal[:1].expand(degree), internal, internal[-1:].expand(degree)])


def cubic_bspline_basis(x: torch.Tensor, num_basis: int = 6, lo: float = -3.0, hi: float = 3.0) -> torch.Tensor:
    degree = 3
    orig_shape = x.shape
    x_flat = x.reshape(-1).clamp(lo, hi)
    N = x_flat.shape[0]
    knots = _make_clamped_knots(num_basis, degree, lo, hi, x.device, x.dtype)
    n_intervals = knots.shape[0] - 1
    eps = 1e-10
    basis = torch.zeros(N, n_intervals, device=x.device, dtype=x.dtype)
    for i in range(n_intervals):
        if i == n_intervals - 1:
            m = (knots[i] <= x_flat) & (x_flat <= knots[i + 1])
        else:
            m = (knots[i] <= x_flat) & (x_flat < knots[i + 1])
        basis[:, i] = m.to(x.dtype)
    for p in range(1, degree + 1):
        new_n = n_intervals - p
        new_basis = torch.zeros(N, new_n, device=x.device, dtype=x.dtype)
        for i in range(new_n):
            d1 = knots[i + p] - knots[i]
            d2 = knots[i + p + 1] - knots[i + 1]
            t1 = ((x_flat - knots[i]) / d1 * basis[:, i]) if torch.abs(d1) > eps else torch.zeros_like(x_flat)
            t2 = ((knots[i + p + 1] - x_flat) / d2 * basis[:, i + 1]) if torch.abs(d2) > eps else torch.zeros_like(x_flat)
            new_basis[:, i] = t1 + t2
        basis = new_basis
    return basis.view(*orig_shape, num_basis)


class BSplineLatentCompressor(nn.Module):
    """用可学习 B-spline 标量映射压缩油品隐变量，小样本下比纯 MLP 更平滑。"""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_basis: int = 6, dropout: float = 0.0):
        super().__init__()
        self.num_basis = int(num_basis)
        self.pre = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim),
        )
        self.coeffs = nn.Parameter(torch.randn(out_dim, self.num_basis) * 0.05)
        self.psi = nn.Parameter(torch.ones(out_dim) * 0.1)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.post = nn.Sequential(nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        basis = cubic_bspline_basis(h, self.num_basis)  # (..., out_dim, num_basis)
        spline = (basis * self.coeffs.view(*([1] * (basis.dim() - 2)), self.coeffs.shape[0], self.coeffs.shape[1])).sum(dim=-1)
        spline = spline + self.psi.view(*([1] * (h.dim() - 1)), -1) * h
        g = torch.sigmoid(self.gate)
        return self.post(g * spline + (1.0 - g) * h)


class SpectralEncoder(nn.Module):
    def __init__(self, input_len: int, embed_dim: int, base_channels: int, n_blocks: int, dropout: float,
                 aug_cfg: Optional[Dict] = None, use_augmentation: bool = True):
        super().__init__()
        self.input_len = input_len
        self.embed_dim = embed_dim
        self.augment = SpectralAugmentation(**(aug_cfg or {})) if use_augmentation else None
        if input_len <= 0:
            self.disabled = True
            self.dummy = nn.Parameter(torch.zeros(embed_dim), requires_grad=False)
            return
        self.disabled = False
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, padding=3),
            nn.GroupNorm(max(1, min(4, base_channels // 4)), base_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            ResidualConvBlock(base_channels, 7, dilation=2 ** (i % 3), dropout=dropout)
            for i in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(base_channels * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B,L,N)
        B, L, N = x.shape
        if self.disabled or N <= 1:
            return torch.zeros(B, L, self.embed_dim, device=x.device, dtype=x.dtype)
        if self.augment is not None:
            x = self.augment(x)
        h = x.reshape(B * L, 1, N)
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(pooled).view(B, L, self.embed_dim)


class SpectralTemporalMixer(nn.Module):
    """沿老化时间维 L 对谱嵌入 depthwise 时间卷积，显式利用多时刻谱的协同变化。"""
    def __init__(self, embed_dim: int, kernel_size: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.n_layers = int(n_layers)
        if self.n_layers <= 0 or embed_dim <= 0:
            self.net = None
            return
        k = int(kernel_size)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        pad = k // 2
        layers: List[nn.Module] = []
        for _ in range(self.n_layers):
            layers += [
                nn.Conv1d(embed_dim, embed_dim, k, padding=pad, groups=embed_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.net is None or z.shape[1] < 2:
            return z
        m = mask.to(dtype=z.dtype).unsqueeze(-1)
        x = (z * m).transpose(1, 2)
        y = self.net(x).transpose(1, 2)
        return y * m


class FourierTimeEncoder(nn.Module):
    def __init__(self, n_freq: int, out_dim: int, max_time_day: float, dropout: float):
        super().__init__()
        self.n_freq = n_freq
        self.max_time = max(float(max_time_day), 1.0)
        in_dim = 4 + 2 * n_freq
        self.net = MLP(in_dim, max(out_dim, 32), out_dim, dropout)

    def forward(self, t_days, T_K):
        t_norm = t_days / self.max_time
        log_t = torch.log1p(t_days) / np.log1p(self.max_time)
        inv_T = (1.0 / T_K).unsqueeze(1).expand_as(t_days)
        inv_T_scaled = (inv_T - 1.0 / 423.15) / (1.0 / 383.15 - 1.0 / 423.15)
        freqs = torch.arange(1, self.n_freq + 1, device=t_days.device, dtype=t_days.dtype).view(1, 1, -1)
        phase = 2.0 * np.pi * t_norm.unsqueeze(-1) * freqs
        feat = torch.cat([
            t_norm.unsqueeze(-1), log_t.unsqueeze(-1), inv_T_scaled.unsqueeze(-1),
            (t_norm * inv_T_scaled).unsqueeze(-1), torch.sin(phase), torch.cos(phase)
        ], dim=-1)
        return self.net(feat)


class DualPhaseSpectralFusion(nn.Module):
    """三谱自适应融合 + 油相/纸相双头注意力。

    输出：
      fused_spec: 主解码器使用的综合微观特征
      oil_spec: 主要服务于 acid/rho/tanδ/BDV 的油相谱特征
      paper_spec: 主要服务于 DP 的纸相谱特征
      w_mean/w_oil/w_paper: 三谱权重，用于绘图解释
      ai: 光谱老化指数预测头
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.oil_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.paper_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.score_oil = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1))
        self.score_paper = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1))
        self.proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.ai_head = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1), nn.Softplus())

    def _fuse(self, z_stack: torch.Tensor, masks: torch.Tensor, scorer: nn.Module):
        scores = scorer(z_stack).squeeze(-1)  # B,L,M
        scores = scores.masked_fill(~masks, -1e4)
        any_valid = masks.any(dim=2, keepdim=True)
        scores = torch.where(any_valid, scores, torch.zeros_like(scores))
        w = torch.softmax(scores, dim=2)
        w = torch.where(any_valid, w, torch.zeros_like(w))
        return (z_stack * w.unsqueeze(-1)).sum(dim=2), w

    def forward(self, z_list: List[torch.Tensor], mask_list: List[torch.Tensor]):
        z = torch.stack(z_list, dim=2)  # B,L,M,D
        masks = torch.stack(mask_list, dim=2).bool()
        z_oil_stack = z * self.oil_gate(z)
        z_paper_stack = z * self.paper_gate(z)
        oil_spec, w_oil = self._fuse(z_oil_stack, masks, self.score_oil)
        paper_spec, w_paper = self._fuse(z_paper_stack, masks, self.score_paper)
        joint = torch.cat([oil_spec, paper_spec], dim=-1)
        fused = self.proj(joint)
        ai = self.ai_head(joint)
        w_mean = 0.5 * (w_oil + w_paper)
        return fused, oil_spec, paper_spec, w_mean, w_oil, w_paper, ai


class CrossModalSpectralFusion(nn.Module):
    """三谱跨模态注意力，残差加回各模态嵌入。"""
    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        nh = max(1, min(n_heads, embed_dim // 4))
        self.attn = nn.MultiheadAttention(embed_dim, nh, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, z_list: List[torch.Tensor], mask_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        z_stack = torch.stack(z_list, dim=2)
        masks = torch.stack(mask_list, dim=2).bool()
        B, L, M, D = z_stack.shape
        flat = z_stack.reshape(B * L, M, D)
        m_flat = masks.reshape(B * L, M)
        key_pad = ~m_flat
        if (~key_pad).any(dim=1).sum() == 0:
            cross = torch.zeros(B, L, D, device=z_stack.device, dtype=z_stack.dtype)
        else:
            out, _ = self.attn(flat, flat, flat, key_padding_mask=key_pad)
            out = self.norm(out + flat)
            pooled = out.mean(dim=1)
            cross = self.proj(pooled).view(B, L, D)
        enhanced = [z_list[i] + cross for i in range(len(z_list))]
        return enhanced, cross


class MacroTrajectoryEncoder(nn.Module):
    """宏观状态轨迹：相对初值 ΔY + 时间尺度 → z_macro(t)。"""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = MLP(in_dim, hidden_dim, out_dim, dropout)

    def forward(self, macro_feat: torch.Tensor) -> torch.Tensor:
        return self.net(macro_feat)


class DscSeqEncoder(nn.Module):
    """DSC 时序编码 (B,L,n_dsc) → z_dsc(t)。"""
    def __init__(self, dsc_dim: int, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dsc_dim = dsc_dim
        self.embed_dim = embed_dim
        if dsc_dim <= 0:
            self.net = None
            return
        self.net = nn.Sequential(
            nn.Conv1d(dsc_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, dsc_seq: torch.Tensor, dsc_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = dsc_seq.shape
        if self.net is None or dsc_seq.shape[-1] == 0:
            return torch.zeros(B, L, self.embed_dim, device=dsc_seq.device, dtype=dsc_seq.dtype)
        x = dsc_seq.transpose(1, 2)
        y = self.net(x).transpose(1, 2)
        return y * dsc_mask.unsqueeze(-1).to(dtype=y.dtype)


class TemporalOilLatentPool(nn.Module):
    """油品隐变量：由时序油相谱嵌入池化，而非仅 t=0。"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(in_dim, out_dim, batch_first=True, dropout=dropout if dropout > 0 else 0.0)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=seq.dtype).unsqueeze(-1)
        x = seq * m
        h, _ = self.gru(x)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (h * m).sum(dim=1) / denom


class PerOilArrhenius(nn.Module):
    """每油种 Arrhenius 温度因子，用于物理投影路径。"""
    def __init__(self, n_oil: int, oil_vocab: Sequence[str], arr_cfg: Optional[Dict] = None):
        super().__init__()
        arr_cfg = dict(arr_cfg or {})
        T_ref_C = float(arr_cfg.get("arrhenius_T_ref_C", 120.0))
        Ea0 = float(arr_cfg.get("Ea_init_kJ_per_mol", 85.0))
        self.Ea_min = float(arr_cfg.get("Ea_min", 50.0))
        self.Ea_max = float(arr_cfg.get("Ea_max", 130.0))
        lo, hi = arr_cfg.get("k_factor_clamp", [0.25, 4.0])
        self.k_lo, self.k_hi = float(lo), float(hi)
        self.register_buffer("T_ref_K", torch.tensor(T_ref_C + 273.15))
        self.Ea_acid = nn.Parameter(torch.full((max(n_oil, 1),), Ea0))
        self.Ea_dp = nn.Parameter(torch.full((max(n_oil, 1),), Ea0))
        self.logA_acid = nn.Parameter(torch.zeros(max(n_oil, 1)))
        self.logA_dp = nn.Parameter(torch.zeros(max(n_oil, 1)))

    def kinetic_factors(self, oil_id: torch.Tensor, T_K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = oil_id >= 0
        oid = oil_id.clamp(min=0)
        invT = 1.0 / T_K.clamp(min=250.0)
        invT0 = 1.0 / self.T_ref_K
        dinv = invT - invT0
        ea_a = self.Ea_acid[oid].clamp(self.Ea_min, self.Ea_max)
        ea_d = self.Ea_dp[oid].clamp(self.Ea_min, self.Ea_max)
        fa = torch.exp(self.logA_acid[oid] - ea_a / R_GAS * dinv)
        fd = torch.exp(self.logA_dp[oid] - ea_d / R_GAS * dinv)
        one = torch.ones_like(fa)
        fa = torch.where(valid, fa, one)
        fd = torch.where(valid, fd, one)
        return fa.clamp(self.k_lo, self.k_hi), fd.clamp(self.k_lo, self.k_hi)


class DscAnchor(nn.Module):
    def __init__(self, dsc_dim: int, embed_dim: int, hidden_dim: int, max_time_day: float, dropout: float,
                 tc_min_day: float, tc_max_day: float):
        super().__init__()
        self.dsc_dim = dsc_dim
        self.embed_dim = embed_dim
        self.max_time = max(float(max_time_day), 1.0)
        self.tc_min = float(tc_min_day)
        self.tc_max = float(tc_max_day)
        if dsc_dim > 0:
            self.encoder = MLP(dsc_dim + 1, hidden_dim, embed_dim, dropout)
            self.tc_head = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.width_head = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Softplus())
        else:
            self.encoder = None

    def forward(self, dsc_seq, dsc_mask, T_K, t_days):
        B, L = t_days.shape
        if self.encoder is None or dsc_seq.shape[-1] == 0:
            z = torch.zeros(B, self.embed_dim, device=t_days.device, dtype=t_days.dtype)
            tc_norm = torch.full((B,), 0.5, device=t_days.device, dtype=t_days.dtype)
            gate = torch.sigmoid((t_days / self.max_time - tc_norm[:, None]) / 0.1)
            return z, tc_norm, gate, torch.zeros(B, dtype=torch.bool, device=t_days.device)
        # dsc 初始或时间匹配均可；用有效时间点均值作为热稳定性锚点
        m = dsc_mask.bool()
        denom = m.float().sum(1, keepdim=True).clamp(min=1.0)
        dsc_mean = (dsc_seq * m.unsqueeze(-1).float()).sum(1) / denom
        invT = (1.0 / T_K).unsqueeze(-1)
        z = self.encoder(torch.cat([dsc_mean, invT], dim=1))
        tc01 = self.tc_head(z).squeeze(-1)
        tc_day = self.tc_min + tc01 * (self.tc_max - self.tc_min)
        tc_norm = (tc_day / self.max_time).clamp(0.0, 1.5)
        width = (self.width_head(z).squeeze(-1) / self.max_time + 0.035).clamp(0.02, 0.35)
        gate = torch.sigmoid((t_days / self.max_time - tc_norm[:, None]) / width[:, None])
        return z, tc_norm, gate, m.any(1)


class PMTANet(nn.Module):
    def __init__(self, artifacts: DataArtifacts, cfg: Dict):
        super().__init__()
        self.artifacts = artifacts
        self.cfg = cfg
        mc = cfg.get("model", {})
        hidden = int(mc.get("hidden_dim", 96))
        branch_dim = int(mc.get("branch_dim", 96))
        trunk_dim = int(mc.get("trunk_dim", 96))
        spec_dim = int(mc.get("spectral_embed_dim", 32))
        dsc_dim = int(mc.get("dsc_embed_dim", 24))
        oil_dim = int(mc.get("oil_latent_dim", 32))
        dropout = float(mc.get("dropout", 0.08))
        base_ch = int(mc.get("spectral_base_channels", 16))
        blocks = int(mc.get("spectral_blocks", 2))
        self.max_time = artifacts.max_time_day
        self.dsc_regime_gate_mix = float(mc.get("dsc_regime_gate_mix", 0.0))
        self.dsc_gate_prior_center = float(mc.get("dsc_gate_prior_center", 0.46))
        self.dsc_gate_prior_width = float(mc.get("dsc_gate_prior_width", 0.11))
        self.use_spectra = bool(mc.get("use_spectra", True))
        self.use_dsc_anchor = bool(mc.get("use_dsc_anchor", True))
        self.use_oil_onehot = bool(mc.get("use_oil_onehot", False))
        self.use_oil_latent = bool(mc.get("use_oil_latent", True))
        self.use_bspline_latent = bool(mc.get("use_bspline_latent", True))
        self.n_oil = len(artifacts.oil_vocab)
        init_in = 6 + (self.n_oil if self.use_oil_onehot else 0)
        self.init_encoder = MLP(init_in, hidden, branch_dim, dropout)
        aug_all = cfg.get("spectral_augmentation", {})
        use_aug = bool(mc.get("use_spectral_augmentation", True))
        self.ftir_encoder = SpectralEncoder(artifacts.modality_dims.get("ftir", 0), spec_dim, base_ch, blocks, dropout,
                                            aug_cfg=aug_all.get("ftir", {}), use_augmentation=use_aug)
        self.raman_encoder = SpectralEncoder(artifacts.modality_dims.get("raman", 0), spec_dim, base_ch, blocks, dropout,
                                             aug_cfg=aug_all.get("raman", {}), use_augmentation=use_aug)
        self.uv_encoder = SpectralEncoder(artifacts.modality_dims.get("uv", 0), spec_dim, base_ch, blocks, dropout,
                                          aug_cfg=aug_all.get("uv", {}), use_augmentation=use_aug)
        srep = cfg.get("spectral_repr") or {}
        self._spectral_repr_cfg = dict(srep)
        t_layers = max(0, int(srep.get("temporal_mix_layers", 0)))
        t_kern = max(3, int(srep.get("temporal_mix_kernel", 5)))
        t_do = float(srep.get("temporal_mix_dropout", dropout))
        if self.use_spectra and t_layers > 0 and spec_dim > 0:
            self.temporal_mix_ftir = SpectralTemporalMixer(spec_dim, t_kern, t_layers, t_do)
            self.temporal_mix_raman = SpectralTemporalMixer(spec_dim, t_kern, t_layers, t_do)
            self.temporal_mix_uv = SpectralTemporalMixer(spec_dim, t_kern, t_layers, t_do)
        else:
            self.temporal_mix_ftir = None
            self.temporal_mix_raman = None
            self.temporal_mix_uv = None
        self.spec_fusion = DualPhaseSpectralFusion(spec_dim)
        self.cross_modal = CrossModalSpectralFusion(spec_dim, n_heads=int(mc.get("cross_modal_heads", 4)), dropout=dropout)
        macro_in = 5 + 2
        macro_dim = int(mc.get("macro_embed_dim", 32))
        self.macro_encoder = MacroTrajectoryEncoder(macro_in, macro_dim, hidden, dropout)
        effective_dsc_dim = artifacts.dsc_dim if self.use_dsc_anchor else 0
        self.dsc_seq_encoder = DscSeqEncoder(effective_dsc_dim, dsc_dim, hidden, dropout)
        self.dsc_anchor = DscAnchor(
            effective_dsc_dim, dsc_dim, hidden, artifacts.max_time_day, dropout,
            cfg.get("physics", {}).get("dsc_tc_min_day", 1.0),
            cfg.get("physics", {}).get("dsc_tc_max_day", artifacts.max_time_day * 2),
        )
        self.temporal_oil_pool = TemporalOilLatentPool(spec_dim, oil_dim, 0.0)
        oil_latent_in = branch_dim + spec_dim + dsc_dim
        if self.use_bspline_latent:
            self.oil_latent = BSplineLatentCompressor(
                oil_latent_in, oil_dim, hidden,
                num_basis=int(mc.get("bspline_num_basis", 6)), dropout=dropout,
            )
        else:
            self.oil_latent = MLP(oil_latent_in, hidden, oil_dim, dropout)
        self.branch_proj = nn.Sequential(
            nn.Linear(branch_dim + oil_dim + dsc_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, branch_dim), nn.LayerNorm(branch_dim), nn.GELU(),
        )
        self.time_encoder = FourierTimeEncoder(int(mc.get("time_fourier_frequencies", 8)), trunk_dim, artifacts.max_time_day, dropout)
        fuse_dim = 2 * branch_dim + trunk_dim + macro_dim + 2 * spec_dim + 2 * dsc_dim
        self.bt_proj = nn.Linear(branch_dim, branch_dim)
        self.use_dual_task_heads = bool(mc.get("use_dual_task_heads", True))
        fuse_chem = 2 * branch_dim + trunk_dim + macro_dim + spec_dim + dsc_dim
        fuse_elec = 2 * branch_dim + trunk_dim + macro_dim + spec_dim + dsc_dim
        if self.use_dual_task_heads:
            self.decoder_early_elec = MLP(fuse_elec, hidden, 3, dropout)
            self.decoder_early_chem = MLP(fuse_chem, hidden, 2, dropout)
            self.decoder_late_elec = MLP(fuse_elec, hidden, 3, dropout)
            self.decoder_late_chem = MLP(fuse_chem, hidden, 2, dropout)
        else:
            self.decoder_early = MLP(fuse_dim, hidden, len(TARGET_NAMES), dropout)
            self.decoder_late = MLP(fuse_dim, hidden, len(TARGET_NAMES), dropout)
        self.use_macro_trajectory = bool(mc.get("use_macro_trajectory", True))
        self.use_cross_modal = bool(mc.get("use_cross_modal", True))
        self.use_temporal_oil_latent = bool(mc.get("use_temporal_oil_latent", True))
        self.use_dsc_seq_encoder = bool(mc.get("use_dsc_seq_encoder", True))
        self.macro_dim = macro_dim
        self.spec_dim = spec_dim
        self.dsc_dim_embed = dsc_dim
        self.use_physics_projection = bool(mc.get("use_physics_projection", True))
        self.inference_blend = float(mc.get("inference_blend", 0.12))
        if self.use_physics_projection:
            self.param_phys_head = nn.Sequential(
                nn.Linear(fuse_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 5),
            )
            if bool(mc.get("use_arrhenius", True)) and self.n_oil > 0:
                self.arrhenius = PerOilArrhenius(self.n_oil, artifacts.oil_vocab, cfg.get("physics_projection", {}))
            else:
                self.arrhenius = None
        else:
            self.param_phys_head = None
            self.arrhenius = None
        self.log_sigma = nn.Parameter(torch.zeros(len(TARGET_NAMES)))
        tw = mc.get("target_loss_weights", [2.5, 1.0, 1.0, 1.0, 2.2])
        if len(tw) != len(TARGET_NAMES):
            tw = [2.5, 1.0, 1.0, 1.0, 2.2]
        self.register_buffer(
            "target_loss_weights",
            torch.tensor(tw, dtype=torch.float32),
            persistent=False,
        )

    def _oil_onehot(self, oil_id, device, dtype):
        if not self.use_oil_onehot:
            return None
        oh = torch.zeros(oil_id.numel(), self.n_oil, device=device, dtype=dtype)
        valid = oil_id >= 0
        if valid.any():
            oh[valid, oil_id[valid]] = 1.0
        return oh

    @staticmethod
    def _merge_dual_raw(raw_elec: torch.Tensor, raw_chem: torch.Tensor) -> torch.Tensor:
        """raw_elec: (rho, tanδ, BDV)；raw_chem: (acid, DP) → acid, rho, tan, bdv, dp。"""
        acid, dp = raw_chem[..., 0:1], raw_chem[..., 1:2]
        rho, tan, bdv = raw_elec[..., 0:1], raw_elec[..., 1:2], raw_elec[..., 2:3]
        return torch.cat([acid, rho, tan, bdv, dp], dim=-1)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        init_std = batch["init_std"]
        oil_id = batch.get("oil_id")
        if self.use_oil_onehot and oil_id is not None:
            init_in = torch.cat([init_std, self._oil_onehot(oil_id, init_std.device, init_std.dtype)], dim=1)
        else:
            init_in = init_std
        z_init = self.init_encoder(init_in)
        B, L = batch["t_days"].shape
        spec_dim = self.ftir_encoder.embed_dim
        mseq = batch["mask"]
        zf = self.ftir_encoder(batch["ftir"])
        zr = self.raman_encoder(batch["raman"])
        zu = self.uv_encoder(batch["uv"])
        if self.temporal_mix_ftir is not None:
            zf = self.temporal_mix_ftir(zf, mseq)
        if self.temporal_mix_raman is not None:
            zr = self.temporal_mix_raman(zr, mseq)
        if self.temporal_mix_uv is not None:
            zu = self.temporal_mix_uv(zu, mseq)
        mod_masks = [batch["ftir_mask"].bool(), batch["raman_mask"].bool(), batch["uv_mask"].bool()]
        if self.use_cross_modal:
            zf, zr, zu = self.cross_modal([zf, zr, zu], mod_masks)[0]
        spec, oil_spec, paper_spec, spec_w, spec_w_oil, spec_w_paper, spec_ai = self.spec_fusion(
            [zf, zr, zu], mod_masks
        )
        if self.use_macro_trajectory:
            z_macro = self.macro_encoder(batch["macro_feat"])
        else:
            z_macro = torch.zeros(B, L, self.macro_dim, device=init_std.device, dtype=init_std.dtype)
        z_dsc, tc_norm, regime_gate, has_dsc = self.dsc_anchor(
            batch["dsc"], batch["dsc_mask"], batch["T_K"], batch["t_days"]
        )
        dsc_scalar_seq = z_dsc[:, None, :].expand(B, L, -1)
        if self.use_dsc_seq_encoder:
            z_dsc_t = self.dsc_seq_encoder(batch["dsc"], batch["dsc_mask"])
        else:
            z_dsc_t = dsc_scalar_seq
        if self.dsc_regime_gate_mix > 0.0:
            t_ratio = batch["t_days"] / self.max_time
            w = max(self.dsc_gate_prior_width, 1e-6)
            gate_prior = torch.sigmoid((t_ratio - self.dsc_gate_prior_center) / w)
            m = float(self.dsc_regime_gate_mix)
            regime_gate = (1.0 - m) * regime_gate + m * gate_prior
        if self.use_oil_latent:
            if self.use_temporal_oil_latent:
                z_oil_seq = self.temporal_oil_pool(oil_spec, mseq)
            else:
                z_oil_seq = oil_spec[:, 0]
            z_oil = self.oil_latent(torch.cat([z_init, z_oil_seq, z_dsc], dim=1))
        else:
            z_oil = torch.zeros(
                z_init.shape[0],
                self.branch_proj[0].in_features - z_init.shape[1] - z_dsc.shape[1],
                device=z_init.device,
                dtype=z_init.dtype,
            )
        branch = self.branch_proj(torch.cat([z_init, z_oil, z_dsc], dim=1))
        trunk = self.time_encoder(batch["t_days"], batch["T_K"])
        branch_seq = branch[:, None, :].expand(B, L, -1)
        bt = self.bt_proj(branch_seq * trunk[..., :branch_seq.shape[-1]])
        h_base = torch.cat([branch_seq, trunk, bt], dim=-1)
        h_chem = torch.cat([h_base, z_macro, paper_spec, z_dsc_t], dim=-1)
        h_elec = torch.cat([h_base, z_macro, oil_spec, z_dsc_t], dim=-1)
        h = torch.cat([h_base, z_macro, oil_spec, paper_spec, z_dsc_t, dsc_scalar_seq], dim=-1)
        if self.use_dual_task_heads:
            raw_e = self._merge_dual_raw(self.decoder_early_elec(h_elec), self.decoder_early_chem(h_chem))
            raw_l = self._merge_dual_raw(self.decoder_late_elec(h_elec), self.decoder_late_chem(h_chem))
        else:
            raw_e = self.decoder_early(h)
            raw_l = self.decoder_late(h)
        gate = regime_gate.unsqueeze(-1)
        raw = (1.0 - gate) * raw_e + gate * raw_l
        y_data = self._physical_output_transform(raw, batch["init"], batch["t_days"])
        y_phys = y_data
        rates_phys = params_to_rates(torch.zeros(B, L, 5, device=y_data.device, dtype=y_data.dtype))
        if self.use_physics_projection and self.param_phys_head is not None:
            params = self.param_phys_head(h)
            k_sa, k_sd = None, None
            if self.arrhenius is not None and oil_id is not None:
                k_sa, k_sd = self.arrhenius.kinetic_factors(oil_id, batch["T_K"])
            y_phys, rates_phys = integrate_trajectory_physics(
                batch["init"],
                batch["t_days"],
                batch["mask"],
                params,
                float(self.artifacts.acid_sat),
                k_scale_acid=k_sa,
                k_scale_dp=k_sd,
            )
        blend = self.inference_blend if not self.training else float(self.cfg.get("model", {}).get("train_phys_blend", 0.0))
        y_hat = (1.0 - blend) * y_data + blend * y_phys
        return {
            "y_hat": y_hat,
            "y_data": y_data,
            "y_phys": y_phys,
            "raw": raw,
            "rates": rates_phys,
            "tc_norm": tc_norm,
            "regime_gate": regime_gate,
            "has_dsc": has_dsc,
            "spec_weights": spec_w,
            "spec_weights_oil": spec_w_oil,
            "spec_weights_paper": spec_w_paper,
            "spec_ai": spec_ai,
            "branch_latent": branch,
            "oil_latent": z_oil,
        }

    def _physical_output_transform(self, raw, init, t_days):
        # 目标顺序 acid, resistivity, loss_factor, bdv, dp
        tf = torch.log1p(t_days).unsqueeze(-1) / np.log1p(self.max_time)
        pos = F.softplus(raw)
        acid0 = init[:, 0:1].unsqueeze(1)
        rho0 = init[:, 1:2].clamp(min=1e-12).unsqueeze(1)
        tan0 = init[:, 2:3].clamp(min=1e-12).unsqueeze(1)
        bdv0 = init[:, 3:4].clamp(min=1e-12).unsqueeze(1)
        dp0 = init[:, 4:5].clamp(min=1.0).unsqueeze(1)
        acid = acid0 + 0.12 * pos[..., 0:1] * tf * (1.0 + acid0.abs())
        rho = rho0 * torch.exp(-1.20 * pos[..., 1:2] * tf)
        tan = tan0 * torch.exp(1.10 * pos[..., 2:3] * tf)
        bdv = bdv0 * torch.exp(-0.80 * pos[..., 3:4] * tf)
        dp = dp0 * torch.exp(-0.75 * pos[..., 4:5] * tf)
        dp = dp.clamp(min=30.0)
        return torch.cat([acid, rho, tan, bdv, dp], dim=-1)

# --------------------------- 训练封装 ---------------------------

class PMTATrainer:
    def __init__(self, cfg: Dict, artifacts: DataArtifacts):
        self.cfg = cfg
        self.artifacts = artifacts
        self.net = PMTANet(artifacts, cfg)
        self.history: List[Dict] = []

    def to(self, device):
        self.net.to(device)
        return self

    def _batch_to_device(self, batch, device):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _data_nll(self, y_hat, y_true, mask, target_weights: Optional[torch.Tensor] = None):
        mean = torch.as_tensor(self.artifacts.target_scaler.mean_, device=y_hat.device, dtype=y_hat.dtype).view(1, 1, -1)
        std = torch.as_tensor(self.artifacts.target_scaler.std_, device=y_hat.device, dtype=y_hat.dtype).view(1, 1, -1)
        pred_std = (y_hat - mean) / std
        true_std = (y_true - mean) / std
        log_sigma = self.net.log_sigma.view(1, 1, -1).clamp(-5.0, 5.0)
        elem = 0.5 * (torch.exp(-2 * log_sigma) * (pred_std - true_std).pow(2) + 2 * log_sigma)
        if target_weights is not None:
            elem = elem * target_weights.view(1, 1, -1).to(device=elem.device, dtype=elem.dtype)
        m = mask.unsqueeze(-1).float()
        return (elem * m).sum() / (m.sum() * y_hat.shape[-1]).clamp(min=1.0)

    def _loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.net(batch)
        lw = self.cfg.get("training", {}).get("loss_weights", {})
        y_true, mask = batch["target"], batch["mask"]
        y_data = out["y_data"]
        y_phys = out["y_phys"]
        tw = getattr(self.net, "target_loss_weights", None)
        losses = {}
        losses["data"] = self._data_nll(y_data, y_true, mask, tw)
        losses["init"] = initial_anchor_loss(y_data, y_true, mask)
        scale = torch.as_tensor(self.artifacts.target_scaler.std_[: len(TARGET_NAMES)], device=y_data.device, dtype=y_data.dtype)
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        losses["physics_align"] = physics_align_loss(y_data, y_phys, mask, tw, scale=scale)
        losses["data_phys"] = self._data_nll(y_phys, y_true, mask, tw)
        ode_beta = self.cfg.get("physics", {}).get("ode_huber_beta", 0.0)
        try:
            ode_beta_f = float(ode_beta) if ode_beta is not None else 0.0
        except Exception:
            ode_beta_f = 0.0
        ode_beta_use = ode_beta_f if ode_beta_f > 0.0 else None
        losses["dp_ode"] = dp_ode_loss(y_phys, out["rates"], batch["t_days"], mask, huber_beta=ode_beta_use)
        losses["acid_ode"] = acid_ode_loss(
            y_phys, out["rates"], batch["t_days"], mask, self.artifacts.acid_sat, huber_beta=ode_beta_use
        )
        losses["monotonic"] = monotonicity_loss(y_data, batch["t_days"], mask)
        losses["coupling"] = coupling_rank_loss(y_data, batch["t_days"], mask)
        losses["dsc_anchor"] = dsc_anchor_loss(out["tc_norm"], batch["tc_target"], out["has_dsc"])
        losses["spec_consistency"] = spectral_consistency_loss(out["spec_ai"], y_true, mask)
        losses["arrhenius_order"] = arrhenius_order_loss(y_phys, batch["T_K"], batch["oil_type"], mask)
        total = torch.zeros((), device=y_data.device)
        for k, v in losses.items():
            total = total + float(lw.get(k, 0.0)) * v
        return total, {k: float(v.detach().cpu()) for k, v in losses.items()} | {"loss": float(total.detach().cpu())}

    def fit(self, experiments: Sequence[ExperimentCurve], train_idx: Sequence[int], val_idx: Sequence[int]) -> List[Dict]:
        train_cfg = self.cfg.get("training", {})
        device = torch.device(train_cfg.get("device", "cpu"))
        self.to(device)
        train_ds = TrajectoryDataset(experiments, train_idx, self.artifacts, self.cfg)
        val_ds = TrajectoryDataset(experiments, val_idx, self.artifacts, self.cfg) if val_idx else None
        loader = DataLoader(train_ds, batch_size=min(int(train_cfg.get("batch_size", 4)), max(1, len(train_ds))),
                            shuffle=True, collate_fn=collate_trajectories)
        val_loader = None
        if val_ds is not None and len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=min(int(train_cfg.get("batch_size", 4)), len(val_ds)),
                                    shuffle=False, collate_fn=collate_trajectories)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(train_cfg.get("lr", 1e-3)),
                                weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
        epochs = int(train_cfg.get("epochs", 200))
        patience = int(train_cfg.get("patience", 40))
        min_delta = float(train_cfg.get("min_delta", 1e-4))
        grad_clip = float(train_cfg.get("grad_clip", 5.0))
        best_state = copy.deepcopy(self.net.state_dict())
        best_val = float("inf")
        no_improve = 0
        self.history = []

        for ep in range(1, epochs + 1):
            self.net.train()
            logs = []
            for batch in loader:
                batch = self._batch_to_device(batch, device)
                opt.zero_grad(set_to_none=True)
                loss, log = self._loss(batch)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                opt.step()
                logs.append(log)
            train_log = {k: float(np.mean([l[k] for l in logs])) for k in logs[0]} if logs else {"loss": np.nan}
            val_loss = np.nan
            if val_loader is not None:
                self.net.eval()
                vals = []
                with torch.no_grad():
                    for vb in val_loader:
                        vb = self._batch_to_device(vb, device)
                        vl, _ = self._loss(vb)
                        vals.append(float(vl.cpu()))
                val_loss = float(np.mean(vals)) if vals else np.nan
            monitor = val_loss if np.isfinite(val_loss) else train_log["loss"]
            row = {"epoch": ep, **{f"train_{k}": v for k, v in train_log.items()}, "val_loss": val_loss}
            self.history.append(row)
            if monitor + min_delta < best_val:
                best_val = monitor
                best_state = copy.deepcopy(self.net.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
        self.net.load_state_dict(best_state)
        self.fit_ood_reference(experiments, train_idx, device=device)
        return self.history

    @torch.no_grad()
    def fit_ood_reference(self, experiments: Sequence[ExperimentCurve], train_idx: Sequence[int], device=None):
        device = device or next(self.net.parameters()).device
        ds = TrajectoryDataset(experiments, train_idx, self.artifacts, self.cfg)
        loader = DataLoader(ds, batch_size=max(1, len(ds)), shuffle=False, collate_fn=collate_trajectories)
        latents = []
        self.net.eval()
        for batch in loader:
            batch = self._batch_to_device(batch, device)
            out = self.net(batch)
            latents.append(out["branch_latent"].detach().cpu())
        if not latents:
            return
        Z = torch.cat(latents, dim=0)
        mu = Z.mean(dim=0)
        Zc = Z - mu
        cov = (Zc.T @ Zc) / max(Z.shape[0] - 1, 1)
        cov = cov + 1e-3 * torch.eye(cov.shape[0])
        self.artifacts.train_latent_mean = mu
        self.artifacts.train_latent_cov_inv = torch.linalg.pinv(cov)
        # Deep-kernel reference: 使用 branch latent 的 RBF 核估计认识不确定性。
        self.artifacts.train_kernel_latents = Z
        with torch.no_grad():
            if Z.shape[0] > 1:
                pd = torch.pdist(Z)
                ls = torch.median(pd[pd > 1e-8]).item() if (pd > 1e-8).any() else 1.0
            else:
                ls = 1.0
            noise = float(self.cfg.get("model", {}).get("deep_kernel_noise", 1e-3))
            K = torch.exp(-0.5 * torch.cdist(Z / max(ls, 1e-6), Z / max(ls, 1e-6)).pow(2))
            K = K + noise * torch.eye(K.shape[0])
            self.artifacts.train_kernel_k_inv = torch.linalg.pinv(K)
            self.artifacts.kernel_lengthscale = float(max(ls, 1e-6))
            self.artifacts.kernel_noise = noise

    @torch.no_grad()
    def predict(self, experiments: Sequence[ExperimentCurve], indices: Sequence[int], mc_samples: int = 1,
                late_time_cutoff: Optional[float] = None) -> Dict:
        device = next(self.net.parameters()).device
        ds = TrajectoryDataset(experiments, indices, self.artifacts, self.cfg, late_time_cutoff=late_time_cutoff)
        loader = DataLoader(ds, batch_size=max(1, len(ds)), shuffle=False, collate_fn=collate_trajectories)
        preds_all, std_all, targets_all, t_all, mask_all, names_all, aux_all = [], [], [], [], [], [], []
        for batch in loader:
            batch = self._batch_to_device(batch, device)
            sample_preds = []
            sample_aux = None
            if mc_samples <= 1:
                self.net.eval()
                out = self.net(batch)
                sample_preds.append(out["y_hat"].detach())
                sample_aux = out
            else:
                # MC Dropout: 保留 dropout 随机性，同时不更新参数。
                self.net.train()
                for _ in range(mc_samples):
                    out = self.net(batch)
                    sample_preds.append(out["y_hat"].detach())
                    sample_aux = out
                self.net.eval()
            P = torch.stack(sample_preds, dim=0)
            mean = P.mean(dim=0)
            std = P.std(dim=0) if P.shape[0] > 1 else torch.zeros_like(mean)
            preds_all.append(mean.cpu())
            std_all.append(std.cpu())
            targets_all.append(batch["target"].cpu())
            t_all.append(batch["t_days"].cpu())
            mask_all.append(batch["mask"].cpu())
            names_all.extend(batch["name"])
            aux_all.append({
                "tc_norm": sample_aux["tc_norm"].detach().cpu(),
                "spec_weights": sample_aux["spec_weights"].detach().cpu(),
                "spec_weights_oil": sample_aux.get("spec_weights_oil", sample_aux["spec_weights"]).detach().cpu(),
                "spec_weights_paper": sample_aux.get("spec_weights_paper", sample_aux["spec_weights"]).detach().cpu(),
                "branch_latent": sample_aux["branch_latent"].detach().cpu(),
            })
        pred_t = torch.cat(preds_all, dim=0)
        std_t = torch.cat(std_all, dim=0)
        target = torch.cat(targets_all, dim=0).numpy()
        t = torch.cat(t_all, dim=0).numpy()
        mask = torch.cat(mask_all, dim=0).numpy().astype(bool)
        tc = torch.cat([a["tc_norm"] for a in aux_all], dim=0).numpy()
        lat = torch.cat([a["branch_latent"] for a in aux_all], dim=0)
        spec_w = torch.cat([a["spec_weights"] for a in aux_all], dim=0).numpy()
        spec_w_oil = torch.cat([a.get("spec_weights_oil", a["spec_weights"]) for a in aux_all], dim=0).numpy()
        spec_w_paper = torch.cat([a.get("spec_weights_paper", a["spec_weights"]) for a in aux_all], dim=0).numpy()
        ood = self.mahalanobis(lat)
        kernel_u = self.deep_kernel_uncertainty(lat)
        dk_weight = float(self.cfg.get("model", {}).get("deep_kernel_weight", 0.0))
        if dk_weight > 0:
            std_t = torch.sqrt(std_t.pow(2) + dk_weight * kernel_u.view(-1, 1, 1).clamp(min=0.0))
        pred = pred_t.numpy()
        std = std_t.numpy()
        return {"pred": pred, "std": std, "target": target, "t_days": t, "mask": mask,
                "names": names_all, "tc_norm": tc, "spec_weights": spec_w,
                "spec_weights_oil": spec_w_oil, "spec_weights_paper": spec_w_paper,
                "ood_distance": ood.numpy(), "kernel_uncertainty": kernel_u.numpy()}

    def mahalanobis(self, z: torch.Tensor) -> torch.Tensor:
        if self.artifacts.train_latent_mean is None or self.artifacts.train_latent_cov_inv is None:
            return torch.zeros(z.shape[0])
        mu = self.artifacts.train_latent_mean.to(z.device)
        ci = self.artifacts.train_latent_cov_inv.to(z.device)
        dz = z - mu
        return torch.sqrt((dz @ ci * dz).sum(dim=1).clamp(min=0.0))

    def deep_kernel_uncertainty(self, z: torch.Tensor) -> torch.Tensor:
        """RBF deep-kernel posterior variance in latent space. 返回每条轨迹的认识不确定性标量。"""
        if (not bool(self.cfg.get("model", {}).get("use_deep_kernel_uncertainty", True))
                or self.artifacts.train_kernel_latents is None
                or self.artifacts.train_kernel_k_inv is None):
            return torch.zeros(z.shape[0], device=z.device)
        Z = self.artifacts.train_kernel_latents.to(z.device)
        K_inv = self.artifacts.train_kernel_k_inv.to(z.device)
        ls = max(float(self.artifacts.kernel_lengthscale), 1e-6)
        k = torch.exp(-0.5 * torch.cdist(z / ls, Z / ls).pow(2))
        var = 1.0 + float(self.artifacts.kernel_noise) - (k @ K_inv * k).sum(dim=1)
        return var.clamp(min=1e-8)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "cfg": self.cfg,
            "model_state": self.net.state_dict(),
            "artifacts": {
                "target_names": self.artifacts.target_names,
                "modality_dims": self.artifacts.modality_dims,
                "dsc_dim": self.artifacts.dsc_dim,
                "init_scaler": self.artifacts.init_scaler.state_dict(),
                "dsc_scaler": self.artifacts.dsc_scaler.state_dict(),
                "target_scaler": self.artifacts.target_scaler.state_dict(),
                "spectral_norm": self.artifacts.spectral_norm.state_dict(),
                "oil_vocab": self.artifacts.oil_vocab,
                "max_time_day": self.artifacts.max_time_day,
                "acid_sat": self.artifacts.acid_sat,
                "train_latent_mean": self.artifacts.train_latent_mean,
                "train_latent_cov_inv": self.artifacts.train_latent_cov_inv,
                "train_kernel_latents": self.artifacts.train_kernel_latents,
                "train_kernel_k_inv": self.artifacts.train_kernel_k_inv,
                "kernel_lengthscale": self.artifacts.kernel_lengthscale,
                "kernel_noise": self.artifacts.kernel_noise,
            },
            "history": self.history,
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "PMTATrainer":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        a = ckpt["artifacts"]
        artifacts = DataArtifacts(
            target_names=list(a["target_names"]), modality_dims=dict(a["modality_dims"]), dsc_dim=int(a["dsc_dim"]),
            init_scaler=Standardizer().load_state_dict(a["init_scaler"]),
            dsc_scaler=Standardizer().load_state_dict(a["dsc_scaler"]),
            target_scaler=Standardizer().load_state_dict(a["target_scaler"]),
            spectral_norm=SpectralNormalizer().load_state_dict(a["spectral_norm"]),
            oil_vocab=list(a["oil_vocab"]), max_time_day=float(a["max_time_day"]), acid_sat=float(a["acid_sat"]),
            train_latent_mean=a.get("train_latent_mean"), train_latent_cov_inv=a.get("train_latent_cov_inv"),
            train_kernel_latents=a.get("train_kernel_latents"), train_kernel_k_inv=a.get("train_kernel_k_inv"),
            kernel_lengthscale=float(a.get("kernel_lengthscale", 1.0)), kernel_noise=float(a.get("kernel_noise", 1e-3)),
        )
        trainer = cls(ckpt["cfg"], artifacts)
        trainer.net.load_state_dict(ckpt["model_state"])
        trainer.history = ckpt.get("history", [])
        return trainer
