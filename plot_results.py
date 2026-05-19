# -*- coding: utf-8 -*-
"""plot_results.py
PMTA-Net v4 论文图：多属性曲线、MC 置信带、三谱权重、寿命面辅助图。
"""

from __future__ import annotations
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = ["acid", "resistivity", "loss_factor", "bdv", "dp"]


def plot_prediction_curves(curves_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(curves_csv)
    for exp, g in df.groupby("experiment"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i, name in enumerate(TARGETS):
            ax = axes[i]
            ax.plot(g["day"], g[f"pred_{name}"], linewidth=2, label="prediction")
            if f"std_{name}" in g:
                lo = g[f"pred_{name}"] - 1.96 * g[f"std_{name}"]
                hi = g[f"pred_{name}"] + 1.96 * g[f"std_{name}"]
                ax.fill_between(g["day"], lo, hi, alpha=0.18, label="95% band")
            ax.scatter(g["day"], g[f"true_{name}"], s=28, label="measurement")
            ax.set_title(name)
            ax.set_xlabel("aging time / day")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[-1].axis("off")
        fig.suptitle(exp)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"curves_{exp}.png"), dpi=300)
        plt.close(fig)


def plot_spectral_weights(weights_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(weights_csv):
        return
    df = pd.read_csv(weights_csv)
    if df.empty:
        return
    for exp, g in df.groupby("experiment"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(g["day"], g["w_ftir"], marker="o", label="FTIR")
        ax.plot(g["day"], g["w_raman"], marker="s", label="Raman")
        ax.plot(g["day"], g["w_uv"], marker="^", label="UV")
        ax.set_xlabel("aging time / day")
        ax.set_ylabel("modality weight")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"spectral_weights_{exp}.png"), dpi=300)
        plt.close(fig)
        phase_cols = ["w_oil_ftir", "w_oil_raman", "w_oil_uv", "w_paper_ftir", "w_paper_raman", "w_paper_uv"]
        if all(c in g.columns for c in phase_cols):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(g["day"], g["w_oil_ftir"], label="Oil-FTIR", linestyle="-")
            ax.plot(g["day"], g["w_oil_raman"], label="Oil-Raman", linestyle="-")
            ax.plot(g["day"], g["w_oil_uv"], label="Oil-UV", linestyle="-")
            ax.plot(g["day"], g["w_paper_ftir"], label="Paper-FTIR", linestyle="--")
            ax.plot(g["day"], g["w_paper_raman"], label="Paper-Raman", linestyle="--")
            ax.plot(g["day"], g["w_paper_uv"], label="Paper-UV", linestyle="--")
            ax.set_xlabel("aging time / day")
            ax.set_ylabel("phase-aware modality weight")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"phase_spectral_weights_{exp}.png"), dpi=300)
            plt.close(fig)
