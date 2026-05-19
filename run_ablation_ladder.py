# -*- coding: utf-8 -*-
"""逐步消融 + 完整模型 + multi_seed 集成；统一 130°C 留出对比表。"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data_io import load_experiments, split_by_temperature, make_internal_validation, determine_acid_saturation
from model import build_artifacts, PMTATrainer
import pandas as pd

from evaluate import evaluate_model, _prediction_to_frames, _metric
from kinetics import TARGET_NAMES


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def composite_score(s: Dict[str, Any]) -> float:
    def g(k, d):
        v = s.get(k, d)
        try:
            return float(v) if v == v else d
        except Exception:
            return d

    return (
        g("acid_rmse_mean", 1.0) / 0.25
        + g("dp_rmse_mean", 100.0) / 900.0
        + g("loss_factor_rmse_mean", 1.0) / 0.45
        + g("bdv_rmse_mean", 5.0) / 4.0
        + (np.log10(max(g("resistivity_rmse_mean", 1e10), 1.0)) / 11.0)
    )


def summary_to_row(tag: str, summ: Dict[str, Any], epochs: int, seeds: str, wall: float) -> Dict[str, Any]:
    return {
        "run": tag,
        "epochs": epochs,
        "seeds": seeds,
        "score": composite_score(summ),
        "acid_rmse": summ.get("acid_rmse_mean"),
        "acid_r2": summ.get("acid_r2_mean"),
        "dp_rmse": summ.get("dp_rmse_mean"),
        "dp_r2": summ.get("dp_r2_mean"),
        "bdv_rmse": summ.get("bdv_rmse_mean"),
        "bdv_r2": summ.get("bdv_r2_mean"),
        "loss_factor_rmse": summ.get("loss_factor_rmse_mean"),
        "loss_factor_r2": summ.get("loss_factor_r2_mean"),
        "resistivity_r2": summ.get("resistivity_r2_mean"),
        "wall_sec": round(wall, 1),
    }


def run_train_eval(cfg: Dict, experiments, tr, va, te, out_sub: Path) -> Dict[str, Any]:
    out_sub.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)
    acid = determine_acid_saturation(experiments, tr, cfg)
    art = build_artifacts(experiments, tr, cfg, acid)
    T = PMTATrainer(cfg, art)
    T.fit(experiments, tr, va)
    T.save(str(out_sub / "pmta_net_v4.pt"))
    mc = int(cfg.get("model", {}).get("mc_dropout_samples", 12))
    summ = evaluate_model(T, experiments, te, cfg, str(out_sub), prefix="holdout", mc_samples=mc)["summary"]
    with open(out_sub / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    return summ


def ensemble_eval(
    cfg: Dict,
    experiments,
    te: Sequence[int],
    ckpt_dirs: List[Path],
    mc: int,
) -> Dict[str, Any]:
    """多 checkpoint 预测均值。"""
    preds, stds, t_days, masks, names, targets = [], [], None, None, None, None
    for d in ckpt_dirs:
        ckpt = d / "pmta_net_v4.pt"
        if not ckpt.is_file():
            continue
        tr = PMTATrainer.load(str(ckpt), map_location="cpu")
        pack = tr.predict(experiments, te, mc_samples=mc)
        preds.append(pack["pred"])
        stds.append(pack["std"])
        if t_days is None:
            t_days = pack["t_days"]
            masks = pack["mask"]
            names = pack["names"]
            targets = pack["target"]
    if not preds:
        return {}
    pred_mean = np.mean(np.stack(preds, axis=0), axis=0)
    pred_std = np.std(np.stack(preds, axis=0), axis=0)
    pack = {
        "pred": pred_mean,
        "std": pred_std,
        "target": targets,
        "t_days": t_days,
        "mask": masks,
        "names": names,
    }
    return _prediction_to_frames(pack, experiments, te, cfg)["summary"]


def ensemble_from_curve_csvs(dirs: List[Path]) -> Dict[str, Any]:
    """无 checkpoint 时：对各 seed 的 holdout_curves.csv 预测列取均值再算指标。"""
    dfs = [pd.read_csv(d / "holdout_curves.csv") for d in dirs if (d / "holdout_curves.csv").is_file()]
    if not dfs:
        return {}
    keys = ["experiment", "oil_type", "temperature_C", "day"]
    base = dfs[0][keys].copy()
    for tgt in TARGET_NAMES:
        pc = f"pred_{tgt}"
        tc = f"true_{tgt}"
        base[tc] = dfs[0][tc]
        base[pc] = np.mean([df[pc].values for df in dfs], axis=0)
    all_metrics = {k: [] for k in TARGET_NAMES}
    for exp, g in base.groupby("experiment"):
        for i, tgt in enumerate(TARGET_NAMES):
            all_metrics[tgt].append(_metric(g[f"true_{tgt}"].values, g[f"pred_{tgt}"].values))
    summary = {}
    for k in TARGET_NAMES:
        for kk in ["mae", "rmse", "r2", "mape"]:
            vals = [m[kk] for m in all_metrics[k] if np.isfinite(m[kk])]
            summary[f"{k}_{kk}_mean"] = float(np.mean(vals)) if vals else np.nan
    summary["n_experiments"] = int(base["experiment"].nunique())
    return summary


def ladder_steps() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("step0_loss_baseline", {
            "model": {
                "use_macro_trajectory": False,
                "use_cross_modal": False,
                "use_temporal_oil_latent": False,
                "use_dsc_seq_encoder": False,
                "use_physics_projection": False,
                "inference_blend": 0.0,
            },
            "training": {
                "loss_weights": {
                    "data": 1.0, "init": 0.4, "physics_align": 0.0, "data_phys": 0.0,
                    "dp_ode": 0.0, "acid_ode": 0.0, "monotonic": 0.0, "coupling": 0.05,
                    "dsc_anchor": 0.03, "spec_consistency": 0.04, "arrhenius_order": 0.0,
                },
            },
        }),
        ("step1_plus_macro", {"model": {"use_macro_trajectory": True}}),
        ("step2_plus_cross_temporal_oil", {
            "model": {"use_cross_modal": True, "use_temporal_oil_latent": True},
        }),
        ("step3_plus_dsc_seq", {"model": {"use_dsc_seq_encoder": True}}),
        ("step4_plus_physics", {
            "model": {"use_physics_projection": True, "inference_blend": 0.12},
            "training": {
                "loss_weights": {
                    "data": 1.0, "init": 0.38, "physics_align": 0.06, "data_phys": 0.12,
                    "dp_ode": 0.015, "acid_ode": 0.0, "monotonic": 0.0, "coupling": 0.03,
                    "dsc_anchor": 0.01, "spec_consistency": 0.05, "arrhenius_order": 0.015,
                },
            },
        }),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v4.json")
    ap.add_argument("--out", default="output_ablation_ladder")
    ap.add_argument("--epochs", type=int, default=120, help="阶梯消融与 matched 完整模型训练轮数")
    ap.add_argument("--full-epochs", type=int, default=220, help="生产配置完整模型轮数")
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--mc", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ensemble-seeds", default="42,43,44")
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--skip-ensemble", action="store_true")
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    os.chdir(str(base_dir))
    with open(args.config, encoding="utf-8") as f:
        base = json.load(f)

    ens_seeds = [int(x.strip()) for x in args.ensemble_seeds.split(",") if x.strip()]
    out_root = base_dir / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(base)
    tr_all, val_c, te = split_by_temperature(experiments, base)
    tr, va = (tr_all, val_c) if val_c else make_internal_validation(tr_all, seed=args.seed)

    rows: List[Dict[str, Any]] = []
    cumulative: Dict[str, Any] = {}
    t0 = time.time()

    if not args.skip_ladder:
        fast = {
            "training": {
                "epochs": args.epochs,
                "patience": args.patience,
                "seed": args.seed,
                "multi_seed": [args.seed],
            },
            "model": {"mc_dropout_samples": args.mc},
        }
        for tag, patch in ladder_steps():
            cumulative = deep_merge(copy.deepcopy(cumulative), patch)
            cfg = copy.deepcopy(base)
            deep_merge(cfg, fast)
            deep_merge(cfg, cumulative)
            cfg.setdefault("output", {})["dir"] = str(out_root / tag)
            print(f"\n=== {tag} (epochs={args.epochs}) ===", flush=True)
            summ = run_train_eval(cfg, experiments, tr, va, te, out_root / tag)
            rows.append(summary_to_row(tag, summ, args.epochs, str(args.seed), time.time() - t0))
            print(rows[-1], flush=True)

    if not args.skip_ensemble and ens_seeds:
        print(f"\n=== step5_multi_seed_ensemble (step4 配置, seeds={ens_seeds}) ===", flush=True)
        cfg_base = copy.deepcopy(base)
        deep_merge(cfg_base, {
            "training": {"epochs": args.epochs, "patience": args.patience, "multi_seed": [args.seed]},
            "model": {"mc_dropout_samples": args.mc},
        })
        deep_merge(cfg_base, cumulative)
        ckpt_dirs = []
        for s in ens_seeds:
            cfg_s = copy.deepcopy(cfg_base)
            cfg_s["training"]["seed"] = s
            sub = out_root / f"step5_physics_seed{s}"
            cfg_s.setdefault("output", {})["dir"] = str(sub)
            run_train_eval(cfg_s, experiments, tr, va, te, sub)
            ckpt_dirs.append(sub)
        summ_e = ensemble_eval(cfg_base, experiments, te, ckpt_dirs, args.mc)
        if not summ_e:
            summ_e = ensemble_from_curve_csvs(ckpt_dirs)
        if summ_e:
            rows.append(summary_to_row(
                "step5_multi_seed_ensemble", summ_e, args.epochs, ",".join(map(str, ens_seeds)), time.time() - t0
            ))
            print(rows[-1], flush=True)
            with open(out_root / "step5_ensemble" / "holdout_summary.json", "w", encoding="utf-8") as f:
                json.dump(summ_e, f, ensure_ascii=False, indent=2)
            (out_root / "step5_ensemble").mkdir(parents=True, exist_ok=True)

    if not args.skip_full:
        print(f"\n=== full_model_matched (config 全开, epochs={args.epochs}) ===", flush=True)
        cfg_m = copy.deepcopy(base)
        deep_merge(cfg_m, {
            "training": {"epochs": args.epochs, "patience": args.patience, "seed": args.seed, "multi_seed": [args.seed]},
            "model": {
                "mc_dropout_samples": args.mc,
                "use_macro_trajectory": True,
                "use_cross_modal": True,
                "use_temporal_oil_latent": True,
                "use_dsc_seq_encoder": True,
                "use_physics_projection": True,
            },
        })
        cfg_m.setdefault("output", {})["dir"] = str(out_root / "full_model_matched")
        summ_m = run_train_eval(cfg_m, experiments, tr, va, te, out_root / "full_model_matched")
        rows.append(summary_to_row("full_model_matched", summ_m, args.epochs, str(args.seed), time.time() - t0))
        print(rows[-1], flush=True)

        if ens_seeds:
            print(f"\n=== full_model_matched_ensemble (epochs={args.epochs}) ===", flush=True)
            ckpt_dirs = []
            for s in ens_seeds:
                cfg_s = copy.deepcopy(cfg_m)
                cfg_s["training"]["seed"] = s
                sub = out_root / f"full_model_matched_seed{s}"
                cfg_s.setdefault("output", {})["dir"] = str(sub)
                run_train_eval(cfg_s, experiments, tr, va, te, sub)
                ckpt_dirs.append(sub)
            summ_me = ensemble_eval(cfg_m, experiments, te, ckpt_dirs, args.mc)
            if not summ_me:
                summ_me = ensemble_from_curve_csvs(ckpt_dirs)
            if summ_me:
                rows.append(summary_to_row(
                    "full_model_matched_ensemble", summ_me, args.epochs, ",".join(map(str, ens_seeds)), time.time() - t0
                ))
                print(rows[-1], flush=True)

        print(f"\n=== full_model_production (config 默认, epochs={args.full_epochs}) ===", flush=True)
        cfg_p = copy.deepcopy(base)
        deep_merge(cfg_p, {
            "training": {"epochs": args.full_epochs, "patience": args.patience, "seed": args.seed, "multi_seed": [args.seed]},
            "model": {"mc_dropout_samples": args.mc},
        })
        cfg_p.setdefault("output", {})["dir"] = str(out_root / "full_model_production")
        summ_p = run_train_eval(cfg_p, experiments, tr, va, te, out_root / "full_model_production")
        rows.append(summary_to_row("full_model_production", summ_p, args.full_epochs, str(args.seed), time.time() - t0))
        print(rows[-1], flush=True)

        if ens_seeds:
            print(f"\n=== full_model_production_ensemble (epochs={args.full_epochs}) ===", flush=True)
            ckpt_dirs = []
            for s in ens_seeds:
                cfg_s = copy.deepcopy(cfg_p)
                cfg_s["training"]["seed"] = s
                sub = out_root / f"full_model_production_seed{s}"
                cfg_s.setdefault("output", {})["dir"] = str(sub)
                run_train_eval(cfg_s, experiments, tr, va, te, sub)
                ckpt_dirs.append(sub)
            summ_pe = ensemble_eval(cfg_p, experiments, te, ckpt_dirs, args.mc)
            if not summ_pe:
                summ_pe = ensemble_from_curve_csvs(ckpt_dirs)
            if summ_pe:
                rows.append(summary_to_row(
                    "full_model_production_ensemble", summ_pe, args.full_epochs,
                    ",".join(map(str, ens_seeds)), time.time() - t0,
                ))
                print(rows[-1], flush=True)

    csv_path = out_root / "comparison_all.csv"
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(out_root / "comparison_all.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {csv_path} ({len(rows)} runs, wall={time.time()-t0:.1f}s)")
    if rows:
        best = min(rows, key=lambda r: r["score"])
        print("BEST (composite):", best["run"], "score", f"{best['score']:.4f}")


if __name__ == "__main__":
    main()
