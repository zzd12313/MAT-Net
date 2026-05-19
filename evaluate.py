# -*- coding: utf-8 -*-
"""evaluate.py
PMTA-Net v4 评估模块：完整轨迹、温度留出、留一油种、后期时间预测、MC 置信带和 OOD 距离。
"""

from __future__ import annotations
from typing import Dict, Sequence, Optional, List
import os
import json
import numpy as np
import pandas as pd

from data_io import ExperimentCurve, split_leave_one_oil, determine_acid_saturation
from kinetics import TARGET_NAMES, judge_thermal_grade, fit_lifetime_surface
from model import build_artifacts, PMTATrainer


def _metric(y, yp):
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yp)
    if mask.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "mape": np.nan}
    y, yp = y[mask], yp[mask]
    mae = float(np.mean(np.abs(y - yp)))
    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - ss_res / max(ss_tot, 1e-12))
    mape = float(np.mean(np.abs((y - yp) / np.maximum(np.abs(y), 1e-12))))
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def _prediction_to_frames(pred_pack: Dict, experiments: Sequence[ExperimentCurve], indices: Sequence[int], cfg: Dict) -> Dict:
    rows, curve_rows, spec_rows = [], [], []
    pred, std, target = pred_pack["pred"], pred_pack["std"], pred_pack["target"]
    t_days, mask = pred_pack["t_days"], pred_pack["mask"]
    names = pred_pack["names"]
    all_metrics = {k: [] for k in TARGET_NAMES}
    grade_records = []
    threshold = float(cfg.get("model", {}).get("ood_mahalanobis_threshold", 9.0))

    for n, name in enumerate(names):
        valid = mask[n]
        curves = {k: pred[n, valid, i] for i, k in enumerate(TARGET_NAMES)}
        true_curves = {k: target[n, valid, i] for i, k in enumerate(TARGET_NAMES)}
        std_curves = {k: std[n, valid, i] for i, k in enumerate(TARGET_NAMES)}
        exp = experiments[indices[n]]
        row = {
            "experiment": name,
            "oil_type": exp.oil_type,
            "temperature_C": exp.temperature_C,
            "ood_distance": float(pred_pack["ood_distance"][n]),
            "ood_flag": bool(pred_pack["ood_distance"][n] > threshold),
            "tc_pred_day": float(pred_pack["tc_norm"][n] * max(t_days[n, valid].max(), 1.0)),
            "kernel_uncertainty": float(pred_pack.get("kernel_uncertainty", np.zeros(len(names)))[n]),
        }
        for i, k in enumerate(TARGET_NAMES):
            m = _metric(true_curves[k], curves[k])
            all_metrics[k].append(m)
            for kk, vv in m.items():
                row[f"{k}_{kk}"] = vv
        grade = judge_thermal_grade(t_days[n, valid], curves, cfg)
        row.update(grade)
        rows.append(row)
        grade_records.append({"temperature_C": exp.temperature_C, **grade})
        for j, d in enumerate(t_days[n, valid]):
            cr = {"experiment": name, "oil_type": exp.oil_type, "temperature_C": exp.temperature_C, "day": float(d)}
            for i, k in enumerate(TARGET_NAMES):
                cr[f"true_{k}"] = float(true_curves[k][j])
                cr[f"pred_{k}"] = float(curves[k][j])
                cr[f"std_{k}"] = float(std_curves[k][j])
            curve_rows.append(cr)
        sw = pred_pack.get("spec_weights")
        if sw is not None:
            for j, d in enumerate(t_days[n, valid]):
                if j < sw.shape[1]:
                    row_sw = {
                        "experiment": name, "day": float(d),
                        "w_ftir": float(sw[n, j, 0]), "w_raman": float(sw[n, j, 1]), "w_uv": float(sw[n, j, 2]),
                    }
                    sw_oil = pred_pack.get("spec_weights_oil")
                    sw_paper = pred_pack.get("spec_weights_paper")
                    if sw_oil is not None and j < sw_oil.shape[1]:
                        row_sw.update({
                            "w_oil_ftir": float(sw_oil[n, j, 0]),
                            "w_oil_raman": float(sw_oil[n, j, 1]),
                            "w_oil_uv": float(sw_oil[n, j, 2]),
                        })
                    if sw_paper is not None and j < sw_paper.shape[1]:
                        row_sw.update({
                            "w_paper_ftir": float(sw_paper[n, j, 0]),
                            "w_paper_raman": float(sw_paper[n, j, 1]),
                            "w_paper_uv": float(sw_paper[n, j, 2]),
                        })
                    spec_rows.append(row_sw)
    detail = pd.DataFrame(rows)
    curves_df = pd.DataFrame(curve_rows)
    spec_df = pd.DataFrame(spec_rows)
    summary = {}
    for k in TARGET_NAMES:
        for kk in ["mae", "rmse", "r2", "mape"]:
            vals = [m[kk] for m in all_metrics[k] if np.isfinite(m[kk])]
            summary[f"{k}_{kk}_mean"] = float(np.mean(vals)) if vals else np.nan
    summary["n_experiments"] = len(names)
    summary["lifetime_surface"] = fit_lifetime_surface(grade_records)
    summary["ood_flag_count"] = int(detail["ood_flag"].sum()) if not detail.empty and "ood_flag" in detail else 0
    return {"summary": summary, "detail": detail, "curves": curves_df, "spectral_weights": spec_df}


def evaluate_model(trainer: PMTATrainer, experiments: Sequence[ExperimentCurve], indices: Sequence[int], cfg: Dict,
                   output_dir: str, prefix: str = "temperature_holdout", mc_samples: Optional[int] = None,
                   late_time_cutoff: Optional[float] = None) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    mc = int(mc_samples if mc_samples is not None else cfg.get("model", {}).get("mc_dropout_samples", 1))
    pack = trainer.predict(experiments, indices, mc_samples=mc, late_time_cutoff=late_time_cutoff)
    result = _prediction_to_frames(pack, experiments, indices, cfg)
    detail_path = os.path.join(output_dir, f"{prefix}_metrics.csv")
    curves_path = os.path.join(output_dir, f"{prefix}_curves.csv")
    spec_path = os.path.join(output_dir, f"{prefix}_spectral_weights.csv")
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    result["detail"].to_csv(detail_path, index=False, encoding="utf-8-sig")
    result["curves"].to_csv(curves_path, index=False, encoding="utf-8-sig")
    result["spectral_weights"].to_csv(spec_path, index=False, encoding="utf-8-sig")
    result["summary"]["detail_csv"] = detail_path
    result["summary"]["curves_csv"] = curves_path
    result["summary"]["spectral_weights_csv"] = spec_path
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)
    return result


def run_leave_one_oil_evaluation(experiments: Sequence[ExperimentCurve], cfg: Dict, output_dir: str) -> Dict:
    """逐油种留出。由于仅两类油时证据有限，结果应表述为迁移初步验证。"""
    os.makedirs(output_dir, exist_ok=True)
    records = {}
    for oil in sorted({e.oil_type for e in experiments}):
        train_idx, test_idx = split_leave_one_oil(experiments, oil)
        if not train_idx or not test_idx:
            continue
        acid_sat = determine_acid_saturation(experiments, train_idx, cfg)
        artifacts = build_artifacts(experiments, train_idx, cfg, acid_sat)
        trainer = PMTATrainer(cfg, artifacts)
        trainer.fit(experiments, train_idx, [])
        records[oil] = evaluate_model(trainer, experiments, test_idx, cfg, output_dir, prefix=f"leave_oil_{oil}")["summary"]
    with open(os.path.join(output_dir, "leave_one_oil_summary.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records
