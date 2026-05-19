# -*- coding: utf-8 -*-
"""predict_new.py
PMTA-Net v4 新油样曲线预测。

支持 JSON 输入：
{
  "oil_type": "NEW_SE_01",
  "temperature_C": 130,
  "acid0": 0.01,
  "resistivity0": 1e12,
  "loss_factor0": 0.003,
  "bdv0": 75,
  "dp0": 1111,
  "viscosity0": 30,
  "ftir_npy": "ftir.npy",      # shape=(L,) 或 (n_time,L)
  "raman_npy": "raman.npy",
  "uv_npy": "uv.npy",
  "dsc": [onset, peak, deltaH, oit]
}
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from data_io import ExperimentCurve, InitialValues
from model import PMTATrainer
from kinetics import judge_thermal_grade, TARGET_NAMES


def _load_modality(path, n_time):
    if not path:
        return None
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        arr = np.repeat(arr[None, :], n_time, axis=0)
    if arr.ndim != 2:
        raise ValueError(f"光谱 npy 应为 (L,) 或 (n_time,L)，实际 {arr.shape}")
    if arr.shape[0] != n_time:
        # 新油只有少量谱时，用最近谱重复到预测网格；真实动态谱可保持 n_time 一致。
        arr = np.repeat(arr[:1], n_time, axis=0)
    return arr


def build_experiment_from_json(obj, trainer: PMTATrainer, predict_days: float, grid_n: int) -> ExperimentCurve:
    t_days = np.linspace(0, float(predict_days), int(grid_n), dtype=np.float32)
    init = InitialValues(
        acid0=float(obj.get("acid0", obj.get("acid", 0.0))),
        resistivity0=float(obj.get("resistivity0", obj.get("resistivity", 1.0))),
        loss_factor0=float(obj.get("loss_factor0", obj.get("loss_factor", 1.0))),
        bdv0=float(obj.get("bdv0", obj.get("bdv", 1.0))),
        dp0=float(obj.get("dp0", obj.get("dp", 1111.0))),
        viscosity0=float(obj.get("viscosity0", 0.0)),
    )
    exp = ExperimentCurve(
        oil_type=str(obj.get("oil_type", "UNKNOWN")), temperature_C=float(obj["temperature_C"]),
        t_days=t_days,
        acid=np.full_like(t_days, init.acid0, dtype=np.float32),
        resistivity=np.full_like(t_days, init.resistivity0, dtype=np.float32),
        loss_factor=np.full_like(t_days, init.loss_factor0, dtype=np.float32),
        bdv=np.full_like(t_days, init.bdv0, dtype=np.float32),
        dp=np.full_like(t_days, init.dp0, dtype=np.float32),
        viscosity=np.full_like(t_days, init.viscosity0, dtype=np.float32),
        ftir=_load_modality(obj.get("ftir_npy"), len(t_days)),
        raman=_load_modality(obj.get("raman_npy"), len(t_days)),
        uv=_load_modality(obj.get("uv_npy"), len(t_days)),
    )
    if obj.get("dsc") is None:
        raise ValueError("新样本 JSON 必须提供 dsc 列表（onset/peak/delta_H/oit 等）。")
    d = np.asarray(obj["dsc"], dtype=np.float32)
    exp.dsc_features = np.repeat(d[None, :], len(t_days), axis=0)
    for m, p in [("ftir", obj.get("ftir_npy")), ("raman", obj.get("raman_npy")), ("uv", obj.get("uv_npy"))]:
        if not p:
            raise ValueError(f"新样本 JSON 必须提供 {m}_npy 光谱文件路径。")
    exp.initial = init
    return exp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="新样本 JSON")
    parser.add_argument("--predict-days", type=float, default=None)
    parser.add_argument("--grid-points", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--output", default="prediction_v4_pmta")
    args = parser.parse_args()

    trainer = PMTATrainer.load(args.model, map_location="cpu")
    with open(args.input, "r", encoding="utf-8") as f:
        obj = json.load(f)
    predict_days = float(args.predict_days or trainer.cfg.get("physics", {}).get("max_predict_days", 420))
    grid_n = int(args.grid_points or trainer.cfg.get("physics", {}).get("curve_grid_points", 300))
    exp = build_experiment_from_json(obj, trainer, predict_days, grid_n)
    mc = int(args.mc_samples or trainer.cfg.get("model", {}).get("mc_dropout_samples", 30))
    pack = trainer.predict([exp], [0], mc_samples=mc)
    pred = pack["pred"][0, pack["mask"][0]]
    std = pack["std"][0, pack["mask"][0]]
    t = pack["t_days"][0, pack["mask"][0]]
    curves = {k: pred[:, i] for i, k in enumerate(TARGET_NAMES)}
    grade = judge_thermal_grade(t, curves, trainer.cfg)
    os.makedirs(args.output, exist_ok=True)
    df = pd.DataFrame({"day": t})
    for i, k in enumerate(TARGET_NAMES):
        df[k] = pred[:, i]
        df[f"std_{k}"] = std[:, i]
    df.to_csv(os.path.join(args.output, "predicted_curves.csv"), index=False, encoding="utf-8-sig")
    sw = pack.get("spec_weights")
    if sw is not None:
        sw_df = pd.DataFrame({
            "day": t,
            "w_ftir": sw[0, :len(t), 0], "w_raman": sw[0, :len(t), 1], "w_uv": sw[0, :len(t), 2]
        })
        sw_oil = pack.get("spec_weights_oil")
        sw_paper = pack.get("spec_weights_paper")
        if sw_oil is not None:
            sw_df["w_oil_ftir"] = sw_oil[0, :len(t), 0]
            sw_df["w_oil_raman"] = sw_oil[0, :len(t), 1]
            sw_df["w_oil_uv"] = sw_oil[0, :len(t), 2]
        if sw_paper is not None:
            sw_df["w_paper_ftir"] = sw_paper[0, :len(t), 0]
            sw_df["w_paper_raman"] = sw_paper[0, :len(t), 1]
            sw_df["w_paper_uv"] = sw_paper[0, :len(t), 2]
        sw_df.to_csv(os.path.join(args.output, "spectral_weights.csv"), index=False, encoding="utf-8-sig")
    meta = {
        "thermal_grade": grade,
        "ood_distance": float(pack["ood_distance"][0]),
        "kernel_uncertainty": float(pack.get("kernel_uncertainty", [0.0])[0]),
    }
    with open(os.path.join(args.output, "prediction_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("预测完成：", args.output)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
