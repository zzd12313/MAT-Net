# -*- coding: utf-8 -*-
"""run.py
PMTA-Net v4 一键流程：
真实数据读取 → 完整轨迹构建 → 温度留出训练 → 曲线监督 + 物理约束训练
→ MC 置信带/OOD/耐热等级评估 → 图和表导出。
"""

from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from data_io import load_experiments, split_by_temperature, make_internal_validation, determine_acid_saturation
from model import build_artifacts, PMTATrainer
from evaluate import evaluate_model, run_leave_one_oil_evaluation
from plot_results import plot_prediction_curves, plot_spectral_weights


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_split(experiments, train_idx, val_idx, test_idx):
    def names(idx): return [experiments[i].name for i in idx]
    print("\n数据划分：")
    print("  Train:", names(train_idx))
    print("  Val:  ", names(val_idx) if val_idx else [])
    print("  Test: ", names(test_idx))


def print_temperature_holdout_summary(cfg: dict, summary: dict) -> None:
    """按 evaluation.metric_priority 先打各目标 R²/MAE/RMSE，再打其余字段。"""
    targets = list(cfg.get("targets", []))
    priority = list(cfg.get("evaluation", {}).get("metric_priority", ["r2", "mae", "rmse"]))
    ordered_keys = []
    for tgt in targets:
        for m in priority:
            k = f"{tgt}_{m}_mean"
            if k in summary:
                ordered_keys.append(k)
    used = set(ordered_keys)
    print("\n温度留出测试 summary（指标顺序 " + " → ".join(priority) + "）：")
    for k in ordered_keys:
        v = summary[k]
        if isinstance(v, float) and v == v:
            print(f"  {k}: {v:.6g}")
    for k, v in summary.items():
        if k in used:
            continue
        if isinstance(v, float) and v == v:
            print(f"  {k}: {v:.6g}")
        elif isinstance(v, int) and not isinstance(v, bool):
            print(f"  {k}: {v}")
        elif isinstance(v, (dict, list)):
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")


def main():
    parser = argparse.ArgumentParser(description="PMTA-Net v4 full-trajectory thermal aging prediction")
    parser.add_argument("--config", default="config_v4.json")
    parser.add_argument("--excel", default=None, help="覆盖 config.data.excel_path")
    parser.add_argument("--output", default=None, help="覆盖 config.output.dir")
    parser.add_argument("--run-looo", action="store_true", help="运行 leave-one-oil-out 迁移评估")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.excel:
        cfg.setdefault("data", {})["excel_path"] = args.excel
    if args.output:
        cfg.setdefault("output", {})["dir"] = args.output
    out_dir = cfg.get("output", {}).get("dir", "output_v4_pmta")
    os.makedirs(out_dir, exist_ok=True)
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    print("=" * 88)
    print("PMTA-Net v4：完整轨迹监督 + Branch-Trunk + 三谱微观编码 + DSC热锚点 + 真ODE物理约束")
    print("=" * 88)

    experiments = load_experiments(cfg)
    train_idx_all, val_idx_cfg, test_idx = split_by_temperature(experiments, cfg)
    if val_idx_cfg:
        train_idx, val_idx = train_idx_all, val_idx_cfg
    else:
        train_idx, val_idx = make_internal_validation(train_idx_all, seed=seed)
    print_split(experiments, train_idx, val_idx, test_idx)

    acid_sat = determine_acid_saturation(experiments, train_idx, cfg)
    print(f"\n训练集 acid_sat 分位估计 = {acid_sat:.6g}（只用训练集，避免测试泄漏）")
    artifacts = build_artifacts(experiments, train_idx, cfg, acid_sat)
    print("\n模型输入维度：")
    print(f"  targets={artifacts.target_names}")
    print(f"  modality_dims={artifacts.modality_dims}; dsc_dim={artifacts.dsc_dim}")
    print(f"  oil_vocab(train)={artifacts.oil_vocab}; use_oil_onehot={cfg.get('model', {}).get('use_oil_onehot', False)}")
    print(f"  max_time_day(train)={artifacts.max_time_day}")

    seeds = list(cfg.get("training", {}).get("multi_seed", [seed]))
    if not seeds:
        seeds = [seed]
    trainers = []
    for s in seeds:
        set_seed(int(s))
        cfg_run = json.loads(json.dumps(cfg))
        cfg_run.setdefault("training", {})["seed"] = int(s)
        acid_sat_s = determine_acid_saturation(experiments, train_idx, cfg_run)
        art_s = build_artifacts(experiments, train_idx, cfg_run, acid_sat_s)
        tr_s = PMTATrainer(cfg_run, art_s)
        tr_s.fit(experiments, train_idx, val_idx)
        sub = os.path.join(out_dir, f"seed_{int(s)}")
        os.makedirs(sub, exist_ok=True)
        tr_s.save(os.path.join(sub, "pmta_net_v4.pt"))
        trainers.append(tr_s)
        print(f"\n[seed {s}] 模型已保存：{os.path.join(sub, 'pmta_net_v4.pt')}")

    trainer = trainers[0]
    if len(trainers) > 1:
        print(f"\n多 seed 训练完成（{len(trainers)} 个），评估使用 seed {seeds[0]} 模型；预测可对多模型取平均。")
    model_path = os.path.join(out_dir, f"seed_{int(seeds[0])}", "pmta_net_v4.pt")
    trainer.save(os.path.join(out_dir, "pmta_net_v4.pt"))

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    mc = int(cfg.get("model", {}).get("mc_dropout_samples", 30))
    result = evaluate_model(trainer, experiments, test_idx, cfg, out_dir, prefix="temperature_holdout", mc_samples=mc)
    print_temperature_holdout_summary(cfg, result["summary"])
    fig_dir = os.path.join(out_dir, "figures")
    plot_prediction_curves(result["summary"]["curves_csv"], fig_dir)
    plot_spectral_weights(result["summary"]["spectral_weights_csv"], fig_dir)

    # 按当前实验设定，训练和预测均使用完整取样阶段（例如 0–180 d）的状态与光谱。
    # early-region 仅作为可选区间误差分析，不作为“早期输入预测后期”的主任务。
    if bool(cfg.get("evaluation", {}).get("run_early_region_eval", False)):
        cutoff = float(cfg.get("data", {}).get("late_time_cutoff_day", 60))
        region_result = evaluate_model(trainer, experiments, test_idx, cfg, out_dir,
                                       prefix=f"early_region_le_{int(cutoff)}d", mc_samples=mc,
                                       late_time_cutoff=cutoff)
        print(f"\n早期区间 <= {cutoff:g} d 误差分析已导出：", region_result["summary"].get("curves_csv"))

    if args.run_looo:
        print("\n运行 leave-one-oil-out 迁移评估。")
        run_leave_one_oil_evaluation(experiments, cfg, os.path.join(out_dir, "leave_one_oil"))

    print(f"\n结果目录：{out_dir}")
    print("  - pmta_net_v4.pt")
    print("  - temperature_holdout_summary.json / metrics.csv / curves.csv")
    print("  - temperature_holdout_spectral_weights.csv")
    print("  - figures/*.png")


if __name__ == "__main__":
    main()
