# -*- coding: utf-8 -*-
"""
data_io.py
PMTA-Net v4 数据入口。

核心变化
1. 保留 v3 的 Excel/CSV 长表读取和 ExperimentCurve 轨迹对象。
2. 将 FTIR/Raman/UV 拆成独立模态，支持 time-matched S(t) 光谱轨迹。
3. 对重复 (oil,T,day) 数据按均值聚合并保留标准差提示，避免重复实验被错误当作独立曲线。
4. acid_sat 按训练集分位数估计，避免单个异常值或测试集泄漏。
5. BDV 列允许 `均值(各次测量...)` 的 Excel 文本：括号前为多次测量的平均，括号内为各次结果；读取时取括号前的均值参与建模。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import os
import re
import numpy as np
import pandas as pd

TARGET_NAMES_DEFAULT = ["acid", "resistivity", "loss_factor", "bdv", "dp"]
REQUIRED_LOGICAL_COLS = [
    "oil_type", "temperature_C", "aging_time_day",
    "acid", "resistivity", "loss_factor", "bdv", "dp",
]


@dataclass
class InitialValues:
    acid0: float
    resistivity0: float
    loss_factor0: float
    bdv0: float
    dp0: float
    viscosity0: float = np.nan


@dataclass
class ExperimentCurve:
    oil_type: str
    temperature_C: float
    t_days: np.ndarray
    acid: np.ndarray
    resistivity: np.ndarray
    loss_factor: np.ndarray
    bdv: np.ndarray
    dp: np.ndarray
    viscosity: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    ftir: Optional[np.ndarray] = None      # (n_time, L_ftir)
    raman: Optional[np.ndarray] = None     # (n_time, L_raman)
    uv: Optional[np.ndarray] = None        # (n_time, L_uv)
    dsc_features: Optional[np.ndarray] = None  # (n_time, n_dsc) or repeated initial DSC
    initial: Optional[InitialValues] = None
    meta: Dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.oil_type}_{int(round(self.temperature_C))}C"

    @property
    def t_hours(self) -> np.ndarray:
        return self.t_days * 24.0

    def target_matrix(self, target_names: Sequence[str] = TARGET_NAMES_DEFAULT) -> np.ndarray:
        arrays = []
        for name in target_names:
            if not hasattr(self, name):
                raise ValueError(f"ExperimentCurve 缺少目标字段: {name}")
            arrays.append(np.asarray(getattr(self, name), dtype=np.float32))
        return np.stack(arrays, axis=-1).astype(np.float32)


def _standardize_column_name(col: str) -> str:
    c = str(col).strip().replace("（", "(").replace("）", ")").replace("℃", "C")
    c_low = re.sub(r"[\s\-/]+", "_", c.lower().strip())
    aliases = {
        "oil": "oil_type", "oiltype": "oil_type", "oil_type": "oil_type", "sample": "oil_type",
        "sample_id": "oil_type", "oil_name": "oil_type",
        "temperature": "temperature_C", "temperature_c": "temperature_C", "temp": "temperature_C",
        "temp_c": "temperature_C", "t_c": "temperature_C",
        "day": "aging_time_day", "days": "aging_time_day", "aging_day": "aging_time_day",
        "aging_days": "aging_time_day", "time_day": "aging_time_day", "time_days": "aging_time_day",
        "t_day": "aging_time_day", "aging_time_day": "aging_time_day",
        "acid_value": "acid", "acid": "acid", "acidity": "acid", "av": "acid",
        "rho": "resistivity", "volume_resistivity": "resistivity", "resistivity": "resistivity",
        "tan_delta": "loss_factor", "tandelta": "loss_factor", "tanδ": "loss_factor",
        "loss_factor": "loss_factor", "dielectric_loss": "loss_factor",
        "breakdown_voltage": "bdv", "bdv": "bdv", "breakdown": "bdv", "ubdv": "bdv",
        "dp": "dp", "degree_of_polymerization": "dp", "polymerization": "dp",
        "viscosity": "viscosity", "kinematic_viscosity": "viscosity",
    }
    return aliases.get(c_low, c_low)


def _read_excel_or_csv(path: str, sheet_name=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if sheet_name is None:
            xls = pd.ExcelFile(path)
            sheet_name = "measurements" if "measurements" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet_name)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"不支持的数据文件格式: {ext}")


def _sorted_prefixed_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]

    def key(c):
        nums = re.findall(r"[-+]?\d*\.?\d+", str(c))
        return float(nums[-1]) if nums else str(c)
    return sorted(cols, key=key)


def _numeric_series_mean_before_parentheses(series: pd.Series) -> pd.Series:
    """纯数字照常解析；无法解析时取字符串开头的数值（约定：`67(63.8,...)` 中 67 为均值，括号内为各次测量）。"""
    out = pd.to_numeric(series, errors="coerce")
    bad = out.isna() & series.notna()
    if not bad.any():
        return out
    s = series[bad].astype(str).str.strip()
    extracted = s.str.extract(r"^([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", expand=False)
    out.loc[bad] = pd.to_numeric(extracted, errors="coerce").values
    return out


def _prepare_dataframe(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    rename = {c: _standardize_column_name(c) for c in df.columns}
    df = df.rename(columns=rename).copy()
    missing = [c for c in REQUIRED_LOGICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Excel/CSV 缺少必要列: {missing}\n当前列名: {list(df.columns)}")

    numeric_cols = [
        "temperature_C", "aging_time_day", "acid", "resistivity", "loss_factor", "bdv", "dp",
        "viscosity",
    ]
    prefixes = [
        cfg.get("columns", {}).get("ftir_prefix", "ftir_"),
        cfg.get("columns", {}).get("raman_prefix", "raman_"),
        cfg.get("columns", {}).get("uv_prefix", "uv_"),
    ]
    for p in prefixes:
        numeric_cols += [c for c in df.columns if str(c).startswith(p)]
    numeric_cols += [c for c in df.columns if str(c).startswith("dsc_")]

    for c in numeric_cols:
        if c in df.columns:
            if c == "bdv":
                df[c] = _numeric_series_mean_before_parentheses(df[c])
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_LOGICAL_COLS)
    df["oil_type"] = df["oil_type"].astype(str).str.strip()

    exclude = set(cfg.get("data", {}).get("exclude_oil_types", []))
    if exclude:
        df = df[~df["oil_type"].isin(exclude)].copy()

    if cfg.get("data", {}).get("duplicate_policy", "aggregate_mean") == "aggregate_mean":
        keys = ["oil_type", "temperature_C", "aging_time_day"]
        value_cols = [c for c in df.columns if c not in keys]
        numeric_value_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
        other_cols = [c for c in value_cols if c not in numeric_value_cols]
        agg = {c: "mean" for c in numeric_value_cols}
        for c in other_cols:
            agg[c] = "first"
        dup_count = df.groupby(keys).size().rename("replicate_count").reset_index()
        df = df.groupby(keys, as_index=False).agg(agg)
        df = df.merge(dup_count, on=keys, how="left")
    return df.sort_values(["oil_type", "temperature_C", "aging_time_day"]).reset_index(drop=True)


def _safe_first(arr: np.ndarray, default: float) -> float:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite[0]) if finite.size else float(default)


def infer_initial_values(exp: ExperimentCurve, cfg: Dict) -> InitialValues:
    phys = cfg.get("physics", {})
    return InitialValues(
        acid0=_safe_first(exp.acid, 0.0),
        resistivity0=max(_safe_first(exp.resistivity, 1.0), 1e-12),
        loss_factor0=max(_safe_first(exp.loss_factor, 1.0), 1e-12),
        bdv0=max(_safe_first(exp.bdv, 1.0), 1e-12),
        dp0=max(_safe_first(exp.dp, float(phys.get("DP0_default", 1111.0))), 1.0),
        viscosity0=_safe_first(exp.viscosity, np.nan) if exp.viscosity.size else np.nan,
    )


def _extract_modality(group: pd.DataFrame, prefix: str) -> Optional[np.ndarray]:
    cols = _sorted_prefixed_columns(group, prefix)
    if not cols:
        return None
    arr = group[cols].to_numpy(dtype=float)
    if np.isfinite(arr).sum() == 0:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _extract_dsc(group: pd.DataFrame, cfg: Dict) -> Optional[np.ndarray]:
    dsc_cols_cfg = cfg.get("columns", {}).get("dsc_columns", [])
    # 列名经 _standardize_column_name 后多为小写；配置里可能写 dsc_delta_H，需不区分大小写匹配
    lower_to_actual = {str(c).lower(): c for c in group.columns}
    cols = []
    for c in dsc_cols_cfg:
        act = lower_to_actual.get(str(c).lower())
        if act is not None:
            cols.append(act)
    if not cols:
        cols = [c for c in group.columns if str(c).startswith("dsc_")]
    if not cols:
        return None
    arr = group[cols].to_numpy(dtype=float)
    if np.isfinite(arr).sum() == 0:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _validate_experiment_modalities(exp: ExperimentCurve, cfg: Dict) -> None:
    """生产配置要求三模态 + DSC 均存在。"""
    if not bool(cfg.get("data", {}).get("require_all_modalities", True)):
        return
    name = exp.name
    for m in ["ftir", "raman", "uv"]:
        arr = getattr(exp, m)
        if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
            raise ValueError(f"实验 {name} 缺少必须输入模态: {m}")
    if exp.dsc_features is None or not isinstance(exp.dsc_features, np.ndarray) or exp.dsc_features.size == 0:
        raise ValueError(f"实验 {name} 缺少必须输入: DSC 热特征")


def load_experiments_from_excel(path: str, cfg: Dict) -> List[ExperimentCurve]:
    df = _prepare_dataframe(_read_excel_or_csv(path, cfg.get("data", {}).get("sheet_name")), cfg)
    cp = cfg.get("columns", {})
    exps: List[ExperimentCurve] = []
    for (oil, temp), group in df.groupby(["oil_type", "temperature_C"], sort=True):
        group = group.sort_values("aging_time_day")
        exp = ExperimentCurve(
            oil_type=str(oil), temperature_C=float(temp),
            t_days=group["aging_time_day"].to_numpy(dtype=float),
            acid=group["acid"].to_numpy(dtype=float),
            resistivity=group["resistivity"].to_numpy(dtype=float),
            loss_factor=group["loss_factor"].to_numpy(dtype=float),
            bdv=group["bdv"].to_numpy(dtype=float),
            dp=group["dp"].to_numpy(dtype=float),
            viscosity=group["viscosity"].to_numpy(dtype=float) if "viscosity" in group else np.array([], dtype=float),
            ftir=_extract_modality(group, cp.get("ftir_prefix", "ftir_")),
            raman=_extract_modality(group, cp.get("raman_prefix", "raman_")),
            uv=_extract_modality(group, cp.get("uv_prefix", "uv_")),
            dsc_features=_extract_dsc(group, cfg),
            meta={"replicate_count": group.get("replicate_count", pd.Series([1]*len(group))).to_numpy().tolist()},
        )
        exp.initial = infer_initial_values(exp, cfg)
        _validate_experiment_modalities(exp, cfg)
        exps.append(exp)
    if not exps:
        raise ValueError("未从 Excel/CSV 中读取到有效实验轨迹。")
    return exps


def generate_demo_experiments(cfg: Dict) -> List[ExperimentCurve]:
    rng = np.random.default_rng(int(cfg.get("training", {}).get("seed", 42)))
    oils = ["FR3", "SE"]
    temps = [110.0, 120.0, 130.0]
    t_days = np.array([0, 4, 7, 15, 22, 35, 60, 120, 180], dtype=float)
    x_ftir = np.linspace(650, 4000, 192)
    x_raman = np.linspace(200, 3200, 160)
    x_uv = np.linspace(200, 700, 96)
    exps: List[ExperimentCurve] = []

    for oil in oils:
        oil_factor = 0.78 if oil == "FR3" else 1.05
        dp0 = 1111.0
        acid0 = 0.008 if oil == "FR3" else 0.012
        rho0 = 1.1e12 if oil == "FR3" else 9.2e11
        tan0 = 0.0028 if oil == "FR3" else 0.0032
        bdv0 = 78.0 if oil == "FR3" else 72.0
        visc0 = 34.0 if oil == "FR3" else 27.0
        dsc_initial = np.array([
            238.0 if oil == "FR3" else 224.0,
            270.0 if oil == "FR3" else 255.0,
            41.0 if oil == "FR3" else 36.0,
            68.0 if oil == "FR3" else 55.0,
        ], dtype=np.float32)
        for T in temps:
            accel = np.exp((T - 110.0) / 17.5)
            td = t_days.copy()
            th = td * 24.0
            stage_center = 78.0 / accel * (1.08 if oil == "FR3" else 0.92)
            gate = 1.0 / (1.0 + np.exp(-(td - stage_center) / max(8.0, 18.0 / accel)))
            rate = (0.0019 * accel * oil_factor) * (1 + 1.2 * gate)
            acid = acid0 + 0.55 * (1 - np.exp(-rate * td)) + rng.normal(0, 0.004, len(td))
            acid = np.maximum(acid, acid0 * 0.7)
            dp_rate = 1.5e-4 * accel * oil_factor * (1 + 1.8 * acid)
            dp_loss = np.cumsum(np.r_[0, 0.5 * (dp_rate[1:] + dp_rate[:-1]) * np.diff(td)])
            dp = dp0 * np.exp(-dp_loss) * (1 + rng.normal(0, 0.009, len(td)))
            rho = rho0 * np.exp(-(0.0026 * accel * oil_factor) * td * (1 + 0.6 * acid)) * (1 + rng.normal(0, 0.025, len(td)))
            tan = tan0 * np.exp((0.0035 * accel * oil_factor) * td * (1 + 0.9 * acid)) * (1 + rng.normal(0, 0.025, len(td)))
            bdv = bdv0 * np.exp(-(0.0018 * accel * oil_factor) * td * (1 + 1.5 * acid)) * (1 + rng.normal(0, 0.015, len(td)))
            visc = visc0 * (1 + 0.28 * acid + 0.0005 * td)

            aging_index = (1 - dp / dp0) + 0.7 * (acid - acid0) + 0.45 * np.log(tan / tan0)
            def gauss(x, mu, sig): return np.exp(-0.5 * ((x - mu) / sig) ** 2)
            ftir = []
            raman = []
            uv = []
            for ai, a, d in zip(aging_index, acid, dp):
                ft = (0.2 + 0.00002 * (x_ftir - 650)
                      + (0.75 + 0.8 * ai) * gauss(x_ftir, 1740, 45)
                      + (0.25 + 1.1 * a) * gauss(x_ftir, 3400, 120)
                      + 0.15 * gauss(x_ftir, 1160, 60)
                      + rng.normal(0, 0.006, len(x_ftir)))
                rm = (0.15 + 0.00002 * x_raman
                      + (0.45 + 0.5 * ai) * gauss(x_raman, 1440, 50)
                      + (0.25 + 0.45 * (dp0 - d) / dp0) * gauss(x_raman, 1120, 40)
                      + 0.18 * gauss(x_raman, 2850, 80)
                      + rng.normal(0, 0.007, len(x_raman)))
                uvv = (0.05 + (0.25 + 0.9 * ai) * np.exp(-((x_uv - 280) / 80) ** 2)
                       + 0.18 * a * np.exp(-((x_uv - 360) / 95) ** 2)
                       + rng.normal(0, 0.004, len(x_uv)))
                ftir.append(ft.astype(np.float32)); raman.append(rm.astype(np.float32)); uv.append(uvv.astype(np.float32))
            dsc = np.repeat(dsc_initial[None, :], len(td), axis=0)
            # 老化后 DSC onset 轻微下降，便于代码验证动态 DSC 也可用
            dsc[:, 0] -= 0.012 * td * accel
            dsc[:, 1] -= 0.010 * td * accel
            exp = ExperimentCurve(
                oil, T, td, acid, rho, tan, bdv, dp,
                viscosity=visc,
                ftir=np.stack(ftir), raman=np.stack(raman), uv=np.stack(uv),
                dsc_features=dsc.astype(np.float32),
            )
            exp.initial = infer_initial_values(exp, cfg)
            _validate_experiment_modalities(exp, cfg)
            exps.append(exp)
    return exps


def load_experiments(cfg: Dict) -> List[ExperimentCurve]:
    path = cfg.get("data", {}).get("excel_path")
    if path:
        return load_experiments_from_excel(path, cfg)
    if cfg.get("data", {}).get("use_demo_data_if_excel_missing", True):
        return generate_demo_experiments(cfg)
    raise ValueError("config.data.excel_path 为空，且未允许 demo 数据。")


def split_by_temperature(experiments: Sequence[ExperimentCurve], cfg: Dict):
    data_cfg = cfg.get("data", {})
    train_t = set(float(x) for x in data_cfg.get("train_temperatures_C", [110, 120]))
    val_t = set(float(x) for x in data_cfg.get("validation_temperatures_C", []))
    test_t = set(float(x) for x in data_cfg.get("test_temperatures_C", [130]))
    train, val, test = [], [], []
    for i, exp in enumerate(experiments):
        T = float(exp.temperature_C)
        if T in test_t:
            test.append(i)
        elif T in val_t:
            val.append(i)
        elif T in train_t:
            train.append(i)
    if not train:
        raise ValueError("训练集为空，请检查 train_temperatures_C。")
    if not test:
        raise ValueError("测试集为空，请检查 test_temperatures_C。")
    return train, val, test


def make_internal_validation(train_idx: Sequence[int], seed: int = 42) -> Tuple[List[int], List[int]]:
    train_idx = list(train_idx)
    if len(train_idx) < 3:
        return train_idx, []
    rng = np.random.default_rng(seed)
    idx = np.array(train_idx)
    rng.shuffle(idx)
    val_n = max(1, int(round(0.2 * len(idx))))
    return idx[val_n:].tolist(), idx[:val_n].tolist()


def split_leave_one_oil(experiments: Sequence[ExperimentCurve], oil: str) -> Tuple[List[int], List[int]]:
    train = [i for i, e in enumerate(experiments) if e.oil_type != oil]
    test = [i for i, e in enumerate(experiments) if e.oil_type == oil]
    return train, test


def determine_acid_saturation(experiments: Sequence[ExperimentCurve], train_indices: Sequence[int], cfg: Dict) -> float:
    phys = cfg.get("physics", {})
    mode = phys.get("acid_saturation_mode", "train_quantile")
    values = np.concatenate([np.asarray(experiments[i].acid, dtype=float) for i in train_indices])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(phys.get("acid_saturation_min", 0.05))
    if mode == "fixed":
        return float(phys.get("acid_saturation_fixed", 0.5))
    if mode == "train_max_factor":
        base = float(np.max(values))
    else:
        q = float(phys.get("acid_saturation_quantile", 0.95))
        base = float(np.quantile(values, np.clip(q, 0.5, 1.0)))
    return max(float(phys.get("acid_saturation_min", 0.05)), base * float(phys.get("acid_saturation_factor", 1.35)))


def modality_lengths(experiments: Sequence[ExperimentCurve], indices: Sequence[int]) -> Dict[str, int]:
    out = {"ftir": 0, "raman": 0, "uv": 0, "dsc": 0}
    for i in indices:
        e = experiments[i]
        for m in ["ftir", "raman", "uv"]:
            arr = getattr(e, m)
            if arr is not None and arr.ndim == 2 and arr.shape[1] > 0:
                out[m] = max(out[m], int(arr.shape[1]))
        if e.dsc_features is not None and e.dsc_features.ndim == 2:
            out["dsc"] = max(out["dsc"], int(e.dsc_features.shape[1]))
    return out
