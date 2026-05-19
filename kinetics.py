# -*- coding: utf-8 -*-
"""
kinetics.py
PMTA-Net v4 物理约束与寿命判定层。

v4 不再把动力学方程作为 theta 伪标签生成器，而是把它们作为直接作用于
预测曲线 y_hat(t) 的 physics loss。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

TARGET_NAMES = ["acid", "resistivity", "loss_factor", "bdv", "dp"]
R_GAS = 8.314e-3  # kJ/(mol K)


def first_crossing_time(t_days: np.ndarray, y: np.ndarray, threshold: float, mode: str) -> Optional[float]:
    t = np.asarray(t_days, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if t.size == 0:
        return None
    if mode == "above":
        hit = y >= threshold
    elif mode == "below":
        hit = y <= threshold
    else:
        raise ValueError("mode 必须为 above 或 below")
    if not np.any(hit):
        return None
    idx = int(np.argmax(hit))
    if idx == 0:
        return float(t[0])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    t0, t1 = float(t[idx - 1]), float(t[idx])
    if abs(y1 - y0) < 1e-12:
        return t1
    frac = np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0)
    return float(t0 + frac * (t1 - t0))


def judge_thermal_grade(t_days: np.ndarray, curves: Dict[str, np.ndarray], config: Dict) -> Dict[str, object]:
    phys = config.get("physics", {})
    grade_cfg = config.get("thermal_grade", {})
    acid_th = float(phys.get("acid_threshold", 0.2))
    dp_th = float(phys.get("dp_failure_threshold", 200.0))
    bdv_rel = float(phys.get("bdv_relative_threshold", 0.7))
    tan_rel = float(phys.get("loss_factor_relative_threshold", 5.0))
    rho_rel = float(phys.get("resistivity_relative_threshold", 0.2))

    out = {
        "acid_exceed_days": first_crossing_time(t_days, curves["acid"], acid_th, "above"),
        "dp_failure_days": first_crossing_time(t_days, curves["dp"], dp_th, "below"),
        "bdv_failure_days": first_crossing_time(t_days, curves["bdv"], bdv_rel * max(float(curves["bdv"][0]), 1e-12), "below"),
        "loss_factor_failure_days": first_crossing_time(t_days, curves["loss_factor"], tan_rel * max(float(curves["loss_factor"][0]), 1e-12), "above"),
        "resistivity_failure_days": first_crossing_time(t_days, curves["resistivity"], rho_rel * max(float(curves["resistivity"][0]), 1e-12), "below"),
    }
    finite = [v for v in out.values() if v is not None and np.isfinite(v)]
    conservative = min(finite) if finite else np.inf
    A = float(grade_cfg.get("life_days_grade_A", 360.0))
    B = float(grade_cfg.get("life_days_grade_B", 240.0))
    C = float(grade_cfg.get("life_days_grade_C", 120.0))
    if not np.isfinite(conservative) or conservative >= A:
        grade = "A"
    elif conservative >= B:
        grade = "B"
    elif conservative >= C:
        grade = "C"
    else:
        grade = "D"
    out["conservative_life_days"] = None if not np.isfinite(conservative) else float(conservative)
    out["thermal_grade"] = grade
    return out


def fit_lifetime_surface(records: Sequence[Dict]) -> Dict[str, float]:
    """由多温度失效时间拟合 log(L)=a+b/T_K。"""
    T, L = [], []
    for r in records:
        life = r.get("conservative_life_days")
        temp = r.get("temperature_C")
        if life is not None and np.isfinite(life) and life > 0 and temp is not None:
            T.append(float(temp) + 273.15)
            L.append(float(life))
    if len(T) < 2 or len(set(round(v, 6) for v in T)) < 2:
        return {"a": np.nan, "b": np.nan, "n": len(T)}
    x = 1.0 / np.asarray(T)
    y = np.log(np.asarray(L))
    b, a = np.polyfit(x, y, deg=1)
    return {"a": float(a), "b": float(b), "n": len(T)}


def empirical_transition_time(t_days: np.ndarray, curves: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """由综合老化指数的二阶差分估计阶段转换点。"""
    t = np.asarray(t_days, dtype=float)
    y = np.asarray(curves, dtype=float)  # (L,5) acid,rho,tan,bdv,dp
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        t, y = t[m], y[m]
    if len(t) < 4:
        return float(np.median(t)) if len(t) else 0.0
    acid, rho, tan, bdv, dp = [y[:, i] for i in range(5)]
    eps = 1e-12
    ai = ((acid - acid[0]) / max(np.nanmax(acid) - acid[0], eps)
          + (np.log(np.maximum(tan, eps)) - np.log(max(tan[0], eps))) / max(np.nanmax(np.log(np.maximum(tan, eps))) - np.log(max(tan[0], eps)), eps)
          + (dp[0] - dp) / max(dp[0] - np.nanmin(dp), eps)
          + (bdv[0] - bdv) / max(bdv[0] - np.nanmin(bdv), eps)) / 4.0
    # 用非均匀时间梯度更稳
    try:
        d1 = np.gradient(ai, t)
        d2 = np.gradient(d1, t)
        idx = int(np.nanargmax(np.abs(d2[1:-1])) + 1)
    except Exception:
        idx = len(t) // 2
    return float(t[idx])


# --------------------------- torch losses ---------------------------

def _masked_mean(x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
    if mask is None:
        return x.mean()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    m = mask.to(dtype=x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def finite_diff(y: "torch.Tensor", t_days: "torch.Tensor", mask: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """返回 dy/dt 和相邻点有效 mask。y(B,L,C), t_days(B,L)."""
    dy = y[:, 1:] - y[:, :-1]
    dt = (t_days[:, 1:] - t_days[:, :-1]).clamp(min=1e-6).unsqueeze(-1)
    valid = mask[:, 1:] & mask[:, :-1]
    return dy / dt, valid


def monotonicity_loss(y_hat: "torch.Tensor", t_days: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """DP↓/BDV↓/rho↓/acid↑/tan↑ 软单调约束。"""
    dy, valid = finite_diff(y_hat, t_days, mask)
    acid_d = dy[..., 0]
    rho_d = dy[..., 1]
    tan_d = dy[..., 2]
    bdv_d = dy[..., 3]
    dp_d = dy[..., 4]
    losses = [
        F.relu(-acid_d).pow(2),
        F.relu(rho_d).pow(2),
        F.relu(-tan_d).pow(2),
        F.relu(bdv_d).pow(2),
        F.relu(dp_d).pow(2),
    ]
    return sum(_masked_mean(v, valid) for v in losses) / len(losses)


def _ode_residual_loss(res: "torch.Tensor", valid: "torch.Tensor", huber_beta: Optional[float]) -> "torch.Tensor":
    """相对残差上的损失：默认可平方；huber_beta>0 时用 smooth_l1 抑制大残差主导梯度。"""
    if huber_beta is None or float(huber_beta) <= 0.0:
        return _masked_mean(res.pow(2), valid)
    beta = float(huber_beta)
    elem = F.smooth_l1_loss(res, torch.zeros_like(res), beta=beta, reduction="none")
    return _masked_mean(elem, valid)


def dp_ode_loss(
    y_hat: "torch.Tensor",
    rates: "torch.Tensor",
    t_days: "torch.Tensor",
    mask: "torch.Tensor",
    huber_beta: Optional[float] = None,
) -> "torch.Tensor":
    """Emsley 主链: d(1/DP)/dt = k_dp * (1 + cA*A)。不引入水分变量。"""
    dp = y_hat[..., 4].clamp(min=30.0)
    inv_dp = 1.0 / dp
    dinv, valid = finite_diff(inv_dp.unsqueeze(-1), t_days, mask)
    acid_mid = 0.5 * (y_hat[:, 1:, 0] + y_hat[:, :-1, 0]).clamp(min=0.0)
    kdp = rates["k_dp"]
    if kdp.dim() == 2:
        kdp_mid = 0.5 * (kdp[:, 1:] + kdp[:, :-1])
    else:
        kdp_mid = kdp[:, None].expand_as(acid_mid)
    rhs = kdp_mid.clamp(min=0.0) * (1.0 + 0.5 * acid_mid)
    # 尺度较小，使用相对残差
    res = (dinv.squeeze(-1) - rhs) / (rhs.detach().abs() + 1e-6)
    return _ode_residual_loss(res, valid, huber_beta)


def acid_ode_loss(
    y_hat: "torch.Tensor",
    rates: Dict[str, "torch.Tensor"],
    t_days: "torch.Tensor",
    mask: "torch.Tensor",
    acid_sat: float,
    huber_beta: Optional[float] = None,
) -> "torch.Tensor":
    acid = y_hat[..., 0].clamp(min=0.0)
    dacid, valid = finite_diff(acid.unsqueeze(-1), t_days, mask)
    a_mid = 0.5 * (acid[:, 1:] + acid[:, :-1])
    ka = rates["k_acid"]
    if ka.dim() == 2:
        ka_mid = 0.5 * (ka[:, 1:] + ka[:, :-1])
    else:
        ka_mid = ka[:, None].expand_as(a_mid)
    sat = torch.as_tensor(float(acid_sat), dtype=acid.dtype, device=acid.device)
    rhs = ka_mid.clamp(min=0.0) * (a_mid + 1e-3) * (1.0 - a_mid / sat.clamp(min=1e-4)).clamp(min=0.0)
    res = (dacid.squeeze(-1) - rhs) / (rhs.detach().abs() + 1e-4)
    return _ode_residual_loss(res, valid, huber_beta)



def coupling_rank_loss(y_hat: "torch.Tensor", t_days: "torch.Tensor", mask: "torch.Tensor", margin: float = 0.0) -> "torch.Tensor":
    """酸值/介损上升与 DP/BDV 下降的排序耦合。"""
    # 相邻点 rank 足够稳定，避免 O(n^2) 在小样本上过拟合
    y0, y1 = y_hat[:, :-1], y_hat[:, 1:]
    valid = mask[:, 1:] & mask[:, :-1]
    d_acid = y1[..., 0] - y0[..., 0]
    d_tan = y1[..., 2] - y0[..., 2]
    drop_dp = y0[..., 4] - y1[..., 4]
    drop_bdv = y0[..., 3] - y1[..., 3]
    # 当前项为正表示趋势一致；小于 margin 惩罚
    l1 = F.relu(margin - d_acid * drop_dp).pow(2)
    l2 = F.relu(margin - d_tan * drop_bdv).pow(2)
    l3 = F.relu(margin - d_acid * drop_bdv).pow(2)
    return (_masked_mean(l1, valid) + _masked_mean(l2, valid) + _masked_mean(l3, valid)) / 3.0


def dsc_anchor_loss(tc_pred: "torch.Tensor", tc_target: "torch.Tensor", has_dsc: "torch.Tensor") -> "torch.Tensor":
    if tc_pred is None or tc_target is None:
        return torch.as_tensor(0.0, device=has_dsc.device)
    loss = (tc_pred - tc_target).pow(2)
    return _masked_mean(loss, has_dsc.bool())


def spectral_consistency_loss(spec_ai: "torch.Tensor", y_true: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """让光谱分支学习综合老化指数。"""
    eps = 1e-8
    acid = y_true[..., 0]
    tan = y_true[..., 2].clamp(min=eps)
    bdv = y_true[..., 3]
    dp = y_true[..., 4]
    acid0 = acid[:, :1]
    tan0 = tan[:, :1]
    bdv0 = bdv[:, :1].clamp(min=eps)
    dp0 = dp[:, :1].clamp(min=eps)
    ai = (F.relu(acid - acid0) / (acid0.abs() + 0.05)
          + F.relu(torch.log(tan / tan0))
          + F.relu((bdv0 - bdv) / bdv0)
          + F.relu((dp0 - dp) / dp0)) / 4.0
    return _masked_mean((spec_ai.squeeze(-1) - ai).pow(2), mask)


def initial_anchor_loss(y_hat: "torch.Tensor", y_true: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """t=0 锚定，避免连续时间解码器起点漂移。"""
    valid0 = mask[:, 0]
    if valid0.sum() == 0:
        return torch.as_tensor(0.0, device=y_hat.device)
    denom = y_true[:, 0].abs().clamp(min=torch.tensor([0.05, 1e10, 1e-3, 1.0, 100.0], device=y_hat.device, dtype=y_hat.dtype))
    rel = (y_hat[:, 0] - y_true[:, 0]) / denom
    return _masked_mean(rel.pow(2), valid0)


def arrhenius_order_loss(y_hat: "torch.Tensor", T_K: "torch.Tensor", oil_ids: Sequence[str], mask: "torch.Tensor") -> "torch.Tensor":
    """同油种高温曲线应具有更大的平均老化指数。"""
    device = y_hat.device
    losses = []
    ai = aging_index_torch(y_hat, mask)  # (B,)
    for i in range(len(oil_ids)):
        for j in range(len(oil_ids)):
            if oil_ids[i] == oil_ids[j] and float(T_K[i].detach().cpu()) > float(T_K[j].detach().cpu()) + 1e-6:
                losses.append(F.relu(ai[j] - ai[i]).pow(2))
    if not losses:
        return torch.as_tensor(0.0, device=device)
    return torch.stack(losses).mean()


def aging_index_torch(y: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    eps = 1e-8
    acid = y[..., 0]
    tan = y[..., 2].clamp(min=eps)
    bdv = y[..., 3]
    dp = y[..., 4]
    acid0 = acid[:, :1]
    tan0 = tan[:, :1]
    bdv0 = bdv[:, :1].clamp(min=eps)
    dp0 = dp[:, :1].clamp(min=eps)
    ai_seq = (F.relu(acid - acid0) / (acid0.abs() + 0.05)
              + F.relu(torch.log(tan / tan0))
              + F.relu((bdv0 - bdv) / bdv0)
              + F.relu((dp0 - dp) / dp0)) / 4.0
    m = mask.to(y.dtype)
    return (ai_seq * m).sum(1) / m.sum(1).clamp(min=1.0)


def aging_index_from_states(
    acid: "torch.Tensor",
    tan: "torch.Tensor",
    bdv: "torch.Tensor",
    dp: "torch.Tensor",
    acid0: "torch.Tensor",
    tan0: "torch.Tensor",
    bdv0: "torch.Tensor",
    dp0: "torch.Tensor",
) -> "torch.Tensor":
    """逐时刻综合老化指数 (B,L)。"""
    eps = 1e-8
    return (
        F.relu(acid - acid0) / (acid0.abs() + 0.05)
        + F.relu(torch.log(tan.clamp(min=eps) / tan0.clamp(min=eps)))
        + F.relu((bdv0 - bdv) / bdv0.clamp(min=eps))
        + F.relu((dp0 - dp) / dp0.clamp(min=eps))
    ) / 4.0


def params_to_rates(params: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
    """params (B,L,5): log_k_acid, log_k_dp, alpha_rho, alpha_tan, alpha_bdv。"""
    return {
        "k_acid": F.softplus(params[..., 0]) * 1e-2 + 1e-6,
        "k_dp": F.softplus(params[..., 1]) * 1e-5 + 1e-8,
        "alpha_rho": F.softplus(params[..., 2]) * 0.5 + 1e-4,
        "alpha_tan": F.softplus(params[..., 3]) * 0.5 + 1e-4,
        "alpha_bdv": F.softplus(params[..., 4]) * 0.5 + 1e-4,
    }


def physics_align_loss(
    y_data: "torch.Tensor",
    y_phys: "torch.Tensor",
    mask: "torch.Tensor",
    target_weights: Optional["torch.Tensor"] = None,
    scale: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """数据主路径与物理投影路径对齐（相对尺度，避免 ρ 量纲主导）。"""
    if scale is None:
        scale = torch.tensor([0.05, 1e10, 1e-3, 1.0, 100.0], device=y_data.device, dtype=y_data.dtype)
    else:
        scale = scale.to(device=y_data.device, dtype=y_data.dtype)
    denom = scale.view(1, 1, -1).clamp(min=1e-8)
    res = (y_data - y_phys) / denom
    diff = res.pow(2)
    if target_weights is not None:
        w = target_weights.view(1, 1, -1).to(device=diff.device, dtype=diff.dtype)
        diff = diff * w
    return _masked_mean(diff, mask)


def integrate_trajectory_physics(
    init: "torch.Tensor",
    t_days: "torch.Tensor",
    mask: "torch.Tensor",
    params: "torch.Tensor",
    acid_sat: float,
    k_scale_acid: Optional["torch.Tensor"] = None,
    k_scale_dp: Optional["torch.Tensor"] = None,
    teacher_target: Optional["torch.Tensor"] = None,
    teacher_forcing_weight: float = 0.0,
) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
    """可微分 Euler 积分：酸/DP ODE + ρ/tanδ/BDV 由老化指数代数耦合。唯一主输出路径。"""
    B, L = t_days.shape
    device, dtype = init.device, init.dtype
    acid0 = init[:, 0].clamp(min=0.0)
    rho0 = init[:, 1].clamp(min=1e-12)
    tan0 = init[:, 2].clamp(min=1e-12)
    bdv0 = init[:, 3].clamp(min=1e-12)
    dp0 = init[:, 4].clamp(min=1.0)

    rates = params_to_rates(params)
    ar, atan, abdv = rates["alpha_rho"], rates["alpha_tan"], rates["alpha_bdv"]
    sat = torch.as_tensor(float(acid_sat), device=device, dtype=dtype).clamp(min=1e-4)

    acid_rows, inv_rows, rho_rows, tan_rows, bdv_rows, dp_rows = [], [], [], [], [], []
    w_tf = float(max(0.0, min(1.0, teacher_forcing_weight)))
    use_tf = teacher_target is not None and w_tf > 0.0

    for b in range(B):
        valid_idx = torch.where(mask[b])[0]
        if valid_idx.numel() == 0:
            z = torch.zeros(L, device=device, dtype=dtype)
            acid_rows.append(z)
            inv_rows.append(z.clone())
            rho_rows.append(rho0[b].expand(L).clone())
            tan_rows.append(tan0[b].expand(L).clone())
            bdv_rows.append(bdv0[b].expand(L).clone())
            dp_rows.append(dp0[b].expand(L).clone())
            continue
        idx_list = valid_idx.tolist()
        acid_nodes = [acid0[b]]
        inv_nodes = [1.0 / dp0[b].clamp(min=30.0)]
        for j in range(len(idx_list) - 1):
            i, inext = int(idx_list[j]), int(idx_list[j + 1])
            dt = (t_days[b, inext] - t_days[b, i]).clamp(min=1e-6)
            a_i = acid_nodes[-1]
            if use_tf:
                a_drive = (1.0 - w_tf) * a_i + w_tf * teacher_target[b, i, 0]
            else:
                a_drive = a_i
            ka = rates["k_acid"][b, i]
            kd = rates["k_dp"][b, i]
            if k_scale_acid is not None:
                ka = ka * k_scale_acid[b]
            if k_scale_dp is not None:
                kd = kd * k_scale_dp[b]
            rhs_a = ka * (a_drive + 1e-3) * (1.0 - a_drive / sat).clamp(min=0.0)
            acid_nodes.append((a_i + dt * rhs_a).clamp(min=0.0))
            inv_nodes.append(inv_nodes[-1] + dt * kd * (1.0 + 0.5 * a_drive))
        idx_t = torch.tensor(idx_list, device=device, dtype=torch.long)
        acid_b = torch.zeros(L, device=device, dtype=dtype)
        inv_b = torch.zeros(L, device=device, dtype=dtype)
        acid_b = acid_b.index_copy(0, idx_t, torch.stack(acid_nodes))
        inv_b = inv_b.index_copy(0, idx_t, torch.stack(inv_nodes))
        dp_b = (1.0 / inv_b.clamp(min=1.0 / 2000.0)).clamp(min=30.0)
        a0v, d0v = acid0[b], dp0[b]
        ai = (
            F.relu(acid_b - a0v) / (a0v.abs() + 0.05)
            + F.relu((d0v - dp_b) / d0v.clamp(min=1.0))
        ) / 2.0
        rho_b = rho0[b] * torch.exp(-ar[b] * ai)
        tan_b = tan0[b] * torch.exp(atan[b] * ai)
        bdv_b = bdv0[b] * torch.exp(-abdv[b] * ai)
        last_v = int(idx_list[-1])
        pad = torch.arange(L, device=device)
        invalid = ~mask[b]
        if invalid.any():
            acid_b = torch.where(invalid, acid_b[last_v], acid_b)
            inv_b = torch.where(invalid, inv_b[last_v], inv_b)
            dp_b = torch.where(invalid, dp_b[last_v], dp_b)
            rho_b = torch.where(invalid, rho_b[last_v], rho_b)
            tan_b = torch.where(invalid, tan_b[last_v], tan_b)
            bdv_b = torch.where(invalid, bdv_b[last_v], bdv_b)
        acid_rows.append(acid_b)
        inv_rows.append(inv_b)
        rho_rows.append(rho_b)
        tan_rows.append(tan_b)
        bdv_rows.append(bdv_b)
        dp_rows.append(dp_b)

    acid = torch.stack(acid_rows, dim=0)
    rho = torch.stack(rho_rows, dim=0)
    tan = torch.stack(tan_rows, dim=0)
    bdv = torch.stack(bdv_rows, dim=0)
    dp = torch.stack(dp_rows, dim=0)
    y_hat = torch.stack([acid, rho, tan, bdv, dp], dim=-1)
    return y_hat, rates


# 兼容旧名
integrate_trajectory_v42 = integrate_trajectory_physics
