# -*- coding: utf-8 -*-
"""
error_spatial.py — 4.5.1 预测误差的空间分布分析

为每个传入的 config（传导/匙孔模式各一），从测试集中挑选一个代表性样本，
绘制 X-Z 中心切片下的逐点绝对误差热力图（温度 + VOF 两通道），并叠加
α_air = 0.5 的熔池/匙腔界面与 T = 800 K 的 focus mask 边界。

输出：paper/code5/figures/4_5_1/error_spatial_<name>.png/.pdf

学术风格：
  - 4 子图布局：[T_GT | T_err | α_GT | α_err]，单位标注完整；
  - 等值线：黑实线 = GT 界面，红虚线 = 预测界面，白虚线 = focus mask 边界；
  - 配色：温度 inferno，VOF viridis，误差 magma；
  - 字体统一 DejaVu Serif 10pt；分辨率 400 dpi（满足学术出版）。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from predictor import (load_predictor, per_sample_metrics, rollout_sample,
                       sample_idx_to_h5, get_condition_params)


# ----------------------------------------------------------------------------- #
# 学术风格 rcParams
# ----------------------------------------------------------------------------- #
def _setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ----------------------------------------------------------------------------- #
# 中心切片插值
# ----------------------------------------------------------------------------- #
def _center_slice_interp(coords: np.ndarray,
                         values: np.ndarray,
                         axis: str = "y",
                         res: int = 256,
                         thickness_ratio: float = 0.04):
    """沿给定轴在中心位置取一薄层节点，将其投影到 (其它两轴) 平面再做 linear 插值。"""
    axis_id = {"x": 0, "y": 1, "z": 2}[axis]
    ax_vals = coords[:, axis_id]
    center = 0.5 * (ax_vals.min() + ax_vals.max())
    span = max(ax_vals.max() - ax_vals.min(), 1e-12)
    thickness = thickness_ratio * span

    mask = np.abs(ax_vals - center) <= thickness
    # 兜底：若节点过少，用最近邻 slab
    if mask.sum() < 32:
        order = np.argsort(np.abs(ax_vals - center))[: max(32, len(ax_vals) // 30)]
        mask = np.zeros_like(ax_vals, dtype=bool)
        mask[order] = True

    other = [i for i in (0, 1, 2) if i != axis_id]
    px, py = coords[mask, other[0]], coords[mask, other[1]]
    vals = values[mask]

    xi = np.linspace(px.min(), px.max(), res)
    yi = np.linspace(py.min(), py.max(), res)
    Xi, Yi = np.meshgrid(xi, yi)
    Z = griddata(np.column_stack([px, py]), vals, (Xi, Yi), method="linear")
    if Z is None or np.isnan(Z).all():
        Z = griddata(np.column_stack([px, py]), vals, (Xi, Yi), method="nearest")
    if np.isnan(Z).any():
        Z_n = griddata(np.column_stack([px, py]), vals, (Xi, Yi), method="nearest")
        Z = np.where(np.isnan(Z), Z_n, Z)
    axis_labels = {0: "X (m)", 1: "Y (m)", 2: "Z (m)"}
    return Xi, Yi, Z, axis_labels[other[0]], axis_labels[other[1]]


# ----------------------------------------------------------------------------- #
# 选样本：选 mean rL2 接近中位数的样本作为 "代表性工况"
# ----------------------------------------------------------------------------- #
def _select_representative(predictor, max_scan: int = 30, seed: int = 42) -> int:
    """在测试集中扫描若干样本，选取 mean rL2 最接近中位数者作为代表样本。"""
    rng = np.random.default_rng(seed)
    n = len(predictor.dataset)
    cand = list(range(n)) if n <= max_scan else rng.choice(n, max_scan, replace=False).tolist()
    rl2_list = []
    for i in cand:
        m = per_sample_metrics(predictor, i)
        rl2_list.append((i, m["rl2_overall"]))
    rl2_arr = np.array([r for _, r in rl2_list])
    median = np.median(rl2_arr)
    pick = min(rl2_list, key=lambda x: abs(x[1] - median))
    return pick[0]


# ----------------------------------------------------------------------------- #
# 数据准备（公共逻辑，供单图和合并图复用）
# ----------------------------------------------------------------------------- #
def _prepare_data(config_path: str,
                  sample_idx: Optional[int],
                  time_step: int,
                  slice_axis: str,
                  res: int):
    """加载模型、选取样本、计算切片插值，返回绘图所需的全部数据字典。"""
    predictor = load_predictor(config_path)
    fields = list(predictor.fields)
    name = predictor.args.name

    # if sample_idx is None:
    #     sample_idx = _select_representative(predictor)



    h5_path = sample_idx_to_h5(predictor, sample_idx)
    cond = get_condition_params(h5_path)

    out = rollout_sample(predictor, sample_idx)
    pred = out["pred"]; gt = out["gt"]; coords = out["coords"]
    horizon = pred.shape[0]
    t = min(time_step, horizon - 1)

    if "T" not in fields or "alpha.air" not in fields:
        raise RuntimeError(f"4.5.1 需要 T 与 alpha.air 同时存在，当前 fields={fields}")
    iT = fields.index("T"); iA = fields.index("alpha.air")

    T_gt, T_pr = gt[t, :, iT], pred[t, :, iT]
    A_gt, A_pr = gt[t, :, iA], pred[t, :, iA]

    Xi, Yi, ZT_gt, xlab, ylab = _center_slice_interp(coords, T_gt, slice_axis, res)
    _, _, ZT_pr, _, _ = _center_slice_interp(coords, T_pr, slice_axis, res)
    _, _, ZA_gt, _, _ = _center_slice_interp(coords, A_gt, slice_axis, res)
    _, _, ZA_pr, _, _ = _center_slice_interp(coords, A_pr, slice_axis, res)

    ZT_err = np.abs(ZT_pr - ZT_gt)
    ZA_err = np.abs(ZA_pr - ZA_gt)

    metrics = per_sample_metrics(predictor, sample_idx)

    return dict(
        Xi=Xi, Yi=Yi, ZT_gt=ZT_gt, ZT_pr=ZT_pr, ZA_gt=ZA_gt, ZA_pr=ZA_pr,
        ZT_err=ZT_err, ZA_err=ZA_err, xlab=xlab, ylab=ylab,
        name=name, sample_idx=sample_idx, t=t, cond=cond, metrics=metrics,
    )


# ----------------------------------------------------------------------------- #
# 在给定的 2×2 axes 上绘制单个模式的子图（供单图和合并图共用）
# ----------------------------------------------------------------------------- #
def _draw_mode_on_axes(fig, axes, data: dict, mode_label: str,
                       label_prefix: str = "", show_ylabel: bool = True):
    """
    在 2×2 的 axes 上绘制温度/VOF 真值 + 误差热力图。
    label_prefix: 子图标号前缀，例如 "(a)" "(e)" 等。
    """
    Xi, Yi = data["Xi"], data["Yi"]
    ZT_gt, ZA_gt = data["ZT_gt"], data["ZA_gt"]
    ZA_pr, ZT_err, ZA_err = data["ZA_pr"], data["ZT_err"], data["ZA_err"]
    xlab, ylab = data["xlab"], data["ylab"]

    extent = (Xi.min(), Xi.max(), Yi.min(), Yi.max())
    imshow_kw = dict(extent=extent, origin="lower", aspect="equal", interpolation="bicubic")

    labels = [f"GT $T$ (K)", f"$|T_{{\\mathrm{{pred}}}}-T_{{\\mathrm{{gt}}}}|$ (K)",
              f"GT $\\alpha_{{\\mathrm{{air}}}}$", f"$|\\alpha_{{\\mathrm{{pred}}}}-\\alpha_{{\\mathrm{{gt}}}}|$"]
    if label_prefix:
        prefixes = [f"({chr(ord(label_prefix[1])+i)})" for i in range(4)]
    else:
        prefixes = ["(a)", "(b)", "(c)", "(d)"]

    # T GT
    im00 = axes[0, 0].imshow(ZT_gt, cmap="inferno", vmin=300, vmax=3500, **imshow_kw)
    axes[0, 0].set_title(f"{prefixes[0]} {labels[0]}")
    cb00 = fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.03,
                        shrink=0.85)
    cb00.set_label("$T$ (K)")

    # T error
    err_T_vmax = max(np.nanpercentile(ZT_err, 99.0), 1e-6)
    im01 = axes[0, 1].imshow(ZT_err, cmap="magma", vmin=0, vmax=err_T_vmax, **imshow_kw)
    axes[0, 1].set_title(f"{prefixes[1]} {labels[1]}")
    cb01 = fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.03,
                        shrink=0.85, extend="max")
    cb01.set_label("Abs. error (K)")

    # alpha GT
    im10 = axes[1, 0].imshow(np.clip(ZA_gt, 0, 1), cmap="viridis", vmin=0, vmax=1, **imshow_kw)
    axes[1, 0].set_title(f"{prefixes[2]} {labels[2]}")
    cb10 = fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.03,
                        shrink=0.85)
    cb10.set_label(r"$\alpha_{\mathrm{air}}$")

    # alpha error
    err_A_vmax = max(np.nanpercentile(ZA_err, 99.0), 1e-6)
    im11 = axes[1, 1].imshow(ZA_err, cmap="magma", vmin=0, vmax=err_A_vmax, **imshow_kw)
    axes[1, 1].set_title(f"{prefixes[3]} {labels[3]}")
    cb11 = fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.03,
                        shrink=0.85, extend="max")
    cb11.set_label("Abs. error")

    # 通过 T > 800 K 找到 x 方向的活跃区间，仅在该区间内绘制熔池/匙腔边界
    active_mask_2d = ZT_gt > 800.0
    if active_mask_2d.any():
        col_active = active_mask_2d.any(axis=0)
        x_vals_1d = Xi[0, :]
        x_active = x_vals_1d[col_active]
        x_pad = 0.02 * (x_vals_1d.max() - x_vals_1d.min())
        x_lo, x_hi = x_active.min() - x_pad, x_active.max() + x_pad
        # 构造 x 区间外置 NaN 的拷贝，以裁剪 contour
        x_keep = (Xi >= x_lo) & (Xi <= x_hi)
        ZA_gt_clip = np.where(x_keep, np.clip(ZA_gt, 0, 1), np.nan)
        ZA_pr_clip = np.where(x_keep, np.clip(ZA_pr, 0, 1), np.nan)
    else:
        ZA_gt_clip = np.clip(ZA_gt, 0, 1)
        ZA_pr_clip = np.clip(ZA_pr, 0, 1)

    # 叠加等值线（GT 用 lime 描边增强可见性）
    for ax in axes.flatten():
        # GT α=0.5：先画深色描边，再画 lime 主线
        ax.contour(Xi, Yi, ZA_gt_clip, levels=[0.5],
                   colors="#004400", linewidths=3.0, linestyles="-")
        ax.contour(Xi, Yi, ZA_gt_clip, levels=[0.5],
                   colors="lime", linewidths=1.5, linestyles="-")
        ax.contour(Xi, Yi, ZA_pr_clip, levels=[0.5],
                   colors="red", linewidths=1.0, linestyles="--")
        ax.contour(Xi, Yi, ZT_gt, levels=[800.0],
                   colors="cyan", linewidths=1.3, linestyles="-.")
        ax.set_xlabel(xlab)
        if show_ylabel:
            ax.set_ylabel(ylab)
        else:
            ax.set_ylabel("")
        ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="both")


# ----------------------------------------------------------------------------- #
# 主绘图（单模式，保留原有接口）
# ----------------------------------------------------------------------------- #
def plot_error_spatial(config_path: str,
                       mode_label: str,
                       out_dir: Path,
                       sample_idx: Optional[int] = None,
                       time_step: int = 10,
                       slice_axis: str = "y",
                       res: int = 256):
    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode_label == "conduction":
        sample_idx = 88
    elif mode_label == "keyhole":
        sample_idx = 62


    data = _prepare_data(config_path, sample_idx, time_step, slice_axis, res)
    name = data["name"]
    sample_idx = data["sample_idx"]
    t = data["t"]
    cond = data["cond"]
    metrics = data["metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f"Spatial distribution of prediction error  ({mode_label}, $t = {t}\\,\\Delta t$)",
                 fontsize=12)

    _draw_mode_on_axes(fig, axes, data, mode_label, label_prefix="(a)")

    # 共用图例放在图底部中央（所有子图外）
    legend_handles = [
        plt.Line2D([0], [0], color="lime", lw=1.5, ls="-",
                   path_effects=[mpe.withStroke(linewidth=3, foreground="#004400")],
                   label=r"GT $\alpha_{\mathrm{air}}=0.5$"),
        plt.Line2D([0], [0], color="red", lw=1.0, ls="--", label=r"Pred $\alpha_{\mathrm{air}}=0.5$"),
        plt.Line2D([0], [0], color="cyan", lw=1.3, ls="-.", label=r"$T=800\,\mathrm{K}$ (focus mask)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, framealpha=0.7, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    # 工况参数注释
    txt = (rf"Sample idx={sample_idx}    $P={cond['P']:.0f}$ W    "
           rf"$r_0={cond['r0']*1e6:.1f}\,\mu$m    $A={cond['A']:.2f}$    "
           rf"$v={cond['v']:.2f}$ m/s    $E_v\!\approx\!{cond['E_v']:.1f}$ J/mm$^3$"
           "\n"
           rf"Overall rL$_2$={metrics['rl2_overall']:.3f}    "
           rf"Active rL$_2$={metrics['rl2_active']:.3f}    "
           rf"Inactive rL$_2$={metrics['rl2_inactive']:.3f}")
    fig.text(0.5, -0.06, txt, ha="center", va="top", fontsize=9)

    stem = f"error_spatial_{mode_label}_{name}_idx{sample_idx}_t{t}"
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"[Saved] {out_dir/stem}.png  (idx={sample_idx}, mean rL2={metrics['rl2_overall']:.4f})")

    return data


# ----------------------------------------------------------------------------- #
# 合并图：2行×4列，左侧传导模式，右侧匙孔模式
# ----------------------------------------------------------------------------- #
def plot_combined(data_list: list, mode_labels: list, out_dir: Path):
    """将两个模式的 2×2 子图合并为 2×4 的大图。"""
    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)

    # 左侧 2×2：第一个模式
    axes_left = axes[:, :2]
    _draw_mode_on_axes(fig, axes_left, data_list[0], mode_labels[0],
                       label_prefix="(a)", show_ylabel=True)

    # 右侧 2×2：第二个模式
    axes_right = axes[:, 2:]
    _draw_mode_on_axes(fig, axes_right, data_list[1], mode_labels[1],
                       label_prefix="(e)", show_ylabel=True)

    # 模式标题
    axes[0, 0].annotate(mode_labels[0], xy=(0.5, 1.18), xycoords="axes fraction",
                        ha="center", fontsize=12, fontweight="bold",
                        annotation_clip=False)
    axes[0, 2].annotate(mode_labels[1], xy=(0.5, 1.18), xycoords="axes fraction",
                        ha="center", fontsize=12, fontweight="bold",
                        annotation_clip=False)

    # 共用图例放在底部中央
    legend_handles = [
        plt.Line2D([0], [0], color="lime", lw=1.5, ls="-",
                   path_effects=[mpe.withStroke(linewidth=3, foreground="#004400")],
                   label=r"GT $\alpha_{\mathrm{air}}=0.5$"),
        plt.Line2D([0], [0], color="red", lw=1.0, ls="--", label=r"Pred $\alpha_{\mathrm{air}}=0.5$"),
        plt.Line2D([0], [0], color="cyan", lw=1.3, ls="-.", label=r"$T=800\,\mathrm{K}$ (focus mask)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, framealpha=0.7, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    # 工况参数注释（两个模式并排）
    lines = []
    for d, lbl in zip(data_list, mode_labels):
        cond, m = d["cond"], d["metrics"]
        lines.append(
            rf"[{lbl}] idx={d['sample_idx']}  $P={cond['P']:.0f}$ W  "
            rf"$v={cond['v']:.2f}$ m/s  "
            rf"$E_v\!\approx\!{cond['E_v']:.1f}$ J/mm$^3$  "
            rf"rL$_2$={m['rl2_overall']:.3f}"
        )
    fig.text(0.5, -0.04, "    |    ".join(lines), ha="center", va="top", fontsize=8.5)

    t_val = data_list[0]["t"]
    fig.suptitle(f"Spatial distribution of prediction error  ($t = {t_val}\\,\\Delta t$)",
                 fontsize=13)

    stem = "error_spatial_combined"
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"[Saved] {out_dir/stem}.png")


def _parse():
    p = argparse.ArgumentParser(description="4.5.1 prediction error spatial distribution")
    p.add_argument("--config", action="append", required=True,
                   help="config json path, can be passed multiple times")
    p.add_argument("--mode_label", action="append", required=True,
                   help="display label for each config, e.g. conduction / keyhole")
    p.add_argument("--sample_idx", type=int, default=None)
    p.add_argument("--time_step", type=int, default=10)
    p.add_argument("--slice_axis", type=str, default="y", choices=["x", "y", "z"])
    p.add_argument("--out_dir", type=str,
                   default=str(Path(__file__).parent / "figures" / "4_5_1"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if len(args.config) != len(args.mode_label):
        raise SystemExit("--config 与 --mode_label 数量必须一致")
    out_dir = Path(args.out_dir)
    data_list = []
    for cfg, label in zip(args.config, args.mode_label):
        data = plot_error_spatial(cfg, label, out_dir,
                                  sample_idx=args.sample_idx,
                                  time_step=args.time_step,
                                  slice_axis=args.slice_axis)
        data_list.append(data)

    # 当有恰好两个模式时，生成合并图（2×4）
    if len(data_list) == 2:
        plot_combined(data_list, list(args.mode_label), out_dir)
