# -*- coding: utf-8 -*-
"""
sensitivity.py — 4.5.2 工况敏感性分析

对传导/匙孔模式的全部测试样本，评估每个样本的全场 mean rL2、活跃区 rL2、
逐通道 rL2，与其工况参数 (P, r0, A, v, L) 一并写入 CSV；并以体积能量密度
E_v = A·P / (v · r_0^2 · h) 为横轴绘制散点图与分模式线性回归。

输出：
    paper/code5/figures/4_5_2/sensitivity_<label>.png/.pdf
    paper/code5/figures/4_5_2/sensitivity_<label>.csv

设计要点：
  - 每个 config 视作一种模式；同时传入 conduction + keyhole 时两个模式叠绘；
  - 散点形状/颜色按模式区分（蓝色圆 = conduction，红色三角 = keyhole）；
  - 每模式拟合一条 OLS 直线，并在标题里给出 Pearson 相关系数；
  - log10(E_v) 横轴可选（--log_x），便于观察跨数量级 E_v 的趋势。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from predictor import (get_condition_params, load_predictor,
                       per_sample_metrics, sample_idx_to_h5)


_MODE_STYLE = {
    "conduction": dict(color="#1f77b4", marker="o", label="Conduction"),
    "keyhole":    dict(color="#d62728", marker="^", label="Keyhole"),
}


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


def _scan_dataset(config_path: str, mode_label: str) -> List[Dict]:
    """对一个 config 对应的全部测试样本计算指标 + 工况参数。"""
    predictor = load_predictor(config_path)
    n = len(predictor.dataset)
    rows = []
    for idx in tqdm(range(n), desc=f"[{mode_label}] eval"):
        try:
            metrics = per_sample_metrics(predictor, idx)
            cond = get_condition_params(sample_idx_to_h5(predictor, idx))
        except Exception as exc:
            print(f"[Warn] idx={idx} failed: {exc}")
            continue
        rows.append({
            "mode": mode_label,
            "config": predictor.args.name,
            "idx": idx,
            "P": cond["P"], "r0": cond["r0"], "A": cond["A"], "v": cond["v"],
            "L": cond["L"], "E_v": cond["E_v"], "E_l": cond["E_l"],
            "rl2_overall": metrics["rl2_overall"],
            "rl2_active": metrics["rl2_active"],
            "rl2_inactive": metrics["rl2_inactive"],
            **{f"rl2_{k}": v for k, v in metrics["rl2_field"].items()},
            **{f"rmse_{k}": v for k, v in metrics["rmse_field"].items()},
        })
    return rows


def _save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    # 拓展到所有 row 的并集
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Saved] {path}  ({len(rows)} rows)")


def _plot_scatter(rows: List[Dict], out_dir: Path, log_x: bool = False,
                  fname_stem: str = "sensitivity"):
    """绘制 E_v vs mean rL2 散点 + 分模式线性拟合。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    for ax, ykey, ylabel, title in [
        (axes[0], "rl2_overall", r"Overall mean rL$_2$",
         r"(a) Sensitivity of overall rL$_2$ to $E_v$"),
        (axes[1], "rl2_active", r"Active-region mean rL$_2$",
         r"(b) Sensitivity of active-region rL$_2$ to $E_v$"),
    ]:
        for mode in sorted({r["mode"] for r in rows}):
            sub = [r for r in rows if r["mode"] == mode]
            if not sub:
                continue
            ev = np.array([r["E_v"] for r in sub])
            yv = np.array([r[ykey] for r in sub])
            valid = np.isfinite(ev) & np.isfinite(yv)
            ev, yv = ev[valid], yv[valid]
            if ev.size == 0:
                continue

            style = _MODE_STYLE.get(mode, dict(marker="s", color="gray", label=mode))
            ax.scatter(ev, yv, s=22, alpha=0.7, edgecolor="black",
                       linewidths=0.4, **style)

            # 线性回归（在原始或 log10(Ev) 空间）
            x_fit = np.log10(ev) if log_x else ev
            if x_fit.size >= 2 and np.std(x_fit) > 1e-12:
                slope, intercept = np.polyfit(x_fit, yv, 1)
                pearson = float(np.corrcoef(x_fit, yv)[0, 1])
                xs = np.linspace(x_fit.min(), x_fit.max(), 100)
                ys = slope * xs + intercept
                xs_plot = (10 ** xs) if log_x else xs
                ax.plot(xs_plot, ys, color=style["color"], lw=1.4, ls="-",
                        alpha=0.85,
                        label=f"{style['label']} fit (r={pearson:+.2f})")

        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(r"Volumetric energy density $E_v$ (J/mm$^3$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        ax.legend(loc="best", framealpha=0.7)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{fname_stem}.png")
    fig.savefig(out_dir / f"{fname_stem}.pdf")
    plt.close(fig)
    print(f"[Saved] {out_dir/fname_stem}.png/.pdf")


def _plot_per_param_grid(rows: List[Dict], out_dir: Path,
                         fname_stem: str = "sensitivity_params"):
    """补充：按工艺参数 P, r0, A, v 单独散点，便于读者排查耦合效应。"""
    params = [("P",  "Laser power $P$ (W)",          1.0),
              ("r0", "Effective radius $r_0$ ($\\mu$m)", 1e6),
              ("A",  "Absorptivity $A$",              1.0),
              ("v",  "Scan speed $v$ (m/s)",          1.0)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    axes = axes.flatten()

    for ax, (pkey, plabel, scale) in zip(axes, params):
        for mode in sorted({r["mode"] for r in rows}):
            sub = [r for r in rows if r["mode"] == mode]
            if not sub:
                continue
            x = np.array([r[pkey] * scale for r in sub])
            y = np.array([r["rl2_overall"] for r in sub])
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            style = _MODE_STYLE.get(mode, dict(marker="s", color="gray", label=mode))
            ax.scatter(x, y, s=20, alpha=0.7, edgecolor="black",
                       linewidths=0.4, **style)
            if x.size >= 3 and np.std(x) > 1e-12:
                pearson = float(np.corrcoef(x, y)[0, 1])
                ax.text(0.04, 0.94 if mode == "conduction" else 0.86,
                        f"{style['label']}: r={pearson:+.2f}",
                        transform=ax.transAxes, color=style["color"],
                        fontsize=9, va="top")
        ax.set_xlabel(plabel)
        ax.set_ylabel(r"Overall mean rL$_2$")
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)

    fig.suptitle("Per-parameter sensitivity of prediction error", fontsize=12)
    fig.savefig(out_dir / f"{fname_stem}.png")
    fig.savefig(out_dir / f"{fname_stem}.pdf")
    plt.close(fig)
    print(f"[Saved] {out_dir/fname_stem}.png/.pdf")


def _parse():
    p = argparse.ArgumentParser(description="4.5.2 operating-condition sensitivity")
    p.add_argument("--config", action="append", required=True,
                   help="config json path, repeatable")
    p.add_argument("--mode_label", action="append", required=True,
                   help="mode tag for each config, e.g. conduction / keyhole")
    p.add_argument("--out_dir", type=str,
                   default=str(Path(__file__).parent / "figures" / "4_5_2"))
    p.add_argument("--log_x", action="store_true",
                   help="use log10(E_v) on x axis")
    return p.parse_args()


if __name__ == "__main__":
    _setup_style()
    args = _parse()
    if len(args.config) != len(args.mode_label):
        raise SystemExit("--config 与 --mode_label 数量必须一致")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for cfg, label in zip(args.config, args.mode_label):
        rows = _scan_dataset(cfg, label)
        all_rows.extend(rows)
        _save_csv(rows, out_dir / f"sensitivity_{label}.csv")

    _save_csv(all_rows, out_dir / "sensitivity_all.csv")
    _plot_scatter(all_rows, out_dir, log_x=args.log_x)
    _plot_per_param_grid(all_rows, out_dir)
