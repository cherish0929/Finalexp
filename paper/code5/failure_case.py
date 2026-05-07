# -*- coding: utf-8 -*-
"""
failure_case.py — 4.5.3 失败案例剖析

扫描测试集，按全场 mean rL2 排序找出 top-K 失败样本（默认 K=5），输出：
  1) failure_ranking_<label>.csv：完整 rank 表（含工况参数与逐通道指标）；
  2) failure_top<rank>_<label>.png/.pdf：每个失败样本的 4 联可视化对比
     （T 真/预/误差 + α 真/预/误差），并在标题上写明工况参数与 rL2；
  3) failure_summary_<label>.txt：汇总 top-K 的工况分布与统计性归因。

用法示例：
    python failure_case.py \
      --config config/easypool/GTO_attnres_3_easypool_stronger.json \
      --mode_label conduction --topk 5
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

from predictor import (get_condition_params, load_predictor,
                       per_sample_metrics, sample_idx_to_h5)


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


def _slice_interp(coords, vals, axis="y", res=240, thickness_ratio=0.04):
    axis_id = {"x": 0, "y": 1, "z": 2}[axis]
    ax_v = coords[:, axis_id]
    center = 0.5 * (ax_v.min() + ax_v.max())
    span = max(ax_v.max() - ax_v.min(), 1e-12)
    th = thickness_ratio * span
    mask = np.abs(ax_v - center) <= th
    if mask.sum() < 32:
        order = np.argsort(np.abs(ax_v - center))[: max(32, len(ax_v) // 30)]
        mask = np.zeros_like(ax_v, dtype=bool); mask[order] = True
    other = [i for i in (0, 1, 2) if i != axis_id]
    px, py = coords[mask, other[0]], coords[mask, other[1]]
    xi = np.linspace(px.min(), px.max(), res); yi = np.linspace(py.min(), py.max(), res)
    Xi, Yi = np.meshgrid(xi, yi)
    Z = griddata(np.column_stack([px, py]), vals[mask], (Xi, Yi), method="linear")
    if Z is None or np.isnan(Z).all():
        Z = griddata(np.column_stack([px, py]), vals[mask], (Xi, Yi), method="nearest")
    if np.isnan(Z).any():
        Zn = griddata(np.column_stack([px, py]), vals[mask], (Xi, Yi), method="nearest")
        Z = np.where(np.isnan(Z), Zn, Z)
    return Xi, Yi, Z, {0: "X (m)", 1: "Y (m)", 2: "Z (m)"}[other[0]], \
           {0: "X (m)", 1: "Y (m)", 2: "Z (m)"}[other[1]]


# ----------------------------------------------------------------------------- #
# 扫描 + 排序
# ----------------------------------------------------------------------------- #
def scan_and_rank(predictor, mode_label: str, score_key: str = "rl2_overall") -> List[Dict]:
    """对全部测试样本计算 metrics 并按 score 排序（降序）。"""
    n = len(predictor.dataset)
    rows = []
    for idx in tqdm(range(n), desc=f"[{mode_label}] failure scan"):
        try:
            m = per_sample_metrics(predictor, idx)
            cond = get_condition_params(sample_idx_to_h5(predictor, idx))
        except Exception as exc:
            print(f"[Warn] idx={idx}: {exc}"); continue
        rows.append({
            "idx": idx,
            "score": m[score_key],
            "rl2_overall": m["rl2_overall"],
            "rl2_active": m["rl2_active"],
            "rl2_inactive": m["rl2_inactive"],
            "rl2_field": m["rl2_field"],
            "rmse_field": m["rmse_field"],
            "P": cond["P"], "r0": cond["r0"], "A": cond["A"],
            "v": cond["v"], "L": cond["L"], "E_v": cond["E_v"],
            "h5_path": sample_idx_to_h5(predictor, idx),
        })
    rows.sort(key=lambda r: -r["score"])
    return rows


# ----------------------------------------------------------------------------- #
# 单个失败样本的 6 子图可视化
# ----------------------------------------------------------------------------- #
def plot_failure_sample(predictor, row: Dict, mode_label: str,
                        out_dir: Path, rank: int,
                        time_step: int = -1, slice_axis: str = "y"):
    fields = list(predictor.fields)
    if "T" not in fields or "alpha.air" not in fields:
        raise RuntimeError("失败案例可视化需要 T 与 alpha.air 字段")
    iT, iA = fields.index("T"), fields.index("alpha.air")

    out = predictor.predict_rollout(sample_idx=row["idx"])
    pred, gt, coords = out["pred"], out["gt"], out["coords"]
    horizon = pred.shape[0]
    t = horizon - 1 if time_step < 0 else min(time_step, horizon - 1)

    Xi, Yi, ZT_gt, xlab, ylab = _slice_interp(coords, gt[t, :, iT], slice_axis)
    _, _, ZT_pr, _, _ = _slice_interp(coords, pred[t, :, iT], slice_axis)
    _, _, ZA_gt, _, _ = _slice_interp(coords, gt[t, :, iA], slice_axis)
    _, _, ZA_pr, _, _ = _slice_interp(coords, pred[t, :, iA], slice_axis)
    ZT_err = np.abs(ZT_pr - ZT_gt); ZA_err = np.abs(ZA_pr - ZA_gt)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), constrained_layout=True)
    extent = (Xi.min(), Xi.max(), Yi.min(), Yi.max())
    kw = dict(extent=extent, origin="lower", aspect="equal", interpolation="bicubic")

    # 第 1 行：温度
    im00 = axes[0, 0].imshow(ZT_gt, cmap="inferno", vmin=300, vmax=3500, **kw)
    axes[0, 0].set_title(r"(a) GT $T$ (K)"); fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.03)
    im01 = axes[0, 1].imshow(ZT_pr, cmap="inferno", vmin=300, vmax=3500, **kw)
    axes[0, 1].set_title(r"(b) Pred $T$ (K)"); fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.03)
    err_T_vmax = max(np.nanpercentile(ZT_err, 99.0), 1e-6)
    im02 = axes[0, 2].imshow(ZT_err, cmap="magma", vmin=0, vmax=err_T_vmax, **kw)
    axes[0, 2].set_title(r"(c) $|T_{\mathrm{pred}}-T_{\mathrm{gt}}|$ (K)")
    fig.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.03, extend="max")

    # 第 2 行：VOF
    im10 = axes[1, 0].imshow(np.clip(ZA_gt, 0, 1), cmap="viridis", vmin=0, vmax=1, **kw)
    axes[1, 0].set_title(r"(d) GT $\alpha_{\mathrm{air}}$"); fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.03)
    im11 = axes[1, 1].imshow(np.clip(ZA_pr, 0, 1), cmap="viridis", vmin=0, vmax=1, **kw)
    axes[1, 1].set_title(r"(e) Pred $\alpha_{\mathrm{air}}$"); fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.03)
    err_A_vmax = max(np.nanpercentile(ZA_err, 99.0), 1e-6)
    im12 = axes[1, 2].imshow(ZA_err, cmap="magma", vmin=0, vmax=err_A_vmax, **kw)
    axes[1, 2].set_title(r"(f) $|\alpha_{\mathrm{pred}}-\alpha_{\mathrm{gt}}|$")
    fig.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.03, extend="max")

    for ax in axes.flatten():
        ax.contour(Xi, Yi, np.clip(ZA_gt, 0, 1), levels=[0.5], colors="black", linewidths=1.0, linestyles="-")
        ax.contour(Xi, Yi, np.clip(ZA_pr, 0, 1), levels=[0.5], colors="red", linewidths=0.9, linestyles="--")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="both")

    rl2_T = row["rl2_field"].get("T", float("nan"))
    rl2_A = row["rl2_field"].get("alpha.air", float("nan"))
    rmse_T = row["rmse_field"].get("T", float("nan"))
    title = (f"Failure case rank #{rank}  ({mode_label}, $t={t}\\,\\Delta t$)   "
             f"idx={row['idx']}   "
             rf"$P={row['P']:.0f}$ W   $r_0={row['r0']*1e6:.1f}\,\mu$m   "
             rf"$A={row['A']:.2f}$   $v={row['v']:.2f}$ m/s   "
             rf"$E_v\!\approx\!{row['E_v']:.1f}$ J/mm$^3$"
             "\n"
             rf"Overall rL$_2$={row['rl2_overall']:.3f}   "
             rf"Active={row['rl2_active']:.3f}   Inactive={row['rl2_inactive']:.3f}   "
             rf"$T$ rL$_2$={rl2_T:.3f} (RMSE {rmse_T:.1f} K)   "
             rf"$\alpha$ rL$_2$={rl2_A:.3f}")
    fig.suptitle(title, fontsize=11, y=1.04)

    stem = f"failure_top{rank:02d}_{mode_label}_idx{row['idx']}"
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"[Saved] {out_dir/stem}.png  (rL2={row['rl2_overall']:.4f})")


def _save_ranking_csv(rows: List[Dict], path: Path, mode_label: str):
    import csv
    fields = ["rank", "mode", "idx", "rl2_overall", "rl2_active", "rl2_inactive",
              "rl2_T", "rl2_alpha", "rmse_T", "rmse_alpha",
              "P", "r0", "A", "v", "L", "E_v", "h5_path"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(fields)
        for rank, r in enumerate(rows, 1):
            w.writerow([
                rank, mode_label, r["idx"],
                f"{r['rl2_overall']:.6f}", f"{r['rl2_active']:.6f}", f"{r['rl2_inactive']:.6f}",
                f"{r['rl2_field'].get('T', float('nan')):.6f}",
                f"{r['rl2_field'].get('alpha.air', float('nan')):.6f}",
                f"{r['rmse_field'].get('T', float('nan')):.4f}",
                f"{r['rmse_field'].get('alpha.air', float('nan')):.6f}",
                r["P"], r["r0"], r["A"], r["v"], r["L"], r["E_v"], r["h5_path"],
            ])
    print(f"[Saved] {path}")


def _save_summary(top_rows: List[Dict], all_rows: List[Dict], path: Path,
                  mode_label: str):
    """对 top-K 失败样本做简单的统计性归因（与全集对比）。"""
    def _stats(rows, key, scale=1.0):
        vals = np.array([r[key] * scale for r in rows], dtype=float)
        return float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)), float(np.max(vals))

    lines = []
    lines.append(f"Failure-case attribution summary  ({mode_label})")
    lines.append(f"  total samples scored : {len(all_rows)}")
    lines.append(f"  top-K considered     : {len(top_rows)}")
    lines.append("")
    lines.append("  metric                         top-K mean ± std  (min, max)        full-set mean ± std")
    for key, scale, label in [
        ("rl2_overall", 1.0, "overall rL2          "),
        ("rl2_active",  1.0, "active rL2           "),
        ("E_v",          1.0, "E_v (J/mm^3)         "),
        ("P",            1.0, "P (W)                "),
        ("r0",           1e6, "r0 (um)              "),
        ("A",            1.0, "A                    "),
        ("v",            1.0, "v (m/s)              "),
    ]:
        tm, ts, tmin, tmax = _stats(top_rows, key, scale)
        am, as_, _, _ = _stats(all_rows, key, scale)
        lines.append(f"  {label}: {tm:9.3f} ± {ts:7.3f}  ({tmin:7.3f}, {tmax:7.3f})    "
                     f"{am:9.3f} ± {as_:7.3f}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Saved] {path}")


def _parse():
    p = argparse.ArgumentParser(description="4.5.3 failure-case analysis")
    p.add_argument("--config", action="append", required=True,
                   help="config json path, repeatable")
    p.add_argument("--mode_label", action="append", required=True,
                   help="mode label for each config, e.g. conduction / keyhole")
    p.add_argument("--topk", type=int, default=5,
                   help="number of top failure samples to visualize")
    p.add_argument("--score_key", type=str, default="rl2_overall",
                   choices=["rl2_overall", "rl2_active"],
                   help="ranking criterion (default: full-field mean rL2)")
    p.add_argument("--time_step", type=int, default=-1,
                   help="time step to visualize, -1 = last horizon step")
    p.add_argument("--slice_axis", type=str, default="y", choices=["x", "y", "z"])
    p.add_argument("--out_dir", type=str,
                   default=str(Path(__file__).parent / "figures" / "4_5_3"))
    return p.parse_args()


if __name__ == "__main__":
    _setup_style()
    args = _parse()
    if len(args.config) != len(args.mode_label):
        raise SystemExit("--config 与 --mode_label 数量必须一致")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for cfg, label in zip(args.config, args.mode_label):
        predictor = load_predictor(cfg)
        rows = scan_and_rank(predictor, label, score_key=args.score_key)
        if not rows:
            print(f"[Warn] no rows for {label}"); continue

        _save_ranking_csv(rows, out_dir / f"failure_ranking_{label}.csv", label)

        topk = rows[: max(1, args.topk)]
        for rank, r in enumerate(topk, 1):
            try:
                plot_failure_sample(predictor, r, label, out_dir, rank,
                                    time_step=args.time_step,
                                    slice_axis=args.slice_axis)
            except Exception as exc:
                print(f"[Warn] plot rank #{rank} idx={r['idx']} failed: {exc}")

        _save_summary(topk, rows, out_dir / f"failure_summary_{label}.txt", label)
