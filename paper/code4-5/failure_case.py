# -*- coding: utf-8 -*-
"""
failure_case.py — 4.5.3 失败案例剖析

扫描测试集，按全场 mean rL2 排序找出 top-K 失败样本（默认 K=5），输出：
  1) failure_ranking_<label>.csv：完整 rank 表（含工况参数与逐通道指标）；
  2) failure_combined.png/.pdf：2×2 误差热力图
     行 = 传导模式 / 匙孔模式各一个失败样本
     列 = T 误差 / α 误差
     切面取 z 轴中心（垂直于 z 轴的 X-Y 平面）
  3) failure_summary_<label>.txt：汇总 top-K 的工况分布与统计性归因。

用法示例：
    python failure_case.py \
      --config config/easypool_scale/GTO_attnres_3_easypool_stronger.json \
      --config config/keyhole_scale/GTO_attnres_3_keyhole_stronger.json \
      --mode_label conduction --mode_label keyhole \
      --topk 5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

from predictor import (get_condition_params, load_predictor,
                       per_sample_metrics, rollout_sample, sample_idx_to_h5)


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


def _center_slice_interp(coords: np.ndarray,
                         values: np.ndarray,
                         axis: str = "z",
                         res: int = 256,
                         thickness_ratio: float = 0.04):
    """沿给定轴在中心位置取一薄层节点，投影到剩余两轴平面做 linear 插值。"""
    axis_id = {"x": 0, "y": 1, "z": 2}[axis]
    ax_vals = coords[:, axis_id]
    center = 0.5 * (ax_vals.min() + ax_vals.max())
    span = max(ax_vals.max() - ax_vals.min(), 1e-12)
    thickness = thickness_ratio * span

    mask = np.abs(ax_vals - center) <= thickness
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
# 扫描 + 排序
# ----------------------------------------------------------------------------- #
def scan_and_rank(predictor, mode_label: str,
                  score_key: str = "rl2_overall") -> List[Dict]:
    """对全部测试样本计算 metrics 并按 score 排序（降序）。"""
    n = len(predictor.dataset)
    rows = []
    for idx in tqdm(range(n), desc=f"[{mode_label}] failure scan"):
        try:
            m = per_sample_metrics(predictor, idx)
            cond = get_condition_params(sample_idx_to_h5(predictor, idx))
        except Exception as exc:
            print(f"[Warn] idx={idx}: {exc}")
            continue
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
# 准备单个失败样本的误差切片数据
# ----------------------------------------------------------------------------- #
def _prepare_failure_data(predictor, row: Dict,
                          time_step: int = -1,
                          slice_axis: str = "z",
                          res: int = 256) -> Dict:
    """对一个失败样本做 rollout、取切片、计算误差场，返回绘图所需数据。"""
    fields = list(predictor.fields)
    if "T" not in fields or "alpha.air" not in fields:
        raise RuntimeError("失败案例可视化需要 T 与 alpha.air 字段")
    iT, iA = fields.index("T"), fields.index("alpha.air")

    out = rollout_sample(predictor, row["idx"])
    pred, gt, coords = out["pred"], out["gt"], out["coords"]
    horizon = pred.shape[0]
    t = horizon - 1 if time_step < 0 else min(time_step, horizon - 1)

    T_gt, T_pr = gt[t, :, iT], pred[t, :, iT]
    A_gt, A_pr = gt[t, :, iA], pred[t, :, iA]

    Xi, Yi, ZT_gt, xlab, ylab = _center_slice_interp(coords, T_gt, slice_axis, res)
    _, _, ZT_pr, _, _ = _center_slice_interp(coords, T_pr, slice_axis, res)
    _, _, ZA_gt, _, _ = _center_slice_interp(coords, A_gt, slice_axis, res)
    _, _, ZA_pr, _, _ = _center_slice_interp(coords, A_pr, slice_axis, res)

    ZT_err = np.abs(ZT_pr - ZT_gt)
    ZA_err = np.abs(ZA_pr - ZA_gt)

    return dict(
        Xi=Xi, Yi=Yi, ZT_gt=ZT_gt, ZA_gt=ZA_gt, ZA_pr=ZA_pr,
        ZT_err=ZT_err, ZA_err=ZA_err,
        xlab=xlab, ylab=ylab, t=t, row=row,
    )


# ----------------------------------------------------------------------------- #
# 2×2 合并图：每行一个模式的失败样本，每列一个物理场误差
# ----------------------------------------------------------------------------- #
def plot_failure_combined(data_list: List[Dict],
                          mode_labels: List[str],
                          out_dir: Path):
    """
    绘制 2×2 失败案例误差热力图。
    行：传导模式 / 匙孔模式（各一个失败样本）
    列：T 绝对误差 / α 绝对误差
    每列共享一个竖置 colorbar（放在该列所有子图右侧）。
    """
    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as ticker

    n_rows = len(data_list)  # typically 2
    # Layout: [img_col0] [cbar_col0] [gap] [img_col1] [cbar_col1]
    # width_ratios: image columns wide, colorbar columns narrow, a small gap
    fig = plt.figure(figsize=(11, 4.2 * n_rows))
    gs = GridSpec(
        n_rows, 5,
        figure=fig,
        width_ratios=[1, 0.045, 0.12, 1, 0.045],
        hspace=0.42,
        wspace=0.03,
        left=0.07, right=0.97,
        top=0.83, bottom=0.13,
    )

    sub_labels = ["(a)", "(b)", "(c)", "(d)"]
    col_map = {0: 0, 1: 3}   # image axes column indices in GridSpec
    cbar_map = {0: 1, 1: 4}  # colorbar axes column indices

    # Collect all ims per image-column to share colorbar limits within each column
    im_store = {0: [], 1: []}
    ax_store = {0: [], 1: []}
    vmax_store = {0: [], 1: []}

    for i, (data, mlabel) in enumerate(zip(data_list, mode_labels)):
        Xi, Yi = data["Xi"], data["Yi"]
        ZT_err, ZA_err = data["ZT_err"], data["ZA_err"]
        ZA_gt, ZA_pr = data["ZA_gt"], data["ZA_pr"]
        ZT_gt = data["ZT_gt"]
        xlab, ylab = data["xlab"], data["ylab"]
        row = data["row"]

        extent = (Xi.min(), Xi.max(), Yi.min(), Yi.max())
        imkw = dict(extent=extent, origin="lower", aspect="equal",
                    interpolation="bicubic")

        # ---- T 误差 ----
        ax_T = fig.add_subplot(gs[i, col_map[0]])
        vmax_T = max(np.nanpercentile(ZT_err, 99.0), 1e-6)
        im_T = ax_T.imshow(ZT_err, cmap="magma", vmin=0, vmax=vmax_T, **imkw)
        rl2_T = row["rl2_field"].get("T", float("nan"))
        rmse_T = row["rmse_field"].get("T", float("nan"))
        ax_T.set_title(
            f"{sub_labels[i * 2]} {mlabel}  "
            rf"$|T_{{\rm pred}}-T_{{\rm gt}}|$" + "\n"
            rf"rL$_2$={rl2_T:.3f},  RMSE={rmse_T:.1f} K",
            fontsize=9.5, pad=4,
        )
        im_store[0].append(im_T)
        ax_store[0].append(ax_T)
        vmax_store[0].append(vmax_T)

        # ---- α 误差 ----
        ax_A = fig.add_subplot(gs[i, col_map[1]])
        vmax_A = max(np.nanpercentile(ZA_err, 99.0), 1e-6)
        im_A = ax_A.imshow(ZA_err, cmap="magma", vmin=0, vmax=vmax_A, **imkw)
        rl2_A = row["rl2_field"].get("alpha.air", float("nan"))
        ax_A.set_title(
            f"{sub_labels[i * 2 + 1]} {mlabel}  "
            rf"$|\alpha_{{\rm pred}}-\alpha_{{\rm gt}}|$" + "\n"
            rf"rL$_2$={rl2_A:.3f}",
            fontsize=9.5, pad=4,
        )
        im_store[1].append(im_A)
        ax_store[1].append(ax_A)
        vmax_store[1].append(vmax_A)

        # ---- 等值线叠加 ----
        active_mask_2d = ZT_gt > 800.0
        if active_mask_2d.any():
            col_active = active_mask_2d.any(axis=0)
            x_vals_1d = Xi[0, :]
            x_active = x_vals_1d[col_active]
            x_pad = 0.02 * (x_vals_1d.max() - x_vals_1d.min())
            x_lo, x_hi = x_active.min() - x_pad, x_active.max() + x_pad
            x_keep = (Xi >= x_lo) & (Xi <= x_hi)
            ZA_gt_clip = np.where(x_keep, np.clip(ZA_gt, 0, 1), np.nan)
            ZA_pr_clip = np.where(x_keep, np.clip(ZA_pr, 0, 1), np.nan)
        else:
            ZA_gt_clip = np.clip(ZA_gt, 0, 1)
            ZA_pr_clip = np.clip(ZA_pr, 0, 1)

        ax_T.contour(Xi, Yi, ZA_gt_clip, levels=[0.5],
                     colors="#00FFFF", linewidths=1.8, linestyles="-")
        ax_T.contour(Xi, Yi, ZA_pr_clip, levels=[0.5],
                     colors="#FF4444", linewidths=1.5, linestyles="--")
        ax_A.contour(Xi, Yi, ZA_gt_clip, levels=[0.5],
                     colors="#00FFFF", linewidths=1.8, linestyles="-")
        ax_A.contour(Xi, Yi, ZA_pr_clip, levels=[0.5],
                     colors="#FF4444", linewidths=1.5, linestyles="--")

        for ax in [ax_T, ax_A]:
            ax.set_xlabel(xlab, fontsize=9)
            ax.set_ylabel(ylab, fontsize=9)
            ax.ticklabel_format(style="sci", scilimits=(-3, 3), axis="both")
            ax.tick_params(labelsize=8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

    # ---- 每列放一个共享 colorbar（取该列最大 vmax） ----
    cbar_labels = ["Absolute error (K)", "Absolute error"]
    for col_idx in (0, 1):
        global_vmax = max(vmax_store[col_idx])
        # 规范化所有该列的 imshow 到统一上限，使 colorbar 刻度一致
        for im in im_store[col_idx]:
            im.set_clim(vmin=0, vmax=global_vmax)
        # 用第一行该列的 im 作 colorbar 代表
        cax = fig.add_subplot(gs[:, cbar_map[col_idx]])
        cb = fig.colorbar(im_store[col_idx][0], cax=cax, extend="max")
        cb.set_label(cbar_labels[col_idx], fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_linewidth(0.6)
        # 3~4 刻度即可，不要过密
        cb.locator = ticker.MaxNLocator(nbins=4, symmetric=False)
        cb.update_ticks()

    # ---- 等值线图例（放在图底部中央，仅 α 子图有分界线） ----
    legend_handles = [
        plt.Line2D([0], [0], color="#00FFFF", lw=1.8, ls="-",
                   label=r"GT  $\alpha_{\rm air}=0.5$"),
        plt.Line2D([0], [0], color="#FF4444", lw=1.5, ls="--",
                   label=r"Pred  $\alpha_{\rm air}=0.5$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        framealpha=0.85,
        edgecolor="0.7",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    # ---- 工况参数注释（图例下方） ----
    cond_lines = []
    for data, mlabel in zip(data_list, mode_labels):
        row = data["row"]
        cond_lines.append(
            rf"[{mlabel}]  "
            rf"$P={row['P']:.0f}$ W,  "
            rf"$r_0={row['r0']*1e6:.1f}\,\mu$m,  "
            rf"$A={row['A']:.2f}$,  "
            rf"$v={row['v']:.2f}$ m/s,  "
            rf"$E_v\approx{row['E_v']:.1f}$ J/mm$^3$,  "
            rf"rL$_2$(overall)$={row['rl2_overall']:.3f}$"
        )
    fig.text(
        0.5, -0.01,
        "\n".join(cond_lines),
        ha="center", va="top",
        fontsize=8, color="0.3",
        transform=fig.transFigure,
    )

    # ---- 总标题 ----
    t_val = data_list[0]["t"]
    fig.suptitle(
        "Failure Case Analysis: Prediction Error Distribution  "
        rf"($t={t_val}\,\Delta t$,  $z$-center slice)",
        fontsize=11, y=0.96,
    )

    stem = "failure_combined"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_dir / stem}.png")


# ----------------------------------------------------------------------------- #
# CSV 排名表
# ----------------------------------------------------------------------------- #
def _save_ranking_csv(rows: List[Dict], path: Path, mode_label: str):
    fields = ["rank", "mode", "idx", "rl2_overall", "rl2_active", "rl2_inactive",
              "rl2_T", "rl2_alpha", "rmse_T", "rmse_alpha",
              "P", "r0", "A", "v", "L", "E_v", "h5_path"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for rank, r in enumerate(rows, 1):
            w.writerow([
                rank, mode_label, r["idx"],
                f"{r['rl2_overall']:.6f}", f"{r['rl2_active']:.6f}",
                f"{r['rl2_inactive']:.6f}",
                f"{r['rl2_field'].get('T', float('nan')):.6f}",
                f"{r['rl2_field'].get('alpha.air', float('nan')):.6f}",
                f"{r['rmse_field'].get('T', float('nan')):.4f}",
                f"{r['rmse_field'].get('alpha.air', float('nan')):.6f}",
                r["P"], r["r0"], r["A"], r["v"], r["L"], r["E_v"], r["h5_path"],
            ])
    print(f"[Saved] {path}")


# ----------------------------------------------------------------------------- #
# 归因统计摘要
# ----------------------------------------------------------------------------- #
def _save_summary(top_rows: List[Dict], all_rows: List[Dict],
                  path: Path, mode_label: str):
    def _stats(rows, key, scale=1.0):
        vals = np.array([r[key] * scale for r in rows], dtype=float)
        return float(np.mean(vals)), float(np.std(vals)), \
               float(np.min(vals)), float(np.max(vals))

    lines = [
        f"Failure-case attribution summary  ({mode_label})",
        f"  total samples scored : {len(all_rows)}",
        f"  top-K considered     : {len(top_rows)}",
        "",
        "  metric                         top-K mean ± std  (min, max)"
        "        full-set mean ± std",
    ]
    for key, scale, label in [
        ("rl2_overall", 1.0, "overall rL2          "),
        ("rl2_active",  1.0, "active rL2           "),
        ("E_v",         1.0, "E_v (J/mm^3)         "),
        ("P",           1.0, "P (W)                "),
        ("r0",          1e6, "r0 (um)              "),
        ("A",           1.0, "A                    "),
        ("v",           1.0, "v (m/s)              "),
    ]:
        tm, ts, tmin, tmax = _stats(top_rows, key, scale)
        am, as_, _, _ = _stats(all_rows, key, scale)
        lines.append(
            f"  {label}: {tm:9.3f} ± {ts:7.3f}  ({tmin:7.3f}, {tmax:7.3f})"
            f"    {am:9.3f} ± {as_:7.3f}"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Saved] {path}")


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# 辅助：从排名列表中去除极端离群值后挑选绘图样本
# ----------------------------------------------------------------------------- #
def _pick_plot_row(rows: List[Dict], plot_rank: int,
                  outlier_pct: float, score_key: str) -> Dict:
    """
    先用 outlier_pct 去掉 score_key 最高的那一分位的极端样本，
    再在过滤后的列表中（仍按 score 降序）取第 plot_rank 个样本。
    outlier_pct=0 表示不过滤，直接按原排名取。
    """
    if outlier_pct <= 0:
        return rows[min(plot_rank - 1, len(rows) - 1)]

    scores = np.array([r["score"] for r in rows], dtype=float)
    cutoff = np.percentile(scores, 100.0 - outlier_pct)
    filtered = [r for r in rows if r["score"] < cutoff]
    if not filtered:
        print(f"[Warn] outlier filter removed all samples; reverting to no filter")
        filtered = rows
    filtered.sort(key=lambda r: -r["score"])
    pick = filtered[min(plot_rank - 1, len(filtered) - 1)]
    print(f"  [outlier_pct={outlier_pct}%] cutoff={cutoff:.4f}, "
          f"kept {len(filtered)}/{len(rows)} samples, "
          f"picked idx={pick['idx']} (score={pick['score']:.4f})")
    return pick


def _parse():
    p = argparse.ArgumentParser(description="4.5.3 failure-case analysis")
    p.add_argument("--config", action="append", default=[],
                   help="config json path, repeatable (one per mode)")
    p.add_argument("--mode_label", action="append", required=True,
                   help="mode label for each config, e.g. conduction / keyhole")
    p.add_argument("--topk", type=int, default=5,
                   help="number of top failure samples for ranking/summary")
    p.add_argument("--plot_rank", type=int, default=1,
                   help="which rank to pick for the combined error plot (1-based)")
    p.add_argument("--score_key", type=str, default="rl2_overall",
                   choices=["rl2_overall", "rl2_active"],
                   help="ranking criterion")
    p.add_argument("--outlier_pct", type=float, default=0.0,
                   help="去除 score_key 最高的 outlier_pct%% 极端样本后再选绘图代表，"
                        "0 表示不过滤（默认）。例如 --outlier_pct 3 去掉最差 3%%。")
    p.add_argument("--time_step", type=int, default=-1,
                   help="time step to visualize, -1 = last horizon step")
    p.add_argument("--slice_axis", type=str, default="z",
                   choices=["x", "y", "z"],
                   help="slice axis (default: z, i.e. X-Y plane)")
    p.add_argument("--out_dir", type=str,
                   default=str(Path(__file__).parent / "figures" / "4_5_3"))
    # 快速重绘模式：已有 CSV 时跳过全量推理，只重新选样本并绘图
    p.add_argument("--from_csv", action="store_true",
                   help="从已有的 failure_ranking_<label>.csv 直接读取排名，"
                        "跳过全量推理扫描（仍需 --config 做 rollout）")
    return p.parse_args()


def _load_rows_from_csv(csv_path: Path, score_key: str) -> List[Dict]:
    """从 CSV 恢复 scan_and_rank 返回格式，用于 --from_csv 快速重绘。"""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rl2_field = {
                "T": float(r["rl2_T"]) if r["rl2_T"] else float("nan"),
                "alpha.air": float(r["rl2_alpha"]) if r["rl2_alpha"] else float("nan"),
            }
            rmse_field = {
                "T": float(r["rmse_T"]) if r["rmse_T"] else float("nan"),
                "alpha.air": float(r["rmse_alpha"]) if r["rmse_alpha"] else float("nan"),
            }
            row = {
                "idx": int(r["idx"]),
                "score": float(r[score_key]) if score_key in r else float(r["rl2_overall"]),
                "rl2_overall": float(r["rl2_overall"]),
                "rl2_active": float(r["rl2_active"]),
                "rl2_inactive": float(r["rl2_inactive"]),
                "rl2_field": rl2_field,
                "rmse_field": rmse_field,
                "P": float(r["P"]), "r0": float(r["r0"]),
                "A": float(r["A"]), "v": float(r["v"]),
                "L": float(r["L"]), "E_v": float(r["E_v"]),
                "h5_path": r["h5_path"],
            }
            rows.append(row)
    rows.sort(key=lambda r: -r["score"])
    return rows


if __name__ == "__main__":
    _setup_style()
    args = _parse()
    if len(args.config) != len(args.mode_label):
        raise SystemExit("--config 与 --mode_label 数量必须一致")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    all_labels = []

    for cfg, label in zip(args.config, args.mode_label):
        predictor = load_predictor(cfg)

        if args.from_csv:
            # 快速重绘：从已有 CSV 恢复排名，跳过全量推理
            csv_path = out_dir / f"failure_ranking_{label}.csv"
            if not csv_path.exists():
                raise SystemExit(f"[Error] --from_csv 指定但找不到 {csv_path}")
            rows = _load_rows_from_csv(csv_path, args.score_key)
            print(f"[{label}] loaded {len(rows)} rows from {csv_path}")
        else:
            rows = scan_and_rank(predictor, label, score_key=args.score_key)
            if not rows:
                print(f"[Warn] no rows for {label}")
                continue
            _save_ranking_csv(rows, out_dir / f"failure_ranking_{label}.csv", label)
            topk = rows[: max(1, args.topk)]
            _save_summary(topk, rows, out_dir / f"failure_summary_{label}.txt", label)

        pick_row = _pick_plot_row(rows, args.plot_rank, args.outlier_pct, args.score_key)
        print(f"[{label}] picked idx={pick_row['idx']}, "
              f"rL2={pick_row['rl2_overall']:.4f}, E_v={pick_row['E_v']:.1f}")

        data = _prepare_failure_data(
            predictor, pick_row,
            time_step=args.time_step,
            slice_axis=args.slice_axis,
        )
        all_data.append(data)
        all_labels.append(label)

    if len(all_data) >= 1:
        plot_failure_combined(all_data, all_labels, out_dir)
    else:
        print("[Warn] no data to plot")
