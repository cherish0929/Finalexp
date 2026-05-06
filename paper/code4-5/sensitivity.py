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
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from predictor import (get_condition_params, load_predictor,
                       per_sample_metrics, sample_idx_to_h5)


def _filter_outliers(rows: List[Dict], key: str = "rl2_overall",
                     iqr_factor: float = 2.5) -> List[Dict]:
    """基于 IQR 去除极端离群值，避免少量异常样本压缩可视化尺度。"""
    vals = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
    if vals.size < 4:
        return rows
    q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    iqr = q3 - q1
    upper = q3 + iqr_factor * iqr
    removed = [r for r in rows if r.get(key, 0) > upper]
    if removed:
        print(f"[Info] Outlier filter ({key}): removed {len(removed)} points "
              f"with value > {upper:.4f}")
    return [r for r in rows if r.get(key, 0) <= upper]


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


def _make_override_config(config_path: str, test_list_paths: List[str]) -> str:
    """生成一个把 data.test_list 替换为指定文件的临时 config，供补充数据扫描使用。

    仅写入 sensitivity 流程内的临时目录。
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("data", {})["test_list"] = list(test_list_paths)
    tmp_dir = tempfile.mkdtemp(prefix="sensitivity_override_")
    out_path = os.path.join(tmp_dir, Path(config_path).name)
    with open(out_path, "w") as f:
        json.dump(cfg, f)
    return out_path


def _read_extra_list_from_config(config_path: str) -> List[str]:
    """读取 config.data.extra_list（文件路径列表），不存在则返回空 list。

    用于在 per-parameter 图中补充扫描更多工况（例如匙孔验证集），不影响主散点。
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)
    extra = cfg.get("data", {}).get("extra_list", [])
    if isinstance(extra, str):
        extra = [extra]
    return [p for p in extra if p]


def _scan_dataset(config_path: str, mode_label: str, step: int = 4,
                  override_test_list: Optional[List[str]] = None) -> List[Dict]:
    """对一个 config 对应的测试样本（均匀抽样 1/step）计算指标 + 工况参数。

    注意：dataset_fast 会把同一个 h5 切成多个时间窗样本，工况参数 (P, r0, A, v)
    在同一 h5 内是相同的。此处仍逐样本输出，便于在 CSV 中追溯；后续绘图与
    回归会按 h5 聚合，避免同一工况被样本数加权。

    为加快敏感性扫描，按步长 step 均匀抽样测试集（默认 1/4）；同一工况
    在聚合阶段会按 h5 取平均，因此抽样不会丢失工况覆盖度。

    若提供 override_test_list，则临时把 config.data.test_list 替换为该列表，
    用于补充扫描验证集等额外数据（仅供 per-parameter 图扩展使用）。
    """
    eff_config = (_make_override_config(config_path, override_test_list)
                  if override_test_list else config_path)
    predictor = load_predictor(eff_config)
    n_total = len(predictor.dataset)
    indices = list(range(0, n_total, step))
    rows = []
    for idx in tqdm(indices, desc=f"[{mode_label}] eval"):
        try:
            metrics = per_sample_metrics(predictor, idx)
            h5_path = sample_idx_to_h5(predictor, idx)
            cond = get_condition_params(h5_path)
        except Exception as exc:
            print(f"[Warn] idx={idx} failed: {exc}")
            continue
        rows.append({
            "mode": mode_label,
            "config": predictor.args.name,
            "idx": idx,
            "h5": h5_path,
            "P": cond["P"], "r0": cond["r0"], "A": cond["A"], "v": cond["v"],
            "L": cond["L"], "E_v": cond["E_v"], "E_l": cond["E_l"],
            "rl2_overall": metrics["rl2_overall"],
            "rl2_active": metrics["rl2_active"],
            "rl2_inactive": metrics["rl2_inactive"],
            **{f"rl2_{k}": v for k, v in metrics["rl2_field"].items()},
            **{f"rmse_{k}": v for k, v in metrics["rmse_field"].items()},
        })
    return rows


def _aggregate_by_h5(rows: List[Dict]) -> List[Dict]:
    """把同一 (mode, h5) 下的多个时间窗样本聚合为一个工况点。

    工况参数 (P, r0, A, v, L, E_v, E_l) 在同一 h5 内恒定，直接取首样本；
    各 rL2 / RMSE 指标在窗口维度上取算术平均，得到该工况的代表性误差。
    """
    groups: Dict[tuple, List[Dict]] = {}
    for r in rows:
        key = (r["mode"], r.get("h5", r.get("config", "") + f":{r['idx']}"))
        groups.setdefault(key, []).append(r)

    metric_prefixes = ("rl2_", "rmse_")
    agg_rows: List[Dict] = []
    for (mode, h5), grp in groups.items():
        head = grp[0]
        out = {
            "mode": mode,
            "config": head["config"],
            "h5": h5,
            "n_samples": len(grp),
            "P": head["P"], "r0": head["r0"], "A": head["A"], "v": head["v"],
            "L": head["L"], "E_v": head["E_v"], "E_l": head["E_l"],
        }
        metric_keys = [k for k in head.keys() if k.startswith(metric_prefixes)]
        for k in metric_keys:
            vals = np.array([g[k] for g in grp if np.isfinite(g.get(k, np.nan))])
            out[k] = float(vals.mean()) if vals.size else float("nan")
        agg_rows.append(out)
    return agg_rows


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
    rows = _filter_outliers(rows, key="rl2_overall")
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
    """补充：按工艺参数 P, r0, v 单独散点（三联图），便于读者排查耦合效应。

    去掉吸收率 A：本研究中吸收率有独立的标定方式，调整 config 中的 A 不会显著
    改变仿真结果，故不纳入敏感性分析。
    绘图前基于 IQR 过滤极端离群值，以保证图面可读性。
    """
    rows = _filter_outliers(rows, key="rl2_overall")
    params = [("P",  "Laser power $P$ (W)",          1.0),
              ("r0", "Effective radius $r_0$ ($\\mu$m)", 1e6),
              ("v",  "Scan speed $v$ (m/s)",          1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
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
    p.add_argument("--step", type=int, default=4,
                   help="uniform subsampling stride over the test set (default 4 = 1/4)")
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
        rows = _scan_dataset(cfg, label, step=args.step)
        all_rows.extend(rows)
        _save_csv(rows, out_dir / f"sensitivity_{label}_per_sample.csv")

    _save_csv(all_rows, out_dir / "sensitivity_all_per_sample.csv")

    # 同一 h5 内的多个时间窗共享同一工况，聚合后再做散点 / 回归，
    # 避免被样本数加权扭曲 Pearson 相关与拟合斜率。
    agg_rows = _aggregate_by_h5(all_rows)
    _save_csv(agg_rows, out_dir / "sensitivity_all.csv")
    _plot_scatter(agg_rows, out_dir, log_x=args.log_x)

    # 补充扫描：仅用于 per-parameter 图，扩展 P / r0 / v 的覆盖范围。
    # 由各 config 中的 data.extra_list 指定（例如匙孔测试集过少时把验证集列在那里）。
    extra_rows: List[Dict] = []
    for cfg, label in zip(args.config, args.mode_label):
        extra_lists = _read_extra_list_from_config(cfg)
        if not extra_lists:
            continue
        rows = _scan_dataset(cfg, label, step=args.step,
                             override_test_list=extra_lists)
        extra_rows.extend(rows)

    params_rows = agg_rows
    if extra_rows:
        _save_csv(extra_rows, out_dir / "sensitivity_params_extra_per_sample.csv")
        extra_agg = _aggregate_by_h5(extra_rows)
        # 与主聚合结果按 (mode, h5) 去重合并，避免重复工况污染回归
        seen = {(r["mode"], r["h5"]) for r in agg_rows}
        merged = list(agg_rows) + [r for r in extra_agg if (r["mode"], r["h5"]) not in seen]
        _save_csv(merged, out_dir / "sensitivity_params_merged.csv")
        params_rows = merged

    _plot_per_param_grid(params_rows, out_dir)
