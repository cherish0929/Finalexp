"""
plot_lpbf_mode_comparison.py
============================
Generates a 2xN academic-style figure comparing cross-sectional field snapshots
from two LPBF simulation modes (conduction vs. keyhole) at aligned physical times.

Overlays on each panel:
  - alpha.air = 0.5 contour (air-metal interface, white dashed)
  - gamma_liquid * (1 - alpha.air) = 0.5 contour (melt-pool boundary, gold solid)

Usage example (edit the CONFIG block below, then run):
    python plot_lpbf_mode_comparison.py

Or via command line:
    python plot_lpbf_mode_comparison.py \
        --conduction /path/to/conduction.h5 \
        --keyhole    /path/to/keyhole.h5 \
        --field      alpha.air \
        --z_slice    5e-4 \
        --times      0.0 2e-5 4e-5 6e-5 8e-5 1e-4 \
        --output_dir ./output
"""

# ============================================================
# CONFIG — edit here when running without CLI arguments
# ============================================================
conduction_list = [
    "/home/ubuntu/MyAI/datasets/collectdata/easypool/easy/04_P80_E75_I5.h5",
    "/home/ubuntu/MyAI/datasets/collectdata/easypool/easy/05_P75_E75_I5.h5"
]
keyhole_list = [
    "/home/ubuntu/MyAI/datasets/collectdata/keyhole/keyhole/03_P210_E75_I1.h5",
    "/home/ubuntu/MyAI/datasets/collectdata/keyhole/keyhole/04_P215_E75_I2.h5"
]
CONDUCTION_H5 = conduction_list[1]
KEYHOLE_H5    = keyhole_list[1]
FIELD_NAME    = "alpha.air"          # "T", "alpha.air", "alpha.titanium", "gamma_liquid", ...
Z_SLICE       = 5e-4                 # target z coordinate (metres)
SELECTED_TIMES = [15e-5, 30e-5, 45e-5, 60e-5]
OUTPUT_DIR    = "src/paint/result"
# ============================================================

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.patheffects import Stroke, Normal
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ------------------------------------------------------------------
# HDF5 helpers
# ------------------------------------------------------------------

def _list_state_fields(f: h5py.File) -> List[str]:
    if "state" not in f:
        return []
    return list(f["state"].keys())


def _read_meta(h5_path: str) -> Dict:
    """Read grid metadata from an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        block = f["mesh/block"][0].astype(int)
        bounds = f["mesh/bounds"][:].astype(np.float32)   # [3, 2]
        points = f["point"][:].astype(np.float32)          # [N, 3]
        time_arr = f["time"][:].astype(np.float64)

        grid_shape = (int(block[0]) + 1, int(block[1]) + 1, int(block[2]) + 1)

        available_fields = _list_state_fields(f)

    return {
        "grid_shape": grid_shape,   # (nx, ny, nz)
        "bounds": bounds,
        "points": points,           # [N, 3]  columns: x, y, z
        "time": time_arr,
        "available_fields": available_fields,
    }


def _find_z_layer(points: np.ndarray, z_target: float) -> Tuple[np.ndarray, float]:
    """
    Return boolean mask of points whose z is closest to z_target,
    and the actual z value selected.
    Uses float32-safe tolerance to avoid NaN gaps from precision mismatch.
    """
    z_vals = points[:, 2]
    unique_z = np.unique(z_vals)
    best_z = unique_z[np.argmin(np.abs(unique_z - z_target))]
    atol = max(abs(float(best_z)) * 1e-5, 1e-7)
    mask = np.abs(z_vals - best_z) <= atol
    return mask, float(best_z)


def _find_time_indices(time_arr: np.ndarray, target_times: List[float]) -> List[int]:
    """For each target time, return the index of the closest entry in time_arr."""
    indices = []
    for t in target_times:
        idx = int(np.argmin(np.abs(time_arr - t)))
        indices.append(idx)
        print(f"  target={t:.3e}s  ->  actual={time_arr[idx]:.6e}s  (index={idx})")
    return indices


def _interp_grid(pts_x, pts_y, vals, Xi, Yi, method="linear"):
    """Interpolate scatter values onto a regular grid, with nearest-fill fallback."""
    if vals is None:
        return None
    points = np.column_stack([pts_x, pts_y])
    values = np.asarray(vals)

    interp_try = [method]
    if method != "linear":
        interp_try.append("linear")
    interp_try.append("nearest")

    Z = None
    for m in interp_try:
        try:
            Z = griddata(points, values, (Xi, Yi), method=m)
        except Exception:
            Z = None
        if Z is not None and not np.all(np.isnan(Z)):
            break

    if Z is None:
        return None

    if np.isnan(Z).any():
        try:
            Z_nearest = griddata(points, values, (Xi, Yi), method="nearest")
            Z = np.where(np.isnan(Z), Z_nearest, Z)
        except Exception:
            pass
    return Z


def _smooth_for_plot(Z, sigma):
    """Apply Gaussian smoothing, preserving NaN regions."""
    if Z is None or sigma is None or sigma <= 0:
        return Z
    if np.all(np.isnan(Z)):
        return Z
    nan_mask = np.isnan(Z)
    fill_value = np.nanmedian(Z)
    if not np.isfinite(fill_value):
        fill_value = 0.0
    Z_fill = np.where(nan_mask, fill_value, Z)
    Z_smooth = gaussian_filter(Z_fill, sigma=sigma, mode="nearest")
    Z_smooth[nan_mask] = np.nan
    return Z_smooth


def _contour_outlined(ax, X, Y, Z, levels, color, lw, ls='-', zorder=5):
    """
    Draw contours with a black outline for visibility on any colormap background.
    First pass: thicker black line (outline).  Second pass: coloured core line.
    """
    ax.contour(X, Y, Z, levels=levels,
               colors='black', linewidths=lw + 1.5, linestyles=ls, zorder=zorder)
    return ax.contour(X, Y, Z, levels=levels,
                      colors=color, linewidths=lw, linestyles=ls, zorder=zorder + 1)


def _read_slice(
    h5_path: str,
    field_name: str,
    time_indices: List[int],
    z_target: float,
    res: int = 320,
    smooth_sigma: float = 0.8,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, float]:
    """
    Read 2-D xy slices at z ~ z_target for each requested time index.
    Interpolates scatter data onto a high-resolution regular grid and
    applies Gaussian smoothing.

    Returns
    -------
    images   : list of 2-D arrays (res, res), one per time index
    x_lin    : 1-D linspace x values used for the grid (metres)
    y_lin    : 1-D linspace y values used for the grid (metres)
    actual_z : the z value actually used
    """
    with h5py.File(h5_path, "r") as f:
        if "state" not in f or field_name not in f["state"]:
            avail = _list_state_fields(f)
            raise KeyError(
                f"Field '{field_name}' not found in {h5_path}.\n"
                f"Available fields under 'state': {avail}"
            )

        points = f["point"][:].astype(np.float32)
        mask, actual_z = _find_z_layer(points, z_target)

        slice_pts = points[mask]          # [M, 3]
        slice_idx = np.where(mask)[0]     # global point indices

        if slice_pts.shape[0] == 0:
            raise ValueError(f"No points found near z={z_target} in {h5_path}.")

        pts_x = slice_pts[:, 0]
        pts_y = slice_pts[:, 1]

        x_lin = np.linspace(pts_x.min(), pts_x.max(), res)
        y_lin = np.linspace(pts_y.min(), pts_y.max(), res)
        Xi, Yi = np.meshgrid(x_lin, y_lin)

        is_phase = any(k in field_name for k in ["alpha", "gamma", "frac"])
        interp_method = "linear"
        sigma = 0.5 if is_phase else smooth_sigma

        fkey = f"state/{field_name}"
        images = []
        for ti in time_indices:
            raw = f[fkey][ti, :, 0].astype(np.float32)   # [N_total]
            vals = raw[slice_idx]                          # [M]

            img = _interp_grid(pts_x, pts_y, vals, Xi, Yi, method=interp_method)
            img = _smooth_for_plot(img, sigma=sigma)

            if is_phase and img is not None:
                img = np.clip(img, 0, 1)

            images.append(img.astype(np.float32) if img is not None else img)

    return images, x_lin, y_lin, actual_z


# ------------------------------------------------------------------
# Colormap / label helpers
# ------------------------------------------------------------------

_FIELD_CMAP = {
    "T":              "inferno",
    "alpha.air":      "viridis",
    "alpha.titanium": "cividis",
    "gamma_liquid":   "plasma",
}
_FIELD_LABEL = {
    "T":              "Temperature (K)",
    "alpha.air":      "Air volume fraction",
    "alpha.titanium": "Titanium volume fraction",
    "gamma_liquid":   "Liquid fraction",
}


def _auto_cmap(field_name: str) -> str:
    return _FIELD_CMAP.get(field_name, "viridis")


def _auto_label(field_name: str) -> str:
    return _FIELD_LABEL.get(field_name, field_name)


# ------------------------------------------------------------------
# Main plotting function
# ------------------------------------------------------------------

def plot_comparison(
    conduction_h5: str,
    keyhole_h5: str,
    field_name: str = "alpha.air",
    z_slice: float = 5e-4,
    selected_times: Optional[List[float]] = None,
    output_dir: str = ".",
    cmap: Optional[str] = None,
    dpi: int = 300,
) -> str:
    """
    Build and save the 2xN comparison figure.

    Returns the path to the saved PNG file.
    """
    if selected_times is None:
        selected_times = [0.0, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4]

    if cmap is None:
        cmap = _auto_cmap(field_name)

    # ---- read metadata ------------------------------------------------
    print("Reading metadata ...")
    meta_c = _read_meta(conduction_h5)
    meta_k = _read_meta(keyhole_h5)

    print(f"\nConduction grid shape (nx,ny,nz): {meta_c['grid_shape']}")
    print(f"Keyhole    grid shape (nx,ny,nz): {meta_k['grid_shape']}")

    # ---- find time indices --------------------------------------------
    print(f"\nConduction time indices for selected_times:")
    t_idx_c = _find_time_indices(meta_c["time"], selected_times)
    print(f"\nKeyhole time indices for selected_times:")
    t_idx_k = _find_time_indices(meta_k["time"], selected_times)

    # ---- read main field slices ---------------------------------------
    print(f"\nReading conduction slices (field='{field_name}', z~{z_slice:.3e}) ...")
    imgs_c, x_c, y_c, actual_z_c = _read_slice(conduction_h5, field_name, t_idx_c, z_slice)
    print(f"  Actual z selected (conduction): {actual_z_c:.6e} m  "
          f"(grid res: {len(y_c)}x{len(x_c)})")

    print(f"\nReading keyhole slices (field='{field_name}', z~{z_slice:.3e}) ...")
    imgs_k, x_k, y_k, actual_z_k = _read_slice(keyhole_h5, field_name, t_idx_k, z_slice)
    print(f"  Actual z selected (keyhole):    {actual_z_k:.6e} m  "
          f"(grid res: {len(y_k)}x{len(x_k)})")

    # ---- read alpha.air for the air-metal interface contour -----------
    has_air = False
    imgs_air_c, imgs_air_k = None, None
    if field_name == "alpha.air":
        imgs_air_c, imgs_air_k = imgs_c, imgs_k
        has_air = True
    elif "alpha.air" in meta_c.get("available_fields", []) and \
         "alpha.air" in meta_k.get("available_fields", []):
        try:
            print(f"\nReading alpha.air for interface contour ...")
            imgs_air_c, _, _, _ = _read_slice(conduction_h5, "alpha.air", t_idx_c, z_slice)
            imgs_air_k, _, _, _ = _read_slice(keyhole_h5, "alpha.air", t_idx_k, z_slice)
            has_air = True
        except Exception as e:
            warnings.warn(f"Could not read alpha.air: {e}")

    # ---- read gamma_liquid for melt-pool boundary contour -------------
    has_meltpool = False
    imgs_gamma_c, imgs_gamma_k = None, None
    if "gamma_liquid" in meta_c.get("available_fields", []) and \
       "gamma_liquid" in meta_k.get("available_fields", []):
        try:
            print(f"\nReading gamma_liquid for melt-pool contour ...")
            imgs_gamma_c, _, _, _ = _read_slice(conduction_h5, "gamma_liquid", t_idx_c, z_slice)
            imgs_gamma_k, _, _, _ = _read_slice(keyhole_h5, "gamma_liquid", t_idx_k, z_slice)
            has_meltpool = True
        except Exception as e:
            warnings.warn(f"Could not read gamma_liquid: {e}")

    # ---- compute melt-pool field: gamma_liquid * (1 - alpha.air) ------
    meltpool_c, meltpool_k = None, None
    if has_air and has_meltpool:
        meltpool_c = [
            np.clip(g * (1.0 - a), 0, 1)
            for g, a in zip(imgs_gamma_c, imgs_air_c)
        ]
        meltpool_k = [
            np.clip(g * (1.0 - a), 0, 1)
            for g, a in zip(imgs_gamma_k, imgs_air_k)
        ]
        print("  Melt-pool field computed: gamma_liquid * (1 - alpha.air)")

    # ---- unified colorbar range ---------------------------------------
    all_vals = np.concatenate(
        [img.ravel() for img in imgs_c + imgs_k if not np.all(np.isnan(img))]
    )
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    print(f"\nColorbar range: [{vmin:.4g}, {vmax:.4g}]")

    norm = Normalize(vmin=vmin, vmax=vmax)

    # ---- contour meshgrids (shared across all fields from same file) ---
    Xc, Yc = np.meshgrid(x_c, y_c)   # conduction
    Xk, Yk = np.meshgrid(x_k, y_k)   # keyhole

    # ---- figure layout ------------------------------------------------
    n_cols = len(selected_times)

    # Physical spans (metres -> mm for display)
    x_span_c = float(x_c[-1] - x_c[0]) if len(x_c) > 1 else 1.0
    y_span_c = float(y_c[-1] - y_c[0]) if len(y_c) > 1 else 1.0
    y_span_k = float(y_k[-1] - y_k[0]) if len(y_k) > 1 else 1.0
    x_span = x_span_c   # same for both datasets (~0.56 mm)

    # Data aspect ratio (height / width) for each dataset
    aspect_c = y_span_c / x_span
    aspect_k = y_span_k / x_span

    # GridSpec margins (fractions of figure)
    gs_left, gs_right = 0.08, 0.94
    gs_top, gs_bottom = 0.93, 0.06
    cbar_width_ratio = 0.05
    hspace = 0.35
    wspace = 0.06

    grid_rel_w = gs_right - gs_left
    grid_rel_h = gs_top - gs_bottom
    sum_aspects = aspect_c + aspect_k
    sum_w_ratios = n_cols + cbar_width_ratio

    fig_w = 12.0
    fig_h = fig_w * sum_aspects * grid_rel_w / (grid_rel_h * sum_w_ratios)

    # ---- academic-style rcParams --------------------------------------
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size":        8,
        "axes.linewidth":   0.6,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.labelsize":  6.5,
        "ytick.labelsize":  6.5,
        "axes.labelsize":   7.5,
        "axes.titlesize":   8,
        "legend.fontsize":  7,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
    })

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    gs = GridSpec(
        2, n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [cbar_width_ratio],
        height_ratios=[aspect_c, aspect_k],
        left=gs_left, right=gs_right,
        top=gs_top, bottom=gs_bottom,
        hspace=hspace, wspace=wspace,
    )

    x_c_mm = x_c * 1e3
    y_c_mm = y_c * 1e3
    x_k_mm = x_k * 1e3
    y_k_mm = y_k * 1e3

    def _fmt_time(t: float) -> str:
        t_us = t * 1e6
        if abs(t_us - round(t_us)) < 1e-3:
            return f"t = {int(round(t_us))} µs"
        return f"t = {t_us:.1f} µs"

    axes_c: List[plt.Axes] = []
    axes_k: List[plt.Axes] = []

    # ---- contour legend handles (built once, added to first panel) ----
    outline_effect = [Stroke(linewidth=2.2, foreground='black'), Normal()]
    contour_handles = []
    contour_labels = []
    if has_air:
        h_air = Line2D([0], [0], color='white', lw=1.0, ls='--')
        h_air.set_path_effects(outline_effect)
        contour_handles.append(h_air)
        contour_labels.append(r"$\mathregular{\alpha_{air}}=0.5$")
    if has_meltpool:
        h_mp = Line2D([0], [0], color='#FFD700', lw=1.0, ls='-')
        h_mp.set_path_effects(outline_effect)
        contour_handles.append(h_mp)
        contour_labels.append(
            r"$\mathregular{\gamma_{liq}(1-\alpha_{air})}=0.5$")

    for col, (t_target, img_c, img_k) in enumerate(
        zip(selected_times, imgs_c, imgs_k)
    ):
        ax_c = fig.add_subplot(gs[0, col])
        ax_k = fig.add_subplot(gs[1, col])
        axes_c.append(ax_c)
        axes_k.append(ax_k)

        # extent: [x_min, x_max, y_min, y_max] in mm
        ext_c = [x_c_mm[0], x_c_mm[-1], y_c_mm[0], y_c_mm[-1]]
        ext_k = [x_k_mm[0], x_k_mm[-1], y_k_mm[0], y_k_mm[-1]]

        # ---- main field image -----------------------------------------
        ax_c.imshow(
            img_c, origin="lower", aspect="auto",
            extent=ext_c, cmap=cmap, norm=norm, interpolation="bicubic",
        )
        ax_k.imshow(
            img_k, origin="lower", aspect="auto",
            extent=ext_k, cmap=cmap, norm=norm, interpolation="bicubic",
        )

        # ---- contour overlays (use metre coordinates for meshgrid) ----
        for ax, Xg, Yg, air_img_list, mp_img_list in [
            (ax_c, Xc, Yc, imgs_air_c, meltpool_c),
            (ax_k, Xk, Yk, imgs_air_k, meltpool_k),
        ]:
            # alpha.air = 0.5  (air-metal interface)
            if has_air and air_img_list is not None:
                _contour_outlined(
                    ax, Xg * 1e3, Yg * 1e3, air_img_list[col],
                    levels=[0.5], color='white', lw=1.0, ls='--', zorder=6,
                )

            # melt-pool boundary  (gamma_liquid * (1-alpha.air) = 0.5)
            if has_meltpool and mp_img_list is not None:
                _contour_outlined(
                    ax, Xg * 1e3, Yg * 1e3, mp_img_list[col],
                    levels=[0.5], color='#FFD700', lw=1.0, ls='-', zorder=6,
                )

        # column title (top row only)
        ax_c.set_title(_fmt_time(t_target), pad=4, fontsize=8, fontweight='normal')

        # x-axis label only on bottom row
        ax_k.set_xlabel("x (mm)", labelpad=1)
        ax_c.set_xticklabels([])

        # y-axis label only on leftmost column
        if col == 0:
            ax_c.set_ylabel("y (mm)", labelpad=1)
            ax_k.set_ylabel("y (mm)", labelpad=1)
        else:
            ax_c.set_yticklabels([])
            ax_k.set_yticklabels([])

        # tick formatting
        for ax in (ax_c, ax_k):
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    # ---- row labels on the left side of each row ----------------------
    axes_c[0].annotate(
        "Conduction mode",
        xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-0.55, 0.5), textcoords="axes fraction",
        fontsize=9, fontweight="bold", va="center", ha="center",
        rotation=90,
    )
    axes_k[0].annotate(
        "Keyhole mode",
        xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-0.55, 0.5), textcoords="axes fraction",
        fontsize=9, fontweight="bold", va="center", ha="center",
        rotation=90,
    )

    # ---- contour legend (top-left panel) ------------------------------
    if contour_handles:
        axes_c[0].legend(
            handles=contour_handles, labels=contour_labels,
            loc='upper right', framealpha=0.85, fontsize=6.5,
            borderpad=0.4, labelspacing=0.3, handlelength=1.5,
            handletextpad=0.5, borderaxespad=0.4,
        )

    # ---- colorbar -----------------------------------------------------
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(_auto_label(field_name), fontsize=7.5, labelpad=3)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)
    cbar.outline.set_linewidth(0.5)

    # ---- save ---------------------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_field = field_name.replace(".", "_")
    out_path = out_dir / f"lpbf_conduction_keyhole_comparison_{safe_field}.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved -> {out_path}")
    return str(out_path)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LPBF conduction vs. keyhole mode cross-section comparison."
    )
    parser.add_argument("--conduction", default=CONDUCTION_H5,
                        help="Path to conduction-mode HDF5 file.")
    parser.add_argument("--keyhole", default=KEYHOLE_H5,
                        help="Path to keyhole-mode HDF5 file.")
    parser.add_argument("--field", default=FIELD_NAME,
                        help="Physical field name, e.g. T, alpha.air, gamma_liquid.")
    parser.add_argument("--z_slice", type=float, default=Z_SLICE,
                        help="Target z coordinate for the cross-section (metres).")
    parser.add_argument("--times", type=float, nargs="+", default=SELECTED_TIMES,
                        help="Target physical times (seconds).")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Directory to save the output PNG.")
    parser.add_argument("--cmap", default=None,
                        help="Matplotlib colormap name (auto-selected if omitted).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output resolution in DPI.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    for label, path in [("Conduction", args.conduction), ("Keyhole", args.keyhole)]:
        if not Path(path).exists():
            print(f"ERROR: {label} HDF5 file not found: {path}", file=sys.stderr)
            sys.exit(1)

    plot_comparison(
        conduction_h5=args.conduction,
        keyhole_h5=args.keyhole,
        field_name=args.field,
        z_slice=args.z_slice,
        selected_times=args.times,
        output_dir=args.output_dir,
        cmap=args.cmap,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
