"""
plot_lpbf_mode_comparison.py
============================
Generates a 2xN academic-style figure comparing cross-sectional field snapshots
from two LPBF simulation modes (conduction vs. keyhole) at aligned physical times.

Overlays on each panel (restricted to the active / metal region):
  - gamma_liquid = 0.5 contour, clipped to y < 1e-4 m  (solid-liquid boundary)
  - alpha.air = 0.5 contour, active region only        (liquid-gas boundary)

Usage example (edit the CONFIG block below, then run):
    python plot_lpbf_mode_comparison.py
"""

# ============================================================
# CONFIG
# ============================================================
conduction_list = [
    "/home/ubuntu/MyAI/datasets/collectdata/easypool/easy/04_P80_E75_I5.h5",
    "/home/ubuntu/MyAI/datasets/collectdata/easypool/easy/05_P75_E75_I5.h5"
]
keyhole_list = [
    "/home/ubuntu/MyAI/datasets/collectdata/keyhole/keyhole/03_P210_E75_I1.h5",
    "/home/ubuntu/MyAI/datasets/collectdata/keyhole/keyhole/04_P215_E75_I2.h5"
]
CONDUCTION_H5 = conduction_list[0]
KEYHOLE_H5    = keyhole_list[1]
FIELD_NAME    = "alpha.air"
Z_SLICE       = 5e-4
SELECTED_TIMES = [15e-5, 30e-5, 45e-5, 60e-5]
OUTPUT_DIR    = "src/paint/result"
# ============================================================

import argparse
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
    with h5py.File(h5_path, "r") as f:
        block = f["mesh/block"][0].astype(int)
        bounds = f["mesh/bounds"][:].astype(np.float32)
        points = f["point"][:].astype(np.float32)
        time_arr = f["time"][:].astype(np.float64)
        grid_shape = (int(block[0]) + 1, int(block[1]) + 1, int(block[2]) + 1)
        available_fields = _list_state_fields(f)
    return {
        "grid_shape": grid_shape,
        "bounds": bounds,
        "points": points,
        "time": time_arr,
        "available_fields": available_fields,
    }


def _find_z_layer(points: np.ndarray, z_target: float) -> Tuple[np.ndarray, float]:
    z_vals = points[:, 2]
    unique_z = np.unique(z_vals)
    best_z = unique_z[np.argmin(np.abs(unique_z - z_target))]
    atol = max(abs(float(best_z)) * 1e-5, 1e-7)
    mask = np.abs(z_vals - best_z) <= atol
    return mask, float(best_z)


def _find_time_indices(time_arr: np.ndarray, target_times: List[float]) -> List[int]:
    indices = []
    for t in target_times:
        idx = int(np.argmin(np.abs(time_arr - t)))
        indices.append(idx)
        print(f"  target={t:.3e}s  ->  actual={time_arr[idx]:.6e}s  (index={idx})")
    return indices


def _interp_grid(pts_x, pts_y, vals, Xi, Yi, method="linear"):
    if vals is None:
        return None
    points = np.column_stack([pts_x, pts_y])
    values = np.asarray(vals)
    Z = None
    for m in ([method] + (["linear"] if method != "linear" else []) + ["nearest"]):
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


# ------------------------------------------------------------------
# Contour drawing utilities
# ------------------------------------------------------------------

def _contour_outlined(ax, X, Y, Z, levels, color, lw, ls='-', zorder=5,
                      outline_color='black', outline_lw=None):
    """Draw a contour with a black outline for visibility against colormaps."""
    if outline_lw is None:
        outline_lw = lw + 1.0
    ax.contour(X, Y, Z, levels=levels,
               colors=outline_color, linewidths=outline_lw,
               linestyles=ls, zorder=zorder)
    return ax.contour(X, Y, Z, levels=levels,
                      colors=color, linewidths=lw,
                      linestyles=ls, zorder=zorder + 1)



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
    "alpha.air":      r"$\alpha_{\rm air}$",
    "alpha.titanium": r"$\alpha_{\rm Ti}$",
    "gamma_liquid":   r"$\gamma_{\rm liq}$",
}


def _auto_cmap(field_name: str) -> str:
    return _FIELD_CMAP.get(field_name, "viridis")


def _auto_label(field_name: str) -> str:
    return _FIELD_LABEL.get(field_name, field_name)


def _compute_active_x_range_mm(field_img, x_mm, threshold=0.01, above=True, margin_frac=0.08):
    """
    Find the x-range (in mm) where field is above/below *threshold*.
    Returns (x_lo, x_hi) with a small margin, or None if no matching region.
    """
    col_val = np.nanmin(field_img, axis=0) if above else np.nanmax(field_img, axis=0)
    active_cols = (np.where(col_val > threshold)[0] if above
                   else np.where(col_val < threshold)[0])
    if len(active_cols) == 0:
        return None
    x_lo = x_mm[active_cols[0]]
    x_hi = x_mm[active_cols[-1]]
    margin = (x_hi - x_lo) * margin_frac
    return float(x_lo - margin), float(x_hi + margin)


def _active_x_range_from_T(T_img, x_mm, T_threshold=800.0, margin_frac=0.05):
    """Return (x_lo, x_hi) in mm where any row has T > T_threshold, or None."""
    col_max = np.nanmax(T_img, axis=0)
    active_cols = np.where(col_max > T_threshold)[0]
    if len(active_cols) == 0:
        return None
    x_lo = x_mm[active_cols[0]]
    x_hi = x_mm[active_cols[-1]]
    margin = (x_hi - x_lo) * margin_frac
    return float(x_lo - margin), float(x_hi + margin)


def _read_slice(
    h5_path: str,
    field_name: str,
    time_indices: List[int],
    z_target: float,
    res: int = 320,
    smooth_sigma: float = 0.8,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, float]:
    with h5py.File(h5_path, "r") as f:
        if "state" not in f or field_name not in f["state"]:
            avail = _list_state_fields(f)
            raise KeyError(
                f"Field '{field_name}' not found in {h5_path}.\n"
                f"Available fields under 'state': {avail}"
            )
        points = f["point"][:].astype(np.float32)
        mask, actual_z = _find_z_layer(points, z_target)
        slice_pts = points[mask]
        slice_idx = np.where(mask)[0]
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
            raw = f[fkey][ti, :, 0].astype(np.float32)
            vals = raw[slice_idx]
            img = _interp_grid(pts_x, pts_y, vals, Xi, Yi, method=interp_method)
            img = _smooth_for_plot(img, sigma=sigma)
            if is_phase and img is not None:
                img = np.clip(img, 0, 1)
            images.append(img.astype(np.float32) if img is not None else img)
    return images, x_lin, y_lin, actual_z


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
    if selected_times is None:
        selected_times = [0.0, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4]
    if cmap is None:
        cmap = _auto_cmap(field_name)

    # ---- read metadata ------------------------------------------------
    print("Reading metadata ...")
    meta_c = _read_meta(conduction_h5)
    meta_k = _read_meta(keyhole_h5)
    print(f"\nConduction grid shape: {meta_c['grid_shape']}")
    print(f"Keyhole    grid shape: {meta_k['grid_shape']}")

    # ---- find time indices --------------------------------------------
    print(f"\nConduction time indices:")
    t_idx_c = _find_time_indices(meta_c["time"], selected_times)
    print(f"\nKeyhole time indices:")
    t_idx_k = _find_time_indices(meta_k["time"], selected_times)

    # ---- read main field slices ---------------------------------------
    print(f"\nReading conduction slices (field='{field_name}') ...")
    imgs_c, x_c, y_c, actual_z_c = _read_slice(conduction_h5, field_name, t_idx_c, z_slice)
    print(f"  z = {actual_z_c:.6e} m  (grid {len(y_c)}x{len(x_c)})")

    print(f"\nReading keyhole slices (field='{field_name}') ...")
    imgs_k, x_k, y_k, actual_z_k = _read_slice(keyhole_h5, field_name, t_idx_k, z_slice)
    print(f"  z = {actual_z_k:.6e} m  (grid {len(y_k)}x{len(x_k)})")

    # ---- read alpha.air -----------------------------------------------
    has_air = False
    imgs_air_c, imgs_air_k = None, None
    if field_name == "alpha.air":
        imgs_air_c, imgs_air_k = imgs_c, imgs_k
        has_air = True
    elif "alpha.air" in meta_c.get("available_fields", []) and \
         "alpha.air" in meta_k.get("available_fields", []):
        try:
            print("\nReading alpha.air for interface contour ...")
            imgs_air_c, _, _, _ = _read_slice(conduction_h5, "alpha.air", t_idx_c, z_slice)
            imgs_air_k, _, _, _ = _read_slice(keyhole_h5, "alpha.air", t_idx_k, z_slice)
            has_air = True
        except Exception as e:
            warnings.warn(f"Could not read alpha.air: {e}")

    # ---- read T for active-region mask (T > 800 K) -------------------
    has_temp = False
    imgs_T_c, imgs_T_k = None, None
    if "T" in meta_c.get("available_fields", []) and \
       "T" in meta_k.get("available_fields", []):
        try:
            print("\nReading T for active-region mask ...")
            imgs_T_c, _, _, _ = _read_slice(conduction_h5, "T", t_idx_c, z_slice)
            imgs_T_k, _, _, _ = _read_slice(keyhole_h5, "T", t_idx_k, z_slice)
            has_temp = True
        except Exception as e:
            warnings.warn(f"Could not read T: {e}")

    # ---- read gamma_liquid --------------------------------------------
    has_meltpool = False
    imgs_gamma_c, imgs_gamma_k = None, None
    if "gamma_liquid" in meta_c.get("available_fields", []) and \
       "gamma_liquid" in meta_k.get("available_fields", []):
        try:
            print("\nReading gamma_liquid for melt-pool contour ...")
            imgs_gamma_c, _, _, _ = _read_slice(conduction_h5, "gamma_liquid", t_idx_c, z_slice)
            imgs_gamma_k, _, _, _ = _read_slice(keyhole_h5, "gamma_liquid", t_idx_k, z_slice)
            has_meltpool = True
        except Exception as e:
            warnings.warn(f"Could not read gamma_liquid: {e}")

    # ---- thresholds ---------------------------------------------------
    AIR_THRESHOLD = 0.99   # alpha.air above this → pure air (excluded from active region)
    Y_GAMMA_CLIP  = 1e-4   # y [m] above which the gamma=0.5 contour is clipped

    # ---- unified colorbar range ---------------------------------------
    all_vals = np.concatenate(
        [img.ravel() for img in imgs_c + imgs_k if not np.all(np.isnan(img))]
    )
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    print(f"\nColorbar range: [{vmin:.4g}, {vmax:.4g}]")
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ---- contour meshgrids --------------------------------------------
    Xc, Yc = np.meshgrid(x_c, y_c)
    Xk, Yk = np.meshgrid(x_k, y_k)

    # ---- figure layout ------------------------------------------------
    n_cols = len(selected_times)
    x_span_c = float(x_c[-1] - x_c[0]) if len(x_c) > 1 else 1.0
    y_span_c = float(y_c[-1] - y_c[0]) if len(y_c) > 1 else 1.0
    y_span_k = float(y_k[-1] - y_k[0]) if len(y_k) > 1 else 1.0
    x_span = x_span_c
    aspect_c = y_span_c / x_span
    aspect_k = y_span_k / x_span

    gs_left, gs_right = 0.06, 0.93
    gs_top, gs_bottom = 0.90, 0.22
    cbar_width_ratio = 0.04
    hspace = 0.30
    wspace = 0.05

    grid_rel_w = gs_right - gs_left
    grid_rel_h = gs_top - gs_bottom
    sum_aspects = aspect_c + aspect_k
    sum_w_ratios = n_cols + cbar_width_ratio

    fig_w = 7.2   # about 183 mm — suitable for double-column journal width
    fig_h = fig_w * sum_aspects * grid_rel_w / (grid_rel_h * sum_w_ratios)

    # ---- academic rcParams --------------------------------------------
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset":   "dejavuserif",
        "font.size":          7,
        "axes.linewidth":     0.4,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.top":          True,
        "ytick.right":        True,
        "xtick.major.size":   2.0,
        "ytick.major.size":   2.0,
        "xtick.major.width":  0.4,
        "ytick.major.width":  0.4,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.labelsize":    6,
        "ytick.labelsize":    6,
        "axes.labelsize":     7,
        "axes.titlesize":     7,
        "legend.fontsize":    5.5,
        "legend.frameon":     True,
        "legend.edgecolor":   "0.6",
        "legend.fancybox":    False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.bbox":       "tight",
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
            return rf"$t$ = {int(round(t_us))} $\mu$s"
        return rf"$t$ = {t_us:.1f} $\mu$s"

    # ---- build legend handles -----------------------------------------
    contour_handles, contour_labels = [], []

    # Solid-liquid boundary (gamma_liquid = 0.5, y < 1e-4 m)
    if has_meltpool:
        h = Line2D([0], [0], color='#E05050', lw=0.9, ls='-',
                   path_effects=[Stroke(linewidth=1.5, foreground='white'),
                                 Normal()])
        contour_handles.append(h)
        contour_labels.append(r"Solid--liquid interface ($\gamma_{\rm liq}=0.5$, $y < 0.1$ mm)")

    # Liquid-gas boundary (alpha.air = 0.5)
    if has_air:
        h = Line2D([0], [0], color='#00CFFF', lw=0.9, ls='--',
                   path_effects=[Stroke(linewidth=1.5, foreground='white'),
                                 Normal()])
        contour_handles.append(h)
        contour_labels.append(r"Liquid--gas interface ($\alpha_{\rm air}=0.5$)")

    axes_c: List[plt.Axes] = []
    axes_k: List[plt.Axes] = []

    for col, (t_target, img_c, img_k) in enumerate(
        zip(selected_times, imgs_c, imgs_k)
    ):
        ax_c = fig.add_subplot(gs[0, col])
        ax_k = fig.add_subplot(gs[1, col])
        axes_c.append(ax_c)
        axes_k.append(ax_k)

        ext_c = [x_c_mm[0], x_c_mm[-1], y_c_mm[0], y_c_mm[-1]]
        ext_k = [x_k_mm[0], x_k_mm[-1], y_k_mm[0], y_k_mm[-1]]

        # ---- main field image -----------------------------------------
        ax_c.imshow(img_c, origin="lower", aspect="auto",
                    extent=ext_c, cmap=cmap, norm=norm, interpolation="bicubic")
        ax_k.imshow(img_k, origin="lower", aspect="auto",
                    extent=ext_k, cmap=cmap, norm=norm, interpolation="bicubic")

        # ---- contour overlays -----------------------------------------
        # Determine T-based active x-range for liquid-gas contour
        T_range_c = None
        T_range_k = None
        if has_temp and imgs_T_c is not None:
            T_range_c = _active_x_range_from_T(imgs_T_c[col], x_c_mm)
        if has_temp and imgs_T_k is not None:
            T_range_k = _active_x_range_from_T(imgs_T_k[col], x_k_mm)

        for ax, Xg, Yg, x_mm_arr, air_list, gamma_list, T_active_range in [
            (ax_c, Xc, Yc, x_c_mm, imgs_air_c, imgs_gamma_c, T_range_c),
            (ax_k, Xk, Yk, x_k_mm, imgs_air_k, imgs_gamma_k, T_range_k),
        ]:
            # Active x-range for gamma contour (melt-pool region)
            gamma_active_range = None
            if has_meltpool and gamma_list is not None:
                gamma_active_range = _compute_active_x_range_mm(
                    gamma_list[col], x_mm_arr, threshold=0.01, above=True)
            elif has_air and air_list is not None:
                gamma_active_range = _compute_active_x_range_mm(
                    air_list[col], x_mm_arr, threshold=0.99, above=False)

            Xg_mm = Xg * 1e3
            Yg_mm = Yg * 1e3

            # --- solid-liquid boundary: gamma_liquid = 0.5, clipped to y < Y_GAMMA_CLIP ---
            if has_meltpool and gamma_list is not None:
                if gamma_active_range is not None:
                    x_lo, x_hi = gamma_active_range
                    col_mask_g = (x_mm_arr >= x_lo) & (x_mm_arr <= x_hi)
                    Xg_sub_g = Xg_mm[:, col_mask_g]
                    Yg_sub_g = Yg_mm[:, col_mask_g]
                    gamma_sub = gamma_list[col][:, col_mask_g]
                    Yg_for_mask = Yg[:, col_mask_g]
                else:
                    Xg_sub_g, Yg_sub_g = Xg_mm, Yg_mm
                    gamma_sub = gamma_list[col]
                    Yg_for_mask = Yg
                gamma_clipped = np.where(Yg_for_mask < Y_GAMMA_CLIP, gamma_sub, np.nan)
                try:
                    _contour_outlined(
                        ax, Xg_sub_g, Yg_sub_g, gamma_clipped,
                        levels=[0.5],
                        color='#E05050', lw=0.9, ls='-', zorder=7,
                        outline_color='white', outline_lw=1.8,
                    )
                except Exception:
                    pass

            # --- liquid-gas boundary: alpha.air = 0.5, restricted to T > 800 x-range ---
            if has_air and air_list is not None:
                if T_active_range is not None:
                    x_lo, x_hi = T_active_range
                    col_mask_a = (x_mm_arr >= x_lo) & (x_mm_arr <= x_hi)
                    Xg_sub_a = Xg_mm[:, col_mask_a]
                    Yg_sub_a = Yg_mm[:, col_mask_a]
                    air_sub = air_list[col][:, col_mask_a]
                else:
                    Xg_sub_a, Yg_sub_a = Xg_mm, Yg_mm
                    air_sub = air_list[col]
                try:
                    _contour_outlined(
                        ax, Xg_sub_a, Yg_sub_a, air_sub,
                        levels=[0.5],
                        color='#00CFFF', lw=0.9, ls='--', zorder=7,
                        outline_color='white', outline_lw=1.8,
                    )
                except Exception:
                    pass

        # ---- panel labels (a)-(h) ------------------------------------
        panel_idx = col + 0 * n_cols
        panel_idx2 = col + 1 * n_cols
        letters = "abcdefghijklmnop"
        if panel_idx < len(letters):
            ax_c.text(0.03, 0.93, f"({letters[panel_idx]})",
                      transform=ax_c.transAxes, fontsize=7, fontweight='bold',
                      va='top', ha='left', color='white',
                      path_effects=[Stroke(linewidth=1.5, foreground='black'),
                                    Normal()])
        if panel_idx2 < len(letters):
            ax_k.text(0.03, 0.93, f"({letters[panel_idx2]})",
                      transform=ax_k.transAxes, fontsize=7, fontweight='bold',
                      va='top', ha='left', color='white',
                      path_effects=[Stroke(linewidth=1.5, foreground='black'),
                                    Normal()])

        # column title
        ax_c.set_title(_fmt_time(t_target), pad=3, fontsize=7)

        # x-axis: bottom row shows label on every column
        ax_c.tick_params(labelbottom=False)
        ax_k.set_xlabel(r"$x$ (mm)", labelpad=1)

        # y-axis: only leftmost column shows tick labels and axis label
        if col == 0:
            ax_c.set_ylabel(r"$y$ (mm)", labelpad=2)
            ax_k.set_ylabel(r"$y$ (mm)", labelpad=2)
        else:
            ax_c.tick_params(labelleft=False)
            ax_k.tick_params(labelleft=False)

        for ax in (ax_c, ax_k):
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

    # ---- row labels --------------------------------------------------
    axes_c[0].annotate(
        "Conduction mode",
        xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-0.36, 0.5), textcoords="axes fraction",
        fontsize=7.5, fontweight="bold", fontstyle="italic",
        va="center", ha="center", rotation=90,
    )
    axes_k[0].annotate(
        "Keyhole mode",
        xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-0.36, 0.5), textcoords="axes fraction",
        fontsize=7.5, fontweight="bold", fontstyle="italic",
        va="center", ha="center", rotation=90,
    )

    # ---- legend at figure bottom ------------------------------------
    if contour_handles:
        fig.legend(
            handles=contour_handles, labels=contour_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.06),
            ncol=len(contour_handles),
            frameon=True, framealpha=0.92, fontsize=6.5,
            borderpad=0.4, labelspacing=0.5, handlelength=2.2,
            handletextpad=0.5, borderaxespad=0.3,
            edgecolor='0.5',
        )

    # ---- colorbar ----------------------------------------------------
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(_auto_label(field_name), fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=6, width=0.4, length=2)
    cbar.outline.set_linewidth(0.4)
    # Set clean colorbar ticks
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune="both"))

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
    parser.add_argument("--conduction", default=CONDUCTION_H5)
    parser.add_argument("--keyhole", default=KEYHOLE_H5)
    parser.add_argument("--field", default=FIELD_NAME)
    parser.add_argument("--z_slice", type=float, default=Z_SLICE)
    parser.add_argument("--times", type=float, nargs="+", default=SELECTED_TIMES)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--cmap", default=None)
    parser.add_argument("--dpi", type=int, default=300)
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
