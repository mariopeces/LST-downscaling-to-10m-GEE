"""Figure generators for LST / urban heat island methodology reports."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from rasterio.windows import Window, from_bounds


BRAND = {
    "dark": "#1b373f",
    "forest": "#426331",
    "olive": "#879753",
    "lime": "#bcbe76",
    "cream": "#fcf5e3",
    "g100": "#f7f7f5",
    "g200": "#e8e8e4",
    "g300": "#d0d0c8",
    "g500": "#8a8a80",
    "g700": "#4a4a44",
    "txt": "#2c2c28",
}

LST_CMAP = LinearSegmentedColormap.from_list(
    "darwin_lst",
    ["#234c6a", "#5f9fb9", "#c9dfb1", "#f4dc83", "#e68b53", "#a83a32"],
)
NDVI_CMAP = LinearSegmentedColormap.from_list(
    "darwin_ndvi", ["#7a4b2a", "#efe6b2", "#8bb66b", "#1f6b45"]
)
NDWI_CMAP = LinearSegmentedColormap.from_list(
    "darwin_ndwi", ["#7b5a42", "#ede7ce", "#86bdd0", "#245a7a"]
)
NDBI_CMAP = LinearSegmentedColormap.from_list(
    "darwin_ndbi", ["#205c58", "#d8dfb0", "#d28a52", "#8b2f2c"]
)


def _valid(data: np.ndarray, nodata: float | None) -> np.ndarray:
    arr = data.astype("float32", copy=False)
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= arr != nodata
    return np.where(mask, arr, np.nan)


def read_raster_stats(path: str | Path) -> dict[str, float | int | str]:
    path = Path(path)
    with rasterio.open(path) as src:
        data = _valid(src.read(1), src.nodata)
        vals = data[np.isfinite(data)]
        bounds = src.bounds
        crs = src.crs.to_string() if src.crs else ""
        res_x, res_y = src.res
        width = src.width
        height = src.height

    if vals.size == 0:
        raise RuntimeError(f"Raster sin píxeles válidos: {path}")

    return {
        "path": str(path),
        "min": float(np.nanmin(vals)),
        "p02": float(np.nanpercentile(vals, 2)),
        "p05": float(np.nanpercentile(vals, 5)),
        "p25": float(np.nanpercentile(vals, 25)),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "p75": float(np.nanpercentile(vals, 75)),
        "p95": float(np.nanpercentile(vals, 95)),
        "p98": float(np.nanpercentile(vals, 98)),
        "max": float(np.nanmax(vals)),
        "std": float(np.nanstd(vals)),
        "valid_pixels": int(vals.size),
        "crs": crs,
        "res_x": float(res_x),
        "res_y": float(res_y),
        "width": int(width),
        "height": int(height),
        "bounds": tuple(float(v) for v in bounds),
    }


def _read_window(
    path: Path,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, rasterio.Affine, float | None, str]:
    with rasterio.open(path) as src:
        if bbox is None:
            data = src.read(1)
            transform = src.transform
        else:
            window = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))
            data = src.read(1, window=window)
            transform = src.window_transform(window)
        nodata = src.nodata
        crs = src.crs.to_string() if src.crs else ""
    return _valid(data, nodata), transform, nodata, crs


def _extent(transform: rasterio.Affine, data: np.ndarray) -> tuple[float, float, float, float]:
    rows, cols = data.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (cols, rows)
    return (left, right, bottom, top)


def _draw_scale_bar(ax, length_m: float, label: str) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]
    x0 = xlim[0] + span_x * 0.05
    y0 = ylim[0] + span_y * 0.055
    bar_h = span_y * 0.006
    ax.add_patch(plt.Rectangle((x0, y0), length_m, bar_h, color="white", ec="black", lw=0.8, zorder=10))
    ax.text(
        x0 + length_m / 2,
        y0 + bar_h * 2.7,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color="white",
        fontweight="600",
        path_effects=[],
        zorder=10,
    )


def _draw_north_arrow(ax) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]
    x = xlim[1] - span_x * 0.07
    y0 = ylim[1] - span_y * 0.16
    y1 = ylim[1] - span_y * 0.06
    ax.add_patch(
        FancyArrowPatch(
            (x, y0),
            (x, y1),
            arrowstyle="-|>",
            mutation_scale=18,
            color="white",
            lw=1.6,
            zorder=10,
        )
    )
    ax.text(x, y1 + span_y * 0.015, "N", color="white", ha="center", va="bottom", fontsize=10, fontweight="700", zorder=10)


def make_lst_render(
    *,
    lst_tif: str | Path,
    output_path: str | Path,
    title: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    data, transform, _, _ = _read_window(Path(lst_tif), bbox)
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        raise RuntimeError(f"No hay datos válidos para renderizar {lst_tif}")
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=170, facecolor="#111111")
    ax.set_facecolor("#111111")
    im = ax.imshow(data, extent=_extent(transform, data), origin="upper", cmap=LST_CMAP, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=15, color="white", fontweight="700", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.018)
    cbar.set_label("LST media estival (°C)", fontsize=10, color="white")
    cbar.ax.tick_params(labelsize=9, colors="white")
    cbar.outline.set_edgecolor("white")
    cbar.ax.set_facecolor("#111111")
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    if span > 6000:
        _draw_scale_bar(ax, 2000, "2 km")
    elif span > 2500:
        _draw_scale_bar(ax, 1000, "1 km")
    else:
        _draw_scale_bar(ax, 500, "500 m")
    _draw_north_arrow(ax)
    fig.savefig(output_path, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return Path(output_path)


def make_lst_histogram(*, lst_tif: str | Path, output_path: str | Path) -> Path:
    with rasterio.open(lst_tif) as src:
        vals = _valid(src.read(1), src.nodata)
    arr = vals[np.isfinite(vals)]
    p25, med, p75 = np.nanpercentile(arr, [25, 50, 75])
    p95 = np.nanpercentile(arr, 95)

    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=170)
    ax.hist(arr, bins=42, color=BRAND["forest"], alpha=0.86, edgecolor="white", linewidth=0.4)
    for value, label, color in [
        (p25, "P25", BRAND["olive"]),
        (med, "Mediana", BRAND["dark"]),
        (p75, "P75", BRAND["olive"]),
        (p95, "P95", "#a83a32"),
    ]:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.4)
        ax.text(value, ax.get_ylim()[1] * 0.92, label, rotation=90, va="top", ha="right", fontsize=9, color=color)
    ax.set_xlabel("LST media estival (°C)", fontsize=11, color=BRAND["dark"])
    ax.set_ylabel("Píxeles", fontsize=11, color=BRAND["dark"])
    ax.grid(axis="y", color=BRAND["g200"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)


def make_predictor_panel(
    *,
    ndvi_tif: str | Path,
    ndwi_tif: str | Path,
    ndbi_tif: str | Path,
    output_path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    items = [
        ("NDVI", Path(ndvi_tif), NDVI_CMAP, "Vegetación"),
        ("NDWI", Path(ndwi_tif), NDWI_CMAP, "Humedad / agua"),
        ("NDBI", Path(ndbi_tif), NDBI_CMAP, "Superficie construida"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=170)
    for ax, (label, path, cmap, subtitle) in zip(axes, items):
        data, transform, _, _ = _read_window(path, bbox)
        vals = data[np.isfinite(data)]
        vmin = float(np.nanpercentile(vals, 2))
        vmax = float(np.nanpercentile(vals, 98))
        im = ax.imshow(data, extent=_extent(transform, data), origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{label}\n{subtitle}", fontsize=11, color=BRAND["dark"], fontweight="700")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7, colors=BRAND["g700"])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)


def make_scene_timeline(*, scene_pairs_json: str | Path, output_path: str | Path) -> Path:
    records = json.loads(Path(scene_pairs_json).read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("scene_pairs.json esta vacio")
    xs = [date.fromisoformat(r["landsat_acquired_on"]) for r in records]
    rmse = [float(r["residual_rmse_c"]) for r in records if r.get("residual_rmse_c") is not None]
    rmse_dates = [date.fromisoformat(r["landsat_acquired_on"]) for r in records if r.get("residual_rmse_c") is not None]
    counts = Counter(d.year for d in xs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=170)
    years = sorted(counts)
    ax1.bar([str(y) for y in years], [counts[y] for y in years], color=BRAND["forest"], edgecolor="white")
    ax1.set_title("Escenas válidas por año", fontsize=11, color=BRAND["dark"], fontweight="700")
    ax1.set_ylabel("N escenas", fontsize=10, color=BRAND["dark"])
    ax1.grid(axis="y", color=BRAND["g200"], linewidth=0.7)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(rmse_dates, rmse, color="#a83a32", marker="o", markersize=3.5, linewidth=1.2)
    ax2.set_title("RMSE residual por escena utilizada", fontsize=11, color=BRAND["dark"], fontweight="700")
    ax2.set_ylabel("RMSE (°C)", fontsize=10, color=BRAND["dark"])
    ax2.grid(axis="y", color=BRAND["g200"], linewidth=0.7)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate(rotation=35)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)


def make_scene_timeline(*, scene_pairs_json: str | Path, output_path: str | Path) -> Path:
    """Create a calendar-style summary of the Landsat scenes used."""
    records = json.loads(Path(scene_pairs_json).read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("scene_pairs.json está vacío")
    dates = [date.fromisoformat(r["landsat_acquired_on"]) for r in records]
    counts = Counter(d.year for d in dates)
    years = sorted(counts)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 4.6),
        dpi=170,
        gridspec_kw={"width_ratios": [0.8, 2.2]},
    )
    ax1.bar([str(y) for y in years], [counts[y] for y in years], color=BRAND["forest"], edgecolor="white")
    ax1.set_title("Escenas válidas por año", fontsize=11, color=BRAND["dark"], fontweight="700")
    ax1.set_ylabel("N escenas", fontsize=10, color=BRAND["dark"])
    ax1.set_ylim(0, max(counts.values()) + 2)
    ax1.grid(axis="y", color=BRAND["g200"], linewidth=0.7)
    ax1.spines[["top", "right"]].set_visible(False)

    month_labels = ["jun", "jul", "ago"]
    month_to_x = {6: 0, 7: 1, 8: 2}
    year_to_y = {year: idx for idx, year in enumerate(years)}
    ax2.set_title("Calendario de escenas utilizadas", fontsize=11, color=BRAND["dark"], fontweight="700")
    ax2.set_xlim(-0.55, 2.55)
    ax2.set_ylim(-0.55, len(years) - 0.45)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(month_labels, fontsize=10, color=BRAND["dark"], fontweight="600")
    ax2.set_yticks(range(len(years)))
    ax2.set_yticklabels([str(year) for year in years], fontsize=10, color=BRAND["dark"])
    ax2.invert_yaxis()

    for y in range(len(years)):
        for x in range(3):
            ax2.add_patch(
                plt.Rectangle(
                    (x - 0.48, y - 0.38),
                    0.96,
                    0.76,
                    facecolor=BRAND["g100"],
                    edgecolor=BRAND["g200"],
                    lw=0.8,
                    zorder=0,
                )
            )
    for d in dates:
        x = month_to_x[d.month] - 0.36 + (d.day - 1) / 30.0 * 0.72
        y = year_to_y[d.year]
        ax2.scatter(x, y, s=145, color=BRAND["forest"], edgecolor="white", linewidth=1.0, zorder=2)
        ax2.text(x, y, str(d.day), ha="center", va="center", fontsize=7.5, color="white", fontweight="700", zorder=3)

    ax2.text(
        0.5,
        -0.16,
        "Cada punto representa una escena Landsat utilizada; el número indica el día del mes.",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=BRAND["g700"],
    )
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax2.tick_params(axis="both", length=0)

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)
