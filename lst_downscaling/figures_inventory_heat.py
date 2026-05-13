"""Figures for inventory x LST heat reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Patch
from rasterio.windows import Window, from_bounds

from .figures import BRAND, LST_CMAP
from .inventory_heat import HEAT_COLORS, HEAT_LABELS


POINT_CMAP = LinearSegmentedColormap.from_list(
    "tree_heat", [HEAT_COLORS[label] for label in HEAT_LABELS]
)
_TEXT_OUTLINE = [path_effects.withStroke(linewidth=2.5, foreground="black", alpha=0.85)]


def _style_axes(ax) -> None:
    ax.tick_params(colors=BRAND["txt"], labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(BRAND["g300"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=BRAND["g200"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def f1_histogram(*, values: np.ndarray, thresholds: dict[str, float], output_path: str | Path) -> Path:
    vals = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    x_min = min(float(vals.min()), 32.0)
    x_max = max(float(vals.max()), 53.0)
    edges = [x_min, thresholds["t35"], thresholds["t40"], thresholds["t45"], thresholds["t50"], x_max]
    for label, lo, hi in zip(HEAT_LABELS, edges[:-1], edges[1:]):
        ax.axvspan(lo, hi, color=HEAT_COLORS[label], alpha=0.10, zorder=0)
    ax.hist(vals, bins=48, color=BRAND["forest"], edgecolor="white", linewidth=0.6, zorder=2)
    for p, label in [(np.percentile(vals, 25), "P25"), (np.median(vals), "Mediana"), (np.percentile(vals, 75), "P75")]:
        ax.axvline(p, color=BRAND["dark"], linestyle="--", linewidth=1.2)
        ax.text(p, ax.get_ylim()[1] * 0.95, f"{label}\n{p:.1f} °C", ha="center", va="top", fontsize=9, color=BRAND["dark"], fontweight="600")
    ax.set_xlabel("LST media estival extraída en cada árbol (°C)", fontsize=11, color=BRAND["txt"])
    ax.set_ylabel("Número de árboles", fontsize=11, color=BRAND["txt"])
    ax.set_xlim(x_min, x_max)
    _style_axes(ax)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)


def f2_heat_bands(*, df: pd.DataFrame, output_path: str | Path) -> Path:
    counts = df["heat_class"].value_counts().reindex(HEAT_LABELS, fill_value=0)
    total = max(int(counts.sum()), 1)
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    y = np.arange(len(HEAT_LABELS))
    ax.barh(y, counts.values, color=[HEAT_COLORS[l] for l in HEAT_LABELS], edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(HEAT_LABELS, fontsize=11)
    ax.invert_yaxis()
    for i, n in enumerate(counts.values):
        ax.text(n + counts.max() * 0.01, i, f"{int(n):,} · {100*n/total:.1f} %".replace(",", "."), va="center", fontsize=10, color=BRAND["dark"], fontweight="600")
    ax.set_xlabel("Número de árboles", fontsize=11, color=BRAND["txt"])
    ax.set_xlim(0, counts.max() * 1.2)
    _style_axes(ax)
    ax.grid(axis="x", color=BRAND["g200"], linewidth=0.6, alpha=0.8)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)


def f3_group_stack(*, df: pd.DataFrame, column: str, output_path: str | Path, title: str, top_n: int = 15) -> Path:
    valid = df.dropna(subset=[column, "heat_class"]).copy()
    top = valid[column].value_counts().head(top_n).index[::-1]
    table = pd.crosstab(valid.loc[valid[column].isin(top), column], valid["heat_class"])
    table = table.reindex(index=top, columns=HEAT_LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=160)
    y = np.arange(len(table.index))
    left = np.zeros(len(table.index))
    for label in HEAT_LABELS:
        vals = table[label].values
        ax.barh(y, vals, left=left, color=HEAT_COLORS[label], edgecolor="white", linewidth=0.4, label=label)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([str(s).title()[:42] for s in table.index], fontsize=9)
    ax.set_xlabel("Número de árboles", fontsize=11, color=BRAND["txt"])
    ax.set_title(title, fontsize=12, color=BRAND["dark"], fontweight="700")
    _style_axes(ax)
    ax.grid(axis="x", color=BRAND["g200"], linewidth=0.6, alpha=0.8)
    ax.legend(title="Clase térmica", loc="lower right", fontsize=8.5, title_fontsize=9.5, frameon=True, framealpha=1, edgecolor=BRAND["g200"])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)


def f4_group_box(*, df: pd.DataFrame, column: str, output_path: str | Path, title: str, min_count: int = 80, top_n: int = 12) -> Path:
    valid = df.dropna(subset=[column, "lst_c"]).copy()
    counts = valid[column].value_counts()
    groups = counts[counts >= min_count].head(top_n).index
    med = valid[valid[column].isin(groups)].groupby(column)["lst_c"].median().sort_values()
    data = [valid.loc[valid[column] == g, "lst_c"].values for g in med.index]
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=160)
    ax.boxplot(
        data,
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=BRAND["dark"], linewidth=1.5),
        boxprops=dict(facecolor=BRAND["forest"], alpha=0.30, edgecolor=BRAND["forest"]),
        whiskerprops=dict(color=BRAND["g500"]),
        capprops=dict(color=BRAND["g500"]),
    )
    ax.set_yticks(range(1, len(med.index) + 1))
    ax.set_yticklabels([str(s).title()[:42] for s in med.index], fontsize=9)
    ax.set_xlabel("LST media estival (°C)", fontsize=11, color=BRAND["txt"])
    ax.set_title(title, fontsize=12, color=BRAND["dark"], fontweight="700")
    _style_axes(ax)
    ax.grid(axis="x", color=BRAND["g200"], linewidth=0.6, alpha=0.8)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)


def f5_numeric_scatter(*, df: pd.DataFrame, xcol: str, output_path: str | Path, xlabel: str) -> Path:
    valid = df.dropna(subset=[xcol, "lst_c"]).copy()
    if len(valid) > 8000:
        valid = valid.sample(8000, random_state=42)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    ax.scatter(valid[xcol], valid["lst_c"], s=8, color=BRAND["forest"], alpha=0.35, edgecolors="none")
    ax.set_xlabel(xlabel, fontsize=11, color=BRAND["txt"])
    ax.set_ylabel("LST media estival (°C)", fontsize=11, color=BRAND["txt"])
    _style_axes(ax)
    ax.grid(axis="both", color=BRAND["g200"], linewidth=0.6, alpha=0.8)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)


def _read_raster_window(path: Path, bbox: tuple[float, float, float, float] | None):
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
    return data, transform, nodata


def _extent(transform, data):
    rows, cols = data.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (cols, rows)
    return left, right, bottom, top


def _draw_scale_bar_and_north(ax) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]

    target = span_x * 0.18
    candidates = [200, 500, 1000, 2000, 5000]
    length = min(candidates, key=lambda c: abs(c - target))
    label = f"{length // 1000} km" if length >= 1000 else f"{length} m"

    x0 = xlim[0] + span_x * 0.04
    y0 = ylim[0] + span_y * 0.04
    bar_h = span_y * 0.006
    ax.add_patch(plt.Rectangle((x0, y0), length, bar_h, color="white", ec="black", lw=0.8, zorder=10))
    txt = ax.text(x0 + length / 2, y0 + bar_h * 2.5, label,
                  ha="center", va="bottom", fontsize=10, color="white", fontweight="600", zorder=10)
    txt.set_path_effects(_TEXT_OUTLINE)

    cx = xlim[1] - span_x * 0.06
    cy = ylim[1] - span_y * 0.10
    arrow_len = span_y * 0.06
    arrow = FancyArrowPatch(
        (cx, cy - arrow_len / 2), (cx, cy + arrow_len / 2),
        arrowstyle="-|>", mutation_scale=18, color="white", linewidth=2.5, zorder=10,
    )
    arrow.set_path_effects(_TEXT_OUTLINE)
    ax.add_patch(arrow)
    n_txt = ax.text(cx, cy + arrow_len / 2 + span_y * 0.012, "N",
                    ha="center", va="bottom", fontsize=12, color="white", fontweight="700", zorder=10)
    n_txt.set_path_effects(_TEXT_OUTLINE)


def f6_inventory_map(
    *,
    df,
    lst_tif: str | Path,
    output_path: str | Path,
    title: str,
    priority_only: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    points = df.dropna(subset=["lst_c"]).copy()
    if priority_only:
        points = points.sort_values("lst_c", ascending=False).head(min(350, len(points)))
    data, transform, nodata = _read_raster_window(Path(lst_tif), bbox)
    arr = data.astype("float32")
    if nodata is not None:
        arr = np.where(np.isclose(arr, nodata), np.nan, arr)
    vals = arr[np.isfinite(arr)]
    vmin, vmax = np.nanpercentile(vals, [2, 98])
    extent = _extent(transform, arr)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    ax.set_facecolor("black")
    if not priority_only:
        ax.imshow(arr, extent=extent, origin="upper", cmap=LST_CMAP, vmin=vmin, vmax=vmax, alpha=0.62, interpolation="nearest")
        point_colors = [HEAT_COLORS[str(c)] for c in points["heat_class"].astype(str)]
        ax.scatter(points.geometry.x, points.geometry.y, c=point_colors, s=4.5, alpha=0.85, linewidths=0.0, zorder=3)
        handles = [Patch(facecolor=HEAT_COLORS[label], edgecolor="black", linewidth=0.4, label=label) for label in HEAT_LABELS]
        leg = ax.legend(handles=handles, title="Clase térmica del árbol", loc="lower right",
                        fontsize=9, title_fontsize=10, frameon=True, framealpha=0.95,
                        edgecolor=BRAND["g200"])
        leg.get_title().set_color(BRAND["dark"])
    else:
        ax.imshow(arr, extent=extent, origin="upper", cmap=LST_CMAP, vmin=vmin, vmax=vmax, alpha=0.62, interpolation="nearest")
        ax.scatter(points.geometry.x, points.geometry.y, c="#b2182b", s=18, alpha=0.92,
                   edgecolors="white", linewidths=0.4, zorder=3)
        txt = ax.text(0.02, 0.98, f"{len(points):,} árboles prioritarios".replace(",", "."),
                      transform=ax.transAxes, ha="left", va="top",
                      fontsize=14, color="white", fontweight="700")
        txt.set_path_effects(_TEXT_OUTLINE)
        sub = ax.text(0.02, 0.945, "Mayor temperatura superficial y mayor porte inventariado",
                      transform=ax.transAxes, ha="left", va="top",
                      fontsize=10, color="white", fontweight="500")
        sub.set_path_effects(_TEXT_OUTLINE)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color(BRAND["g300"])
        ax.spines[side].set_linewidth(0.8)
    _draw_scale_bar_and_north(ax)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(output_path)
