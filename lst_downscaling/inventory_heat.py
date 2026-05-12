"""Inventory loading, LST sampling and analysis helpers for heat reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


HEAT_LABELS = ["Muy fresco", "Fresco", "Medio", "Caliente", "Muy caliente"]
HEAT_BINS = [-np.inf, 35.0, 40.0, 45.0, 50.0, np.inf]
HEAT_THRESHOLDS = {"t35": 35.0, "t40": 40.0, "t45": 45.0, "t50": 50.0}
HEAT_COLORS = {
    "Muy fresco": "#234c6a",
    "Fresco": "#5f9fb9",
    "Medio": "#c9dfb1",
    "Caliente": "#e68b53",
    "Muy caliente": "#a83a32",
}


@dataclass
class Coverage:
    total: int
    valid: int
    outside: int
    nodata: int

    @property
    def valid_pct(self) -> float:
        return 100.0 * self.valid / max(self.total, 1)


def _clean_text(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text or np.nan


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _looks_like_species_code(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if text in {"", "XX", "SD", "SBA", "TON"}:
        return True
    return bool(re.fullmatch(r"[A-Z]{2,4}", text))


def _looks_like_species_name(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text or _looks_like_species_code(text):
        return False
    lowered = text.lower()
    if lowered in {"tocón", "toc\xf3n", "marra o alcorque vacío", "marra o alcorque vac\xedo"}:
        return False
    return " " in text or "." in text


def _aranjuez_species(df: pd.DataFrame) -> pd.Series:
    if "ESPECIE" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="object")
    return df["ESPECIE"].map(_clean_text)


def load_aranjuez_inventory(urban_path: str | Path, singular_path: str | Path | None = None) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    urban = gpd.read_file(urban_path)
    urban["inventory_source"] = "Arbolado urbano"
    frames.append(urban)
    if singular_path and Path(singular_path).exists():
        singular = gpd.read_file(singular_path)
        singular["inventory_source"] = "Arbolado singular"
        frames.append(singular)
    gdf = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=frames[0].crs)


def load_inventory(path: str | Path, *, municipality: str, singular_path: str | Path | None = None) -> gpd.GeoDataFrame:
    if municipality.lower() == "aranjuez":
        gdf = load_aranjuez_inventory(path, singular_path)
    else:
        gdf = gpd.read_file(path)
        gdf["inventory_source"] = "Inventario"
    if gdf.crs is None:
        raise RuntimeError(f"El inventario no tiene CRS: {path}")
    return gdf


def normalize_inventory(gdf: gpd.GeoDataFrame, *, municipality: str) -> gpd.GeoDataFrame:
    df = gdf.copy()
    lower = municipality.lower()

    if lower == "aranjuez":
        zone_col = _first_existing(df, ["NOMBRE", "Sector", "CLASE"])
        location_col = _first_existing(df, ["CALLE", "NOMBRE"])
        height_col = _first_existing(df, ["Altura"])
        perimeter_col = _first_existing(df, ["Perímetro_Normal"])
        crown_col = _first_existing(df, ["Diámetro_Copa"])
        irrigation_col = _first_existing(df, ["RIEGO", "Riego"])
        health_col = _first_existing(df, ["Estado_Fitosanitario", "Estado"])
        id_col = _first_existing(df, ["ID_ARBOL", "ID_SINGULAR", "OBJECTID", "ID"])
    else:
        species_col = _first_existing(df, ["especie"])
        zone_col = _first_existing(df, ["ubicacion", "ZONA_MIN", "GESTION"])
        location_col = _first_existing(df, ["ubicacion", "GESTION"])
        height_col = _first_existing(df, ["altura"])
        perimeter_col = _first_existing(df, ["perimetro"])
        crown_col = _first_existing(df, ["radio"])
        irrigation_col = _first_existing(df, ["riego"])
        health_col = _first_existing(df, ["edad_relat"])
        id_col = _first_existing(df, ["id_arbolad", "id_posici", "id_antiguo"])

    df["tree_id"] = df[id_col] if id_col else np.arange(len(df)) + 1
    if lower == "aranjuez":
        df["species"] = _aranjuez_species(df)
    else:
        df["species"] = df[species_col].map(_clean_text) if species_col else np.nan
    df["zone"] = df[zone_col].map(_clean_text) if zone_col else np.nan
    df["location"] = df[location_col].map(_clean_text) if location_col else np.nan
    df["height_m"] = _numeric(df[height_col]) if height_col else np.nan
    df["perimeter_cm"] = _numeric(df[perimeter_col]) if perimeter_col else np.nan
    df["crown_m"] = _numeric(df[crown_col]) if crown_col else np.nan
    df["irrigation"] = df[irrigation_col].map(_clean_text) if irrigation_col else np.nan
    df["condition"] = df[health_col].map(_clean_text) if health_col else np.nan

    if lower == "majadahonda" and "edad_relat" in df.columns:
        df["age_class"] = df["edad_relat"].map(_clean_text).astype("string").str.upper().str.replace("RECIEN", "RECIÉN", regex=False)
    else:
        df["age_class"] = np.nan

    if lower == "majadahonda" and "ZONA_MIN" in df.columns:
        df["urban_type"] = df["ZONA_MIN"].map(_clean_text)
    elif lower == "aranjuez" and "CLASE" in df.columns:
        df["urban_type"] = df["CLASE"].map(_clean_text)
    else:
        df["urban_type"] = np.nan

    return gpd.GeoDataFrame(df, geometry="geometry", crs=gdf.crs)


def sample_raster(gdf: gpd.GeoDataFrame, raster_path: str | Path, *, value_name: str = "lst_c") -> tuple[gpd.GeoDataFrame, Coverage]:
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        bounds = src.bounds
        nodata = src.nodata
        points = gdf.to_crs(raster_crs)
        coords = [(geom.x, geom.y) if geom and not geom.is_empty else (np.nan, np.nan) for geom in points.geometry]
        sampled = np.full(len(coords), np.nan, dtype="float32")
        valid_coord = np.array([np.isfinite(x) and np.isfinite(y) for x, y in coords])
        inside = np.array([
            valid_coord[i] and bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top
            for i, (x, y) in enumerate(coords)
        ])
        if inside.any():
            vals = np.fromiter((v[0] for v in src.sample([coords[i] for i in np.where(inside)[0]])), dtype="float32")
            sampled[np.where(inside)[0]] = vals
        if nodata is not None:
            sampled = np.where(np.isclose(sampled, nodata), np.nan, sampled)
        sampled = np.where(np.isfinite(sampled), sampled, np.nan)

    out = points.copy()
    out[value_name] = sampled
    coverage = Coverage(
        total=len(out),
        valid=int(np.isfinite(sampled).sum()),
        outside=int((~inside).sum()),
        nodata=int((inside & ~np.isfinite(sampled)).sum()),
    )
    return out, coverage


def assign_heat_classes(values: pd.Series) -> tuple[pd.Categorical, dict[str, float]]:
    cats = pd.cut(values, bins=HEAT_BINS, labels=HEAT_LABELS, include_lowest=True)
    return pd.Categorical(cats, categories=HEAT_LABELS, ordered=True), HEAT_THRESHOLDS.copy()


def overall_summary(df: pd.DataFrame) -> dict[str, float | int]:
    s = df["lst_c"].dropna()
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "pct_hot": float(df["heat_class"].isin(["Caliente", "Muy caliente"]).mean() * 100),
        "pct_vhot": float((df["heat_class"] == "Muy caliente").mean() * 100),
        "pct_cool": float(df["heat_class"].isin(["Muy fresco", "Fresco"]).mean() * 100),
    }


def grouped_stats(df: pd.DataFrame, by: str, *, min_count: int = 30, top_n: int = 15) -> pd.DataFrame:
    valid = df.dropna(subset=["lst_c", by]).copy()
    if valid.empty:
        return pd.DataFrame()
    grp = valid.groupby(by, dropna=True)["lst_c"]
    out = grp.agg(n="count", mean="mean", median="median", p90=lambda s: s.quantile(0.90)).reset_index()
    hot = valid.assign(_hot=valid["heat_class"].isin(["Caliente", "Muy caliente"])).groupby(by)["_hot"].mean() * 100
    vhot = valid.assign(_vhot=valid["heat_class"].eq("Muy caliente")).groupby(by)["_vhot"].mean() * 100
    out["pct_hot"] = out[by].map(hot)
    out["pct_vhot"] = out[by].map(vhot)
    out = out[out["n"] >= min_count].sort_values(["pct_hot", "mean"], ascending=False)
    return out.head(top_n).reset_index(drop=True)


def priority_trees(df: pd.DataFrame, *, top_n: int = 80) -> pd.DataFrame:
    valid = df.dropna(subset=["lst_c"]).copy()
    valid["priority_score"] = valid["lst_c"]
    if "height_m" in valid.columns:
        h = valid["height_m"].fillna(0)
        valid["priority_score"] = valid["priority_score"] + np.clip(h, 0, 20) * 0.08
    if "perimeter_cm" in valid.columns:
        p = valid["perimeter_cm"].fillna(0)
        valid["priority_score"] = valid["priority_score"] + np.clip(p, 0, 250) * 0.006
    return valid.sort_values("priority_score", ascending=False).head(top_n).reset_index(drop=True)
