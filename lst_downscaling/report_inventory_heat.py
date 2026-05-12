"""HTML report builder for inventory x urban heat island cross-analysis."""

from __future__ import annotations

import base64
import html
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .figures_inventory_heat import (
    f1_histogram,
    f2_heat_bands,
    f3_group_stack,
    f4_group_box,
    f5_numeric_scatter,
    f6_inventory_map,
)
from .inventory_heat import (
    HEAT_LABELS,
    assign_heat_classes,
    grouped_stats,
    load_inventory,
    normalize_inventory,
    overall_summary,
    priority_trees,
    sample_raster,
)
from .report import _STYLE_BLOCK, _spanish_date


_EXTRA_STYLE = """<style>
.kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}
.kpi .kc{background:var(--g100);border-radius:8px;padding:22px;border-top:3px solid var(--forest)}
.kpi .num{font-family:'Space Grotesk',sans-serif;font-size:33px;font-weight:700;color:var(--forest);line-height:1}
.kpi .lbl{font-family:'Space Grotesk',sans-serif;font-size:13px;color:var(--dark);margin-top:8px;font-weight:600}
.kpi .sub{font-size:12px;color:var(--g500);margin-top:4px}
.fig{margin:24px 0;text-align:center}
.fig img{max-width:100%;height:auto;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.fig figcaption{font-family:'Space Grotesk',sans-serif;font-size:12px;color:var(--g700);margin-top:10px;letter-spacing:.3px}
.fig.full img{width:100%}
.smallt table{font-size:12.5px}
.smallt thead th{padding:9px 10px;font-size:11px}
.smallt tbody td{padding:7px 10px}
.qa{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0 18px}
.qa .qc{background:var(--g100);border-radius:6px;padding:14px;text-align:center}
.qa .qc .qn{font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:var(--dark);line-height:1}
.qa .qc .ql{font-size:11px;color:var(--g500);margin-top:4px;letter-spacing:.5px;text-transform:uppercase}
.note{font-size:12px;color:var(--g700);margin-top:-8px}
</style>"""


def _b64_image(path: Path, mime: str = "image/png") -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _logo_uri(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("data:"):
        return text
    possible_path = Path(text)
    if possible_path.exists():
        return _b64_image(possible_path)
    return f"data:image/png;base64,{text}"


def _fmt(value: float | int, decimals: int = 1) -> str:
    return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_int(value: float | int) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


def _fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{_fmt(value, decimals)} %"


def _esc(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return html.escape(str(value))


def _title(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    if re.fullmatch(r"[A-Z0-9]{2,5}", text.strip()):
        return text.strip()
    return text.lower().title()


def _row(*cells: object) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _fig(path: Path, caption: str, *, full: bool = False) -> str:
    cls = "fig full" if full else "fig"
    return f"""<figure class="{cls}">
  <img src="{_b64_image(path)}" alt="{_esc(caption)}" />
  <figcaption>{_esc(caption)}</figcaption>
</figure>"""


def _coverage_html(coverage) -> str:
    return f"""
<div class="qa">
  <div class="qc"><div class="qn">{_fmt_int(coverage.total)}</div><div class="ql">Inventario total</div></div>
  <div class="qc"><div class="qn">{_fmt_int(coverage.valid)}</div><div class="ql">Cruces válidos</div></div>
  <div class="qc"><div class="qn">{_fmt_pct(coverage.valid_pct)}</div><div class="ql">Cobertura</div></div>
  <div class="qc"><div class="qn">{_fmt_int(coverage.outside + coverage.nodata)}</div><div class="ql">Sin valor LST</div></div>
</div>
"""


def _summary_table_html(summary: dict) -> str:
    return f"""
<table>
<thead><tr><th>Indicador</th><th>Valor</th></tr></thead>
<tbody>
{_row("Árboles con temperatura extraída", _fmt_int(summary["n"]))}
{_row("Temperatura superficial media", f"{_fmt(summary['mean'])} °C")}
{_row("Mediana", f"{_fmt(summary['median'])} °C")}
{_row("Percentil 25", f"{_fmt(summary['p25'])} °C")}
{_row("Percentil 75", f"{_fmt(summary['p75'])} °C")}
{_row("Percentil 90", f"{_fmt(summary['p90'])} °C")}
{_row("Percentil 95", f"{_fmt(summary['p95'])} °C")}
{_row("% en clase Caliente o Muy caliente", _fmt_pct(summary["pct_hot"]))}
{_row("% en clase Muy caliente", _fmt_pct(summary["pct_vhot"]))}
</tbody></table>
"""


def _group_table_html(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return "<p class=\"note\">No hay suficientes registros con esta variable para generar una tabla robusta.</p>"
    rows = []
    col = df.columns[0]
    for _, r in df.iterrows():
        rows.append(_row(
            f"<strong>{_esc(_title(r[col]))}</strong>",
            _fmt_int(r["n"]),
            f"{_fmt(r['mean'])} °C",
            f"{_fmt(r['median'])} °C",
            f"{_fmt(r['p90'])} °C",
            _fmt_pct(r["pct_hot"]),
            _fmt_pct(r["pct_vhot"]),
        ))
    return f"""
<table class="smallt">
<thead><tr>
<th>{_esc(label)}</th><th>Ejemplares</th><th>Media</th><th>Mediana</th><th>P90</th><th>% Caliente+</th><th>% Muy caliente</th>
</tr></thead>
<tbody>{chr(10).join(rows)}</tbody>
</table>
"""


def _priority_table_html(df: pd.DataFrame, *, top_n: int = 50) -> str:
    rows = []
    for _, r in df.head(top_n).iterrows():
        height = "—" if pd.isna(r.get("height_m")) else f"{_fmt(r['height_m'])} m"
        perim = "—" if pd.isna(r.get("perimeter_cm")) else f"{_fmt(r['perimeter_cm'], 0)} cm"
        rows.append(_row(
            _esc(r.get("tree_id")),
            _esc(_title(r.get("species"))),
            _esc(_title(r.get("zone"))),
            _esc(_title(r.get("location"))),
            f"<strong>{_fmt(r['lst_c'])} °C</strong>",
            _esc(r.get("heat_class")),
            height,
            perim,
        ))
    return f"""
<table class="smallt">
<thead><tr>
<th>ID</th><th>Especie</th><th>Zona</th><th>Localización</th><th>LST</th><th>Clase</th><th>Altura</th><th>Perímetro</th>
</tr></thead>
<tbody>{chr(10).join(rows)}</tbody>
</table>
"""


def _hot_group_stats(df: pd.DataFrame, column: str, *, min_count: int = 30) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    valid = df.dropna(subset=[column, "lst_c", "heat_class"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["_hot"] = valid["heat_class"].isin(["Caliente", "Muy caliente"])
    valid["_vhot"] = valid["heat_class"].eq("Muy caliente")
    out = (
        valid.groupby(column)
        .agg(
            n=("lst_c", "count"),
            mean=("lst_c", "mean"),
            n_hot=("_hot", "sum"),
            n_vhot=("_vhot", "sum"),
            pct_hot=("_hot", lambda s: float(s.mean() * 100)),
            pct_vhot=("_vhot", lambda s: float(s.mean() * 100)),
        )
        .reset_index()
    )
    return out[out["n"] >= min_count].copy()


def _top_label(df: pd.DataFrame, column: str) -> str:
    if df.empty:
        return "sin grupo suficiente"
    return _esc(_title(df.iloc[0][column]))


def _conclusions_html(
    *,
    df: pd.DataFrame,
    summary: dict,
    zone_stats: pd.DataFrame,
    species_stats: pd.DataFrame,
) -> str:
    n_total = int(summary["n"])
    n_hot = int(df["heat_class"].isin(["Caliente", "Muy caliente"]).sum())
    n_vhot = int(df["heat_class"].eq("Muy caliente").sum())

    zone_col = zone_stats.columns[0] if not zone_stats.empty else "zone"
    species_col = species_stats.columns[0] if not species_stats.empty else "species"
    zone_by_volume = _hot_group_stats(df, "zone", min_count=60).sort_values(["n_hot", "pct_hot"], ascending=False)
    species_by_volume = _hot_group_stats(df, "species", min_count=30).sort_values(["n_hot", "pct_hot"], ascending=False)
    zone_by_pct = zone_stats.sort_values(["pct_hot", "mean"], ascending=False).head(1) if not zone_stats.empty else zone_stats
    species_by_pct = species_stats.sort_values(["pct_hot", "mean"], ascending=False).head(1) if not species_stats.empty else species_stats

    bullets = [
        (
            "Dimensionar la intervención",
            f"Hay <strong>{_fmt_int(n_hot)}</strong> árboles en clase Caliente o Muy caliente "
            f"(<strong>{_fmt_pct(summary['pct_hot'])}</strong> del inventario analizado). "
            f"De ellos, <strong>{_fmt_int(n_vhot)}</strong> superan los 50 °C de LST, por lo que deberían concentrar la revisión de detalle."
        )
    ]

    if not zone_by_volume.empty:
        r = zone_by_volume.iloc[0]
        bullets.append((
            "Priorizar por volumen de arbolado afectado",
            f"La zona con más árboles por encima de 45 °C es <strong>{_esc(_title(r['zone']))}</strong>, "
            f"con <strong>{_fmt_int(r['n_hot'])}</strong> ejemplares en Caliente o Muy caliente "
            f"(<strong>{_fmt_pct(r['pct_hot'])}</strong> de sus {_fmt_int(r['n'])} árboles analizados)."
        ))

    if not zone_by_pct.empty:
        r = zone_by_pct.iloc[0]
        bullets.append((
            "Priorizar por intensidad relativa",
            f"La zona con mayor proporción térmica es <strong>{_esc(_title(r[zone_col]))}</strong>: "
            f"<strong>{_fmt_pct(r['pct_hot'])}</strong> de su arbolado queda por encima de 45 °C, "
            f"con una media de <strong>{_fmt(r['mean'])} °C</strong>."
        ))

    if not species_by_pct.empty:
        r = species_by_pct.iloc[0]
        bullets.append((
            "Detectar grupos pequeños pero muy concentrados",
            f"Entre los grupos con muestra suficiente, <strong>{_esc(_title(r[species_col]))}</strong> presenta "
            f"<strong>{_fmt_pct(r['pct_hot'])}</strong> de ejemplares en Caliente o Muy caliente "
            f"(n = <strong>{_fmt_int(r['n'])}</strong>)."
        ))

    items = "\n".join(
        f"<li><strong>{title}.</strong> {text}</li>"
        for title, text in bullets
    )
    return f"<ul>{items}</ul>"


def _enough(df: pd.DataFrame, column: str, *, min_valid: int = 80, min_groups: int = 2) -> bool:
    if column not in df.columns:
        return False
    valid = df.dropna(subset=[column, "lst_c"])
    return len(valid) >= min_valid and valid[column].nunique() >= min_groups


def _numeric_enough(df: pd.DataFrame, column: str, *, min_valid: int = 200) -> bool:
    return column in df.columns and int(df[column].notna().sum()) >= min_valid


def _municipality_notes(municipality: str) -> str:
    key = municipality.lower()
    if key == "aranjuez":
        return (
            "En Aranjuez se han unido el arbolado urbano y el arbolado singular. "
            "Cuando el inventario distingue ambos orígenes, el informe los mantiene como una variable de lectura adicional."
        )
    if key == "majadahonda":
        return (
            "En Majadahonda el inventario aporta, además de especie y ubicación, variables de estructura como altura, perímetro, riego, edad relativa y zona mínima cuando están informadas."
        )
    return "El informe utiliza las variables disponibles en el inventario municipal y omite las que no alcanzan una muestra suficiente."


def build_heat_inventory_report(
    *,
    inventory_path: str | Path,
    lst_tif: str | Path,
    output_html: str | Path,
    figures_dir: str | Path,
    municipality_label: str,
    singular_path: str | Path | None = None,
    logo_data_path: str | Path | None = None,
    report_date: str | None = None,
    activity_label: str = "Actividad 3",
) -> Path:
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    raw = load_inventory(inventory_path, municipality=municipality_label, singular_path=singular_path)
    inv = normalize_inventory(raw, municipality=municipality_label)
    sampled, coverage = sample_raster(inv, lst_tif, value_name="lst_c")
    sampled = sampled[np.isfinite(sampled["lst_c"])].copy()
    sampled["heat_class"], thresholds = assign_heat_classes(sampled["lst_c"])

    summary = overall_summary(sampled)
    priorities = priority_trees(sampled, top_n=80)

    hist_png = figures / "heat_inventory_histogram.png"
    bands_png = figures / "heat_inventory_bands.png"
    species_stack_png = figures / "heat_inventory_species_stack.png"
    species_box_png = figures / "heat_inventory_species_box.png"
    zone_stack_png = figures / "heat_inventory_zone_stack.png"
    zone_box_png = figures / "heat_inventory_zone_box.png"
    map_png = figures / "heat_inventory_map.png"
    priority_map_png = figures / "heat_inventory_priority_map.png"

    f1_histogram(values=sampled["lst_c"].to_numpy(), thresholds=thresholds, output_path=hist_png)
    f2_heat_bands(df=sampled, output_path=bands_png)
    f6_inventory_map(df=sampled, lst_tif=lst_tif, output_path=map_png, title=f"Inventario sobre isla de calor - {municipality_label}")
    f6_inventory_map(df=priorities, lst_tif=lst_tif, output_path=priority_map_png, title=f"Árboles prioritarios - {municipality_label}", priority_only=True)

    figures_html: dict[str, str] = {}
    species_stats = grouped_stats(sampled, "species", min_count=30, top_n=15)
    if _enough(sampled, "species", min_valid=100, min_groups=3):
        f3_group_stack(df=sampled, column="species", output_path=species_stack_png, title="Clases térmicas en las especies más representadas")
        f4_group_box(df=sampled, column="species", output_path=species_box_png, title="Temperatura superficial por especie", min_count=60, top_n=12)
        figures_html["species_stack"] = _fig(species_stack_png, "Distribución de clases térmicas para las especies con más ejemplares.")
        figures_html["species_box"] = _fig(species_box_png, "Rango térmico de las especies con muestra suficiente.")

    zone_stats = grouped_stats(sampled, "zone", min_count=60, top_n=15)
    if _enough(sampled, "zone", min_valid=100, min_groups=3):
        f3_group_stack(df=sampled, column="zone", output_path=zone_stack_png, title="Clases térmicas por zona o ubicación")
        f4_group_box(df=sampled, column="zone", output_path=zone_box_png, title="Temperatura superficial por zona o ubicación", min_count=80, top_n=12)
        figures_html["zone_stack"] = _fig(zone_stack_png, "Peso de las clases térmicas por zona con más arbolado inventariado.")
        figures_html["zone_box"] = _fig(zone_box_png, "Comparación de la temperatura superficial por zona.")

    extra_sections: list[str] = []
    if _enough(sampled, "urban_type", min_valid=100, min_groups=2):
        urban_stats = grouped_stats(sampled, "urban_type", min_count=30, top_n=12)
        extra_sections.append(f"""
<h3>Tipo urbano o ámbito de gestión</h3>
<p>Esta lectura agrupa el inventario por el campo municipal que mejor describe el tipo de emplazamiento. Sirve para diferenciar si el patrón térmico se concentra más en viario, zonas verdes, alineaciones u otros ámbitos disponibles en el inventario.</p>
{_group_table_html(urban_stats, "Tipo")}
""")
    if _enough(sampled, "age_class", min_valid=100, min_groups=2):
        age_stats = grouped_stats(sampled, "age_class", min_count=30, top_n=12)
        extra_sections.append(f"""
<h3>Edad relativa</h3>
<p>La edad relativa ayuda a interpretar si las posiciones más cálidas se concentran en arbolado joven, reciente o ya desarrollado. La temperatura se toma del entorno inmediato del punto inventariado, por lo que debe leerse como contexto térmico de la posición, no como temperatura del árbol.</p>
{_group_table_html(age_stats, "Edad relativa")}
""")
    if _enough(sampled, "inventory_source", min_valid=100, min_groups=2):
        src_stats = grouped_stats(sampled, "inventory_source", min_count=30, top_n=12)
        extra_sections.append(f"""
<h3>Origen del inventario</h3>
<p>La comparación entre arbolado urbano y arbolado singular permite comprobar si los ejemplares singulares aparecen en entornos térmicos diferentes al conjunto urbano general.</p>
{_group_table_html(src_stats, "Origen")}
""")

    numeric_figs: list[str] = []
    if _numeric_enough(sampled, "height_m"):
        height_png = figures / "heat_inventory_height_scatter.png"
        f5_numeric_scatter(df=sampled, xcol="height_m", output_path=height_png, xlabel="Altura inventariada (m)")
        numeric_figs.append(_fig(height_png, "Relación entre altura inventariada y temperatura superficial del entorno."))
    if _numeric_enough(sampled, "perimeter_cm"):
        perimeter_png = figures / "heat_inventory_perimeter_scatter.png"
        f5_numeric_scatter(df=sampled, xcol="perimeter_cm", output_path=perimeter_png, xlabel="Perímetro inventariado (cm)")
        numeric_figs.append(_fig(perimeter_png, "Relación entre perímetro inventariado y temperatura superficial del entorno."))

    today = report_date or _spanish_date(date.today())
    logo_uri = _logo_uri(Path(logo_data_path)) if logo_data_path else ""

    class_text = (
        "Las cinco clases se calculan con umbrales absolutos de temperatura superficial: "
        f"Muy fresco hasta {_fmt(thresholds['t35'], 0)} °C, Fresco de {_fmt(thresholds['t35'], 0)} a {_fmt(thresholds['t40'], 0)} °C, "
        f"Medio de {_fmt(thresholds['t40'], 0)} a {_fmt(thresholds['t45'], 0)} °C, Caliente de {_fmt(thresholds['t45'], 0)} a {_fmt(thresholds['t50'], 0)} °C "
        f"y Muy caliente por encima de {_fmt(thresholds['t50'], 0)} °C. "
        "Así, si una ciudad tiene poco arbolado por encima de 50 °C, la clase Muy caliente no aparece inflada artificialmente."
    )

    html_doc = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Análisis del arbolado urbano frente a la isla de calor - {municipality_label}</title>
{_STYLE_BLOCK}
{_EXTRA_STYLE}
</head>
<body>

<section class="cover">
  {f'<img class="cover-logo" src="{logo_uri}" alt="Darwin Geospatial" />' if logo_uri else ''}
  <div class="cover-label">Informe técnico</div>
  <h1>Análisis del arbolado urbano frente a la isla de calor</h1>
  <div class="cover-sub">{municipality_label} · Cruce del inventario con la capa Darwin LST</div>
  <div class="cover-meta">
    <div class="cover-meta-item"><div class="lbl">Cliente</div>INFFE Ingeniería para el Medio Ambiente, S.L.</div>
    <div class="cover-meta-item"><div class="lbl">Preparado por</div>Darwin Geospatial</div>
    <div class="cover-meta-item"><div class="lbl">Referencia</div>26.03-INFFE - {activity_label}</div>
    <div class="cover-meta-item"><div class="lbl">Fecha</div>{today}</div>
    <div class="cover-meta-item"><div class="lbl">Versión</div>1.0</div>
    <div class="cover-meta-item"><div class="lbl">Resolución térmica de análisis</div>10 m / píxel</div>
  </div>
</section>

<div class="page">

<section class="sh"><div class="sn">§ 1</div><h2>Resumen ejecutivo</h2></section>

<p>Este informe cruza el inventario municipal de arbolado con la capa de temperatura superficial terrestre media estival generada para el análisis de isla de calor. El resultado permite identificar qué ejemplares, especies y zonas aparecen en entornos más cálidos y, por tanto, dónde conviene priorizar revisión, mantenimiento o actuaciones de mejora de sombra y suelo.</p>

<div class="kpi">
  <div class="kc"><div class="num">{_fmt_int(summary['n'])}</div><div class="lbl">Árboles analizados</div><div class="sub">con valor LST válido</div></div>
  <div class="kc"><div class="num">{_fmt(summary['mean'])} °C</div><div class="lbl">LST media</div><div class="sub">en posiciones de arbolado</div></div>
  <div class="kc"><div class="num">{_fmt_pct(summary['pct_hot'])}</div><div class="lbl">Caliente o muy caliente</div><div class="sub">más de 45 °C de LST</div></div>
</div>

<div class="hb">La lectura clave no es si un árbol está “caliente” por sí mismo, sino si su posición inventariada se encuentra dentro de un entorno urbano superficialmente más cálido. Esto ayuda a localizar calles, plazas o alineaciones donde el arbolado puede tener más presión térmica o donde su refuerzo puede aportar más valor.</div>

<section class="sh"><div class="sn">§ 2</div><h2>Cómo se ha hecho el cruce</h2></section>

<h3>Datos utilizados</h3>
<p>Se ha utilizado el inventario de arbolado disponible para {municipality_label} y la capa raster de temperatura superficial terrestre media estival a 10 m. La capa térmica procede del proceso de downscaling Landsat-Sentinel descrito en el informe metodológico de isla de calor.</p>
<p>{_municipality_notes(municipality_label)}</p>

<h3>Extracción de temperatura por árbol</h3>
<p>Cada punto del inventario se reproyecta al sistema de coordenadas del raster y se extrae el valor del píxel de temperatura superficial que cae bajo ese punto. Los registros fuera de cobertura o sobre píxeles sin dato se excluyen de los cálculos estadísticos.</p>
{_coverage_html(coverage)}

<h3>Clases térmicas</h3>
<p>{class_text}</p>

<section class="sh"><div class="sn">§ 3</div><h2>Distribución general</h2></section>

<p>La distribución general muestra cómo se reparte el inventario sobre el gradiente térmico de la ciudad. La mediana representa el comportamiento central y los percentiles altos ayudan a detectar posiciones especialmente expuestas dentro del conjunto municipal.</p>
{_summary_table_html(summary)}
{_fig(hist_png, "Distribución de temperatura superficial extraída en las posiciones del inventario.")}
{_fig(bands_png, "Número de árboles por clase térmica absoluta.")}

<section class="sh"><div class="sn">§ 4</div><h2>Análisis por especie</h2></section>

<p>El análisis por especie separa dos cuestiones: cuántos ejemplares de cada especie aparecen en zonas cálidas y cuál es la temperatura típica del entorno donde se ubican. Las especies con muchos individuos en clase Caliente o Muy caliente no son necesariamente especies problemáticas; pueden estar más presentes en calles o barrios más expuestos.</p>
{figures_html.get("species_stack", "")}
{figures_html.get("species_box", "")}
{_group_table_html(species_stats, "Especie")}

<section class="sh"><div class="sn">§ 5</div><h2>Zonas y localización urbana</h2></section>

<p>La lectura por zona permite pasar del árbol individual a decisiones de gestión urbana. Las zonas con mayor porcentaje de arbolado en clases superiores son candidatas a revisión de continuidad de copa, alcorques, pavimentos, riego y oportunidades de plantación complementaria.</p>
{figures_html.get("zone_stack", "")}
{figures_html.get("zone_box", "")}
{_group_table_html(zone_stats, "Zona")}

{chr(10).join(extra_sections)}

<section class="sh"><div class="sn">§ 6</div><h2>Tamaño y estructura del arbolado</h2></section>

<p>Cuando el inventario incluye medidas como altura o perímetro, se comparan con la temperatura superficial del entorno. Esta relación debe interpretarse con prudencia: un árbol grande puede estar en una calle cálida y un árbol joven puede estar en un parque fresco. El valor está en detectar patrones operativos, no en atribuir causalidad directa al ejemplar.</p>
{''.join(numeric_figs) if numeric_figs else '<p class="note">El inventario no contiene suficientes medidas numéricas comparables para generar esta sección con robustez.</p>'}

<section class="sh"><div class="sn">§ 7</div><h2>Árboles prioritarios</h2></section>

<p>El mapa muestra una preselección espacial de árboles ubicados en los entornos más cálidos, ponderando ligeramente altura y perímetro cuando esos campos están disponibles. Se mantiene solo como lectura cartográfica para localizar focos de revisión, sin incorporar una tabla individual de ejemplares.</p>
{_fig(priority_map_png, "Árboles prioritarios sobre la capa LST con la misma simbología térmica del resto de mapas.", full=True)}

<section class="sh"><div class="sn">§ 8</div><h2>Mapa general</h2></section>

<p>El mapa general combina la capa térmica con el inventario clasificado. El fondo oscuro se mantiene para mejorar la lectura de textos y puntos sobre la rampa de temperatura.</p>
{_fig(map_png, "Inventario completo clasificado por clase térmica absoluta.", full=True)}

<section class="sh"><div class="sn">§ 9</div><h2>Conclusiones operativas</h2></section>

{_conclusions_html(df=sampled, summary=summary, zone_stats=zone_stats, species_stats=species_stats)}

<div class="hb w"><strong>Lectura final.</strong> La capa representa temperatura radiativa de superficie, no temperatura del aire ni confort térmico humano. Su utilidad principal es comparar patrones urbanos y orientar inspecciones o actuaciones allí donde el inventario coincide con superficies persistentemente más cálidas.</div>

<div class="footer">Darwin Geospatial · Informe generado para INFFE · {municipality_label}</div>
</div>
</body>
</html>
"""

    output_html.write_text(html_doc, encoding="utf-8")
    return output_html
