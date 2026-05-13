"""HTML report builder for LST urban heat island deliverables."""

from __future__ import annotations

import base64
import json
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import mean, median

from pyproj import Transformer

from .figures import (
    make_lst_histogram,
    make_lst_render,
    make_predictor_panel,
    make_scene_timeline,
    read_raster_stats,
)


_SPANISH_MONTHS = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def _spanish_date(d: date) -> str:
    return f"{d.day} de {_SPANISH_MONTHS[d.month - 1]} de {d.year}"


_STYLE_BLOCK = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--dark:#1b373f;--forest:#426331;--olive:#879753;--lime:#bcbe76;--cream:#fcf5e3;--g100:#f7f7f5;--g200:#e8e8e4;--g300:#d0d0c8;--g500:#8a8a80;--g700:#4a4a44;--txt:#2c2c28}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter','TT Norms',sans-serif;color:var(--txt);background:#fff;line-height:1.7;font-size:15px;-webkit-font-smoothing:antialiased}
.page{max-width:900px;margin:0 auto;padding:0 40px}
.cover{background:linear-gradient(135deg,var(--dark) 0%,#2a4f3a 50%,var(--forest) 100%);color:#fff;padding:80px 60px;position:relative;overflow:hidden;page-break-after:always}
.cover::before{content:'';position:absolute;top:-100px;right:-100px;width:500px;height:500px;border-radius:50%;background:rgba(188,190,118,.08)}
.cover::after{content:'';position:absolute;bottom:-150px;left:-80px;width:400px;height:400px;border-radius:50%;background:rgba(135,151,83,.06)}
.cover-logo{width:220px;margin-bottom:60px;position:relative;z-index:1;-webkit-mask-image:radial-gradient(ellipse 85% 85% at 50% 50%,rgba(0,0,0,1) 50%,rgba(0,0,0,0) 100%);mask-image:radial-gradient(ellipse 85% 85% at 50% 50%,rgba(0,0,0,1) 50%,rgba(0,0,0,0) 100%)}
.cover-label{font-size:12px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:var(--lime);margin-bottom:16px;position:relative;z-index:1}
.cover h1{font-family:'Space Grotesk','Codec Pro',sans-serif;font-size:38px;font-weight:700;line-height:1.2;margin-bottom:12px;position:relative;z-index:1}
.cover-sub{font-size:18px;font-weight:300;color:rgba(255,255,255,.8);margin-bottom:50px;position:relative;z-index:1}
.cover-meta{display:grid;grid-template-columns:1fr 1fr;gap:20px;position:relative;z-index:1;border-top:1px solid rgba(255,255,255,.15);padding-top:30px;margin-top:30px}
.cover-meta-item{font-size:13px}
.cover-meta-item .lbl{color:var(--lime);font-weight:600;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px}
.sh{background:var(--dark);color:#fff;padding:32px 60px;margin:50px -40px 30px;position:relative;page-break-before:always}
.sh::after{content:'';position:absolute;bottom:0;left:60px;width:60px;height:3px;background:var(--lime)}
.sh .sn{font-family:'Space Grotesk',sans-serif;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:var(--lime);margin-bottom:6px}
.sh h2{font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;line-height:1.3}
h3{font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:600;color:var(--dark);margin:32px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--lime);display:inline-block}
h4{font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:600;color:var(--forest);margin:24px 0 10px}
p{margin-bottom:14px}
strong{color:var(--dark);font-weight:600}
.hb{background:var(--cream);border-left:4px solid var(--forest);padding:18px 24px;margin:20px 0;border-radius:0 6px 6px 0;font-size:14px}
.hb.w{border-left-color:var(--olive);background:#fefdf5}
table{width:100%;border-collapse:collapse;margin:18px 0 24px;font-size:14px}
thead th{background:var(--dark);color:#fff;font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:12px;letter-spacing:.5px;text-transform:uppercase;padding:12px 16px;text-align:left}
tbody td{padding:11px 16px;border-bottom:1px solid var(--g200)}
tbody tr:nth-child(even){background:var(--g100)}
ul,ol{margin:10px 0 16px 24px}
li{margin-bottom:6px}
li::marker{color:var(--olive)}
.sg{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}
.sc{background:var(--g100);border-radius:8px;padding:24px;border-top:3px solid var(--forest)}
.sc .cn{font-family:'Space Grotesk',sans-serif;font-size:32px;font-weight:700;color:var(--lime);opacity:.72}
.sc .ct{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:15px;color:var(--dark);margin:6px 0}
.sc .cl{color:var(--olive);font-size:13px;font-weight:600}
.phase{margin:20px 0;padding:20px 24px;background:var(--g100);border-radius:8px;border-left:3px solid var(--olive)}
.phase .pn{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--olive);margin-bottom:6px}
.phase h4{margin-top:0;color:var(--dark)}
.mc{background:var(--g100);border-radius:8px;padding:30px 20px;margin:24px 0;text-align:center;overflow-x:auto}
.fig{margin:28px 0;text-align:center}
.fig img{max-width:100%;height:auto;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.fig figcaption{font-family:'Space Grotesk',sans-serif;font-size:12px;color:var(--g700);margin-top:10px;letter-spacing:.3px}
.fig.full img{width:100%}
.formula{font-family:'JetBrains Mono','Consolas',monospace;background:var(--cream);border-radius:6px;padding:14px 18px;margin:14px 0;font-size:14px;color:var(--dark);overflow-x:auto;border-left:3px solid var(--forest)}
.rl{font-size:12px;color:var(--g700);line-height:1.9}
.footer{margin-top:60px;padding:30px 0;border-top:2px solid var(--g200);text-align:center;color:var(--g500);font-size:12px}
@media print{
  @page{size:A4;margin:18mm 17mm 20mm 17mm}
  @page cover{margin:0}
  body{font-size:12.5px;line-height:1.55;background:#fff}
  .cover{page:cover;page-break-after:always;break-after:page}
  .page{page:content;max-width:none;margin:0;padding:0}
  .sh{page-break-before:always;break-before:page;margin:0 0 22px;padding:22px 28px}
  .hb,.phase,.formula,figure,.fig,table,.sg,.mc{page-break-inside:avoid;break-inside:avoid}
  h1,h2,h3,h4{page-break-after:avoid;break-after:avoid}
  p{orphans:3;widows:3}
  thead{display:table-header-group}
  tr{page-break-inside:avoid;break-inside:avoid}
}
</style>
"""


def _b64_image(path: Path, mime: str = "image/png") -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _logo_data_uri(logo_uri_path: Path | None) -> str:
    if logo_uri_path is None:
        return ""
    text = Path(logo_uri_path).read_text(encoding="utf-8").strip()
    if text.startswith("data:"):
        return text
    return _b64_image(Path(text))


def _fmt(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _scene_summary(scene_pairs: list[dict]) -> dict[str, object]:
    counts = Counter(date.fromisoformat(r["landsat_acquired_on"]).year for r in scene_pairs)
    gaps = [float(r["date_gap_days"]) for r in scene_pairs]
    clouds_l8 = [float(r["landsat_cloud_cover"]) for r in scene_pairs]
    clouds_s2 = [float(r["sentinel_cloud_cover_mean"]) for r in scene_pairs]
    rmse = [float(r["residual_rmse_c"]) for r in scene_pairs if r.get("residual_rmse_c") is not None]
    return {
        "years": counts,
        "gap_mean": mean(gaps) if gaps else 0.0,
        "gap_max": max(gaps) if gaps else 0.0,
        "landsat_cloud_mean": mean(clouds_l8) if clouds_l8 else 0.0,
        "sentinel_cloud_mean": mean(clouds_s2) if clouds_s2 else 0.0,
        "rmse_mean": mean(rmse) if rmse else 0.0,
        "rmse_median": median(rmse) if rmse else 0.0,
        "rmse_max": max(rmse) if rmse else 0.0,
    }


def _years_rows(counts: Counter[int]) -> str:
    return "\n".join(
        f"<tr><td><strong>{year}</strong></td><td>{counts[year]}</td></tr>"
        for year in sorted(counts)
    )


_URBAN_CENTERS_LONLAT = {
    "aranjuez": (-3.6044, 40.0319),
    "majadahonda": (-3.8737, 40.4737),
}


def _urban_bbox(
    *,
    municipality_label: str,
    crs: str,
    fallback_bounds: tuple[float, float, float, float],
    span_m: float = 3600.0,
) -> tuple[float, float, float, float]:
    key = municipality_label.strip().lower()
    if key in _URBAN_CENTERS_LONLAT:
        lon, lat = _URBAN_CENTERS_LONLAT[key]
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        cx, cy = transformer.transform(lon, lat)
    else:
        left, bottom, right, top = fallback_bounds
        cx = (left + right) / 2
        cy = (bottom + top) / 2
    half = span_m / 2
    left, bottom, right, top = fallback_bounds
    return (
        max(left, cx - half),
        max(bottom, cy - half),
        min(right, cx + half),
        min(top, cy + half),
    )


def build_heat_report(
    *,
    pipeline_dir: str | Path,
    output_html: str | Path,
    figures_dir: str | Path | None = None,
    logo_data_path: str | Path | None = None,
    report_date: str | None = None,
    municipality_label: str,
    activity_label: str = "Actividad 3",
) -> Path:
    pdir = Path(pipeline_dir)
    meta = json.loads((pdir / "pipeline_outputs.json").read_text(encoding="utf-8"))
    scene_pairs = json.loads((pdir / "scene_pairs.json").read_text(encoding="utf-8"))
    skipped_pairs = json.loads((pdir / "skipped_scene_pairs.json").read_text(encoding="utf-8"))

    lst_tif = Path(meta["aggregated_lst"])
    landsat_tif = Path(meta["aggregated_landsat_lst"])
    ndvi_tif = Path(meta["aggregated_ndvi"])
    ndwi_tif = Path(meta["aggregated_ndwi"])
    ndbi_tif = Path(meta["aggregated_ndbi"])

    figures = Path(figures_dir) if figures_dir else pdir / "_report"
    figures.mkdir(parents=True, exist_ok=True)
    render_full = figures / "lst_render_full.png"
    render_detail = figures / "lst_render_detail.png"
    hist = figures / "lst_histogram.png"
    predictors = figures / "predictor_panel.png"
    timeline = figures / "scene_timeline.png"

    lst_stats = read_raster_stats(lst_tif)
    urban_bbox = _urban_bbox(
        municipality_label=municipality_label,
        crs=str(meta.get("output_crs", lst_stats["crs"])),
        fallback_bounds=lst_stats["bounds"],
    )
    make_lst_render(lst_tif=lst_tif, output_path=render_full, title=f"LST media estival - {municipality_label}")
    make_lst_render(lst_tif=lst_tif, output_path=render_detail, title=f"Centro urbano - {municipality_label}", bbox=urban_bbox)
    make_lst_histogram(lst_tif=lst_tif, output_path=hist)
    make_predictor_panel(ndvi_tif=ndvi_tif, ndwi_tif=ndwi_tif, ndbi_tif=ndbi_tif, output_path=predictors, bbox=urban_bbox)
    make_scene_timeline(scene_pairs_json=pdir / "scene_pairs.json", output_path=timeline)

    l8_stats = read_raster_stats(landsat_tif)
    ndvi_stats = read_raster_stats(ndvi_tif)
    ndwi_stats = read_raster_stats(ndwi_tif)
    ndbi_stats = read_raster_stats(ndbi_tif)
    scenes = _scene_summary(scene_pairs)

    today = report_date or _spanish_date(date.today())
    logo_uri = _logo_data_uri(Path(logo_data_path)) if logo_data_path and Path(logo_data_path).exists() else ""
    n_scenes = len(scene_pairs)
    n_skipped = len(skipped_pairs)
    scale = float(meta.get("scale_m", 10.0))
    crs = str(meta.get("output_crs", lst_stats["crs"]))
    mode = str(meta.get("mode", "original_like"))
    years_label = "2021-2025"
    months_label = "junio, julio y agosto"

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Mapa de isla de calor urbana - {municipality_label} - Metodología y resultados</title>
{_STYLE_BLOCK}
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>document.addEventListener("DOMContentLoaded",function(){{mermaid.initialize({{startOnLoad:true,theme:"base",themeVariables:{{primaryColor:"#fcf5e3",primaryBorderColor:"#426331",primaryTextColor:"#1b373f",lineColor:"#879753",fontFamily:"Inter,sans-serif"}}}})}});</script>
</head>
<body>

<section class="cover">
  {f'<img class="cover-logo" src="{logo_uri}" alt="Darwin Geospatial" />' if logo_uri else ''}
  <div class="cover-label">Informe técnico</div>
  <h1>Mapa de isla de calor urbana de {municipality_label}</h1>
  <div class="cover-sub">Metodología y resultados</div>
  <div class="cover-meta">
    <div class="cover-meta-item"><div class="lbl">Cliente</div>INFFE Ingeniería para el Medio Ambiente, S.L.</div>
    <div class="cover-meta-item"><div class="lbl">Preparado por</div>Darwin Geospatial</div>
    <div class="cover-meta-item"><div class="lbl">Referencia</div>26.03-INFFE - {activity_label}</div>
    <div class="cover-meta-item"><div class="lbl">Fecha</div>{today}</div>
    <div class="cover-meta-item"><div class="lbl">Versión</div>1.0</div>
    <div class="cover-meta-item"><div class="lbl">Resolución</div>{_fmt(scale, 0)} m / píxel</div>
  </div>
</section>

<div class="page">

<section class="sh"><div class="sn">§ 1</div><h2>Resumen ejecutivo</h2></section>
<p>Este documento recoge la metodología y los resultados de la <strong>capa de temperatura superficial del suelo (LST)</strong> producida por Darwin Geospatial para el término municipal de <strong>{municipality_label}</strong>, dentro del encargo <strong>26.03-INFFE</strong>. El producto final es un ráster GeoTIFF a <strong>{_fmt(scale, 0)} m/píxel</strong> que representa la <strong>LST media estival</strong> de {months_label} para el periodo <strong>{years_label}</strong>.</p>

<div class="sg">
  <div class="sc"><div class="cn">{_fmt(scale, 0)} m</div><div class="ct">Resolución final</div><div class="cl">Downscaling Landsat + Sentinel-2</div></div>
  <div class="sc"><div class="cn">{n_scenes}</div><div class="ct">Escenas utilizadas</div><div class="cl">{n_skipped} escenas descartadas</div></div>
  <div class="sc"><div class="cn">{_fmt(lst_stats['mean'], 1)} °C</div><div class="ct">LST media</div><div class="cl">P95 = {_fmt(lst_stats['p95'], 1)} °C</div></div>
</div>

<div class="hb">
  <strong>Cómo leer el mapa.</strong> La capa representa <strong>temperatura radiativa de superficie</strong>, no temperatura del aire a la altura de una persona. Los valores altos identifican superficies que, de forma recurrente en los veranos 2021-2025, se comportan como focos térmicos: pavimentos, cubiertas, suelos desnudos o áreas con baja humedad/vegetación. Los valores bajos corresponden generalmente a vegetación densa, zonas húmedas, agua o superficies con mayor capacidad de enfriamiento.
</div>

<section class="sh"><div class="sn">§ 2</div><h2>Datos de partida</h2></section>

<div class="phase">
<div class="pn">Fase 1</div><h4>Ámbito de estudio</h4>
<p>El ámbito se define mediante el AOI vectorial local <code>{Path(meta['aoi']).name}</code>. Las salidas se entregan en <strong>{crs}</strong>, con una malla final de <strong>{_fmt(scale, 0)} m</strong> y recorte al límite de trabajo.</p>
</div>

<div class="phase">
<div class="pn">Fase 2</div><h4>Fuente térmica Landsat</h4>
<p>La temperatura base procede de <strong>Landsat 8/9 Collection 2 Level-2</strong>, banda <code>ST_B10</code>, seleccionando escenas <code>L2SP</code> con producto de temperatura superficial disponible. La banda se escala a grados Celsius mediante los factores oficiales de USGS y se conserva también una media Landsat a 30 m como producto auxiliar.</p>
</div>

<div class="phase">
<div class="pn">Fase 3</div><h4>Predictores Sentinel-2</h4>
<p>Para cada fecha Landsat se busca la escena Sentinel-2 SR Harmonized más cercana dentro de una ventana máxima de 5 días. A partir de Sentinel-2 se calculan <strong>NDVI</strong>, <strong>NDWI</strong> y <strong>NDBI</strong> a 10 m, que funcionan como predictores espaciales de vegetación, humedad y superficie construida.</p>
</div>

<div class="mc">
<pre class="mermaid">
flowchart TD
  A[AOI municipal] --> B[Búsqueda Landsat 8/9 L2SP]
  B --> C[ST_B10 escalado a grados Celsius]
  D[Sentinel-2 SR Harmonized] --> E[NDVI + NDWI + NDBI 10 m]
  C --> F[Regresión múltiple por escena utilizada]
  E --> F
  F --> G[LST downscaled 10 m por escena]
  G --> H[Media Jun-Jul-Ago 2021-2025]
  H --> I[GeoTIFF LST media estival 10 m]
</pre>
</div>

<section class="sh"><div class="sn">§ 3</div><h2>Selección de escenas</h2></section>
<p>El procesamiento utiliza escenas de verano entre <strong>2021 y 2025</strong>, con filtro de nubosidad a nivel de escena inferior al 20 % y emparejamiento Landsat/Sentinel-2 por proximidad temporal. En total se han integrado <strong>{n_scenes} escenas utilizadas</strong>; la diferencia temporal media entre Landsat y Sentinel-2 es de <strong>{_fmt(scenes['gap_mean'], 1)} días</strong> y la máxima de <strong>{_fmt(scenes['gap_max'], 0)} días</strong>.</p>

<table>
<thead><tr><th>Año</th><th>Escenas válidas por año</th></tr></thead>
<tbody>
{_years_rows(scenes['years'])}
</tbody>
</table>

<figure class="fig full"><img src="{_b64_image(timeline)}" alt="Calendario de escenas utilizadas" /><figcaption>Figura 1 - Calendario de escenas utilizadas en la agregación estival 2021-2025.</figcaption></figure>

<section class="sh"><div class="sn">§ 4</div><h2>Modelo de downscaling</h2></section>

<h3>4.1 Relación entre temperatura e índices espectrales</h3>
<p>Para cada escena se ajusta una <strong>regresión lineal múltiple</strong>. Dicho de forma sencilla: el modelo aprende, dentro del municipio, cómo cambia la temperatura superficial cuando cambian tres señales visibles en Sentinel-2: vegetación, humedad y superficie construida.</p>
<div class="formula">Temperatura estimada = valor base + peso vegetación · NDVI + peso construcción · NDBI + peso humedad · NDWI</div>
<p>El <strong>valor base</strong> es la temperatura de partida de esa fecha. Los <strong>pesos</strong> indican cuánto sube o baja la temperatura cuando aumenta cada índice: NDVI suele capturar el efecto refrescante de la vegetación, NDBI la presencia de superficies construidas o impermeables, y NDWI la humedad o presencia de agua. El modelo se ajusta de forma independiente para cada fecha, porque un día de julio y otro de agosto pueden tener condiciones térmicas distintas.</p>

<h3>4.2 Corrección por residuales</h3>
<p>Después de predecir la LST a escala Landsat, se calcula el residual entre la LST observada y la LST modelada. Ese residual es, en la práctica, la parte de la temperatura que el modelo todavía no ha explicado. Se remuestrea, se suaviza con un kernel gaussiano y se suma a la predicción Sentinel-2 a 10 m. Este paso ayuda a conservar patrones térmicos amplios que no dependen solo de NDVI, NDWI o NDBI.</p>
<div class="formula">LST_10m = modelo_Sentinel2_10m + residual_Landsat_suavizado</div>

<h3>4.3 Agregación temporal</h3>
<p>Las imágenes downscaled válidas se agregan mediante media píxel a píxel. El resultado final no describe una fecha concreta: resume el <strong>comportamiento térmico estival medio</strong> del municipio durante cinco veranos consecutivos.</p>

<div class="hb w">
  <strong>Nota metodológica.</strong> La hoja de encargo describía el cálculo de LST desde radiancia, emisividad y corrección atmosférica. En esta implementación se utiliza directamente <strong>ST_B10 de Landsat Collection 2 Level-2</strong>, producto oficial de USGS ya corregido y preparado como temperatura superficial. Es una fuente operacional robusta, pero debe citarse así en la memoria técnica.
</div>

<section class="sh"><div class="sn">§ 5</div><h2>Resultados</h2></section>

<p>La capa final se entrega como GeoTIFF de 32 bits con <em>NoData</em> = -9999. La LST media municipal es de <strong>{_fmt(lst_stats['mean'], 1)} °C</strong>, con mediana de <strong>{_fmt(lst_stats['median'], 1)} °C</strong> y percentil 95 de <strong>{_fmt(lst_stats['p95'], 1)} °C</strong>. La diferencia entre el percentil 95 y el percentil 5 es de <strong>{_fmt(lst_stats['p95'] - lst_stats['p05'], 1)} °C</strong>, una lectura directa del contraste térmico espacial dentro del municipio.</p>

<table>
<thead><tr><th>Producto</th><th>Resolución</th><th>Media</th><th>P5</th><th>P95</th></tr></thead>
<tbody>
<tr><td><strong>LST downscaled</strong></td><td>{_fmt(scale, 0)} m</td><td>{_fmt(lst_stats['mean'], 2)} °C</td><td>{_fmt(lst_stats['p05'], 2)} °C</td><td>{_fmt(lst_stats['p95'], 2)} °C</td></tr>
<tr><td><strong>Landsat LST base</strong></td><td>30 m</td><td>{_fmt(l8_stats['mean'], 2)} °C</td><td>{_fmt(l8_stats['p05'], 2)} °C</td><td>{_fmt(l8_stats['p95'], 2)} °C</td></tr>
<tr><td><strong>NDVI Sentinel-2</strong></td><td>10 m</td><td>{_fmt(ndvi_stats['mean'], 3)}</td><td>{_fmt(ndvi_stats['p05'], 3)}</td><td>{_fmt(ndvi_stats['p95'], 3)}</td></tr>
<tr><td><strong>NDWI Sentinel-2</strong></td><td>10 m</td><td>{_fmt(ndwi_stats['mean'], 3)}</td><td>{_fmt(ndwi_stats['p05'], 3)}</td><td>{_fmt(ndwi_stats['p95'], 3)}</td></tr>
<tr><td><strong>NDBI Sentinel-2</strong></td><td>10 m</td><td>{_fmt(ndbi_stats['mean'], 3)}</td><td>{_fmt(ndbi_stats['p05'], 3)}</td><td>{_fmt(ndbi_stats['p95'], 3)}</td></tr>
</tbody>
</table>

<figure class="fig full"><img src="{_b64_image(render_full)}" alt="Mapa LST completo" /><figcaption>Figura 2 - Vista municipal de la LST media estival. La escala se ajusta a los percentiles 2-98 para mejorar la lectura espacial.</figcaption></figure>
<figure class="fig full"><img src="{_b64_image(render_detail)}" alt="Centro urbano LST" /><figcaption>Figura 3 - Detalle centrado en el casco urbano, pensado para leer la isla de calor en calles, plazas, parques y zonas edificadas.</figcaption></figure>
<figure class="fig full"><img src="{_b64_image(hist)}" alt="Histograma LST" /><figcaption>Figura 4 - Distribución de LST sobre los píxeles válidos del municipio.</figcaption></figure>
<figure class="fig full"><img src="{_b64_image(predictors)}" alt="Panel de predictores Sentinel-2" /><figcaption>Figura 5 - Predictores Sentinel-2 agregados: NDVI, NDWI y NDBI.</figcaption></figure>

<div class="hb">
  <strong>Interpretación rápida.</strong> Los focos de mayor temperatura se interpretan como superficies con menor capacidad de enfriamiento: cubiertas, asfaltos, suelos minerales y zonas con baja vegetación. Las áreas frescas suelen corresponder a parques, masas arboladas, riberas, agua o suelos con mayor humedad. La lectura operativa debe hacerse cruzando esta capa con inventario arbóreo, tipologías de espacio público y usos del suelo.
</div>

<section class="sh"><div class="sn">§ 6</div><h2>Calidad y validación</h2></section>

<p>Como control interno se conserva, para cada escena utilizada, el <strong>error cuadrático medio (RMSE)</strong> del ajuste del modelo. RMSE viene de <em>Root Mean Square Error</em>. En términos prácticos, indica la diferencia típica, en grados Celsius, entre la temperatura Landsat observada y la temperatura que el modelo reconstruye a escala Landsat antes de bajarla a 10 m. Cuanto más bajo es el RMSE, más fiel es el ajuste de esa escena.</p>

<p>En {municipality_label}, el RMSE medio es de <strong>{_fmt(scenes['rmse_mean'], 2)} °C</strong> y la mediana de <strong>{_fmt(scenes['rmse_median'], 2)} °C</strong>. Leído de forma sencilla: el modelo reproduce la estructura térmica de Landsat con un error típico de alrededor de <strong>{_fmt(scenes['rmse_mean'], 1)} °C</strong> por escena. La nubosidad media de las escenas Landsat usadas es de <strong>{_fmt(scenes['landsat_cloud_mean'], 1)} %</strong> y la de Sentinel-2 emparejada de <strong>{_fmt(scenes['sentinel_cloud_mean'], 1)} %</strong>.</p>

<ul>
  <li><strong>Producto superficial.</strong> La capa no representa temperatura del aire, confort térmico humano ni temperatura bajo copa; representa temperatura radiativa de superficie.</li>
  <li><strong>Resolución downscaled.</strong> El detalle a 10 m procede de predictores Sentinel-2; la información térmica original sigue estando controlada por Landsat.</li>
</ul>

<div class="footer">
  Darwin Geospatial · {today} · Documento 26.03-INFFE / {activity_label} / modo {mode}
</div>

</div>
</body>
</html>
"""

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html
