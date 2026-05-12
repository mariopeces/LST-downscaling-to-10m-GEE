"""Build an INFFE inventory x urban heat island HTML report."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lst_downscaling.report_inventory_heat import build_heat_inventory_report


DEFAULT_PROJECT_DIR = Path("G:/Unidades compartidas/6. Projects/Projects/26.03 INFFE")
DEFAULT_DATA_DIR = DEFAULT_PROJECT_DIR / "Datos INFFE"
DEFAULT_LOGO = DEFAULT_PROJECT_DIR / "logo_b64.txt"


DEFAULTS = {
    "aranjuez": {
        "inventory": DEFAULT_DATA_DIR / "Aranjuez arbolado/aranjuez_arbolado/aranjuez_arbolado_datos/arbolado_urbano.geojson",
        "singular": DEFAULT_DATA_DIR / "Aranjuez arbolado/aranjuez_arbolado/aranjuez_arbolado_datos/arbolado_singular.geojson",
        "pipeline": Path("outputs/aranjuez"),
        "activity": "Actividad 2",
    },
    "majadahonda": {
        "inventory": DEFAULT_DATA_DIR / "Majadahonda arbolado/ARBOLADO_TODO_ATRA26-30_JUNTO.shp",
        "singular": None,
        "pipeline": Path("outputs/majadahonda"),
        "activity": "Actividad 3",
    },
}


def _safe_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.replace(" ", "_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality", required=True, help="Municipality label, e.g. Aranjuez or Majadahonda")
    parser.add_argument("--inventory", default=None, help="Inventory vector path")
    parser.add_argument("--singular", default=None, help="Optional singular tree inventory path")
    parser.add_argument("--pipeline-dir", default=None, help="Directory containing lst_mean_*.tif")
    parser.add_argument("--lst-tif", default=None, help="LST raster path. Overrides --pipeline-dir discovery")
    parser.add_argument("--activity", default=None, help="INFFE activity label")
    parser.add_argument("--output-html", default=None, help="Destination HTML path")
    parser.add_argument("--figures-dir", default=None, help="Directory for report figures")
    parser.add_argument("--logo", default=str(DEFAULT_LOGO))
    parser.add_argument("--report-date", default=None)
    args = parser.parse_args(argv)

    key = args.municipality.strip().lower()
    defaults = DEFAULTS.get(key, {})

    inventory = Path(args.inventory) if args.inventory else defaults.get("inventory")
    if inventory is None or not Path(inventory).exists():
        raise FileNotFoundError(f"No existe el inventario para {args.municipality}: {inventory}")

    singular = Path(args.singular) if args.singular else defaults.get("singular")
    if singular is not None and not Path(singular).exists():
        singular = None

    pipeline_dir = Path(args.pipeline_dir) if args.pipeline_dir else defaults.get("pipeline")
    if args.lst_tif:
        lst_tif = Path(args.lst_tif)
    else:
        if pipeline_dir is None:
            raise FileNotFoundError("Indica --lst-tif o --pipeline-dir")
        matches = sorted(Path(pipeline_dir).glob("lst_mean_*_10m.tif"))
        if not matches:
            raise FileNotFoundError(f"No se encontró lst_mean_*_10m.tif en {pipeline_dir}")
        lst_tif = matches[0]

    safe = _safe_name(args.municipality)
    output_html = Path(args.output_html) if args.output_html else DEFAULT_PROJECT_DIR / f"{safe}_Inventario_Arbolado_x_Isla_Calor.html"
    figures_dir = Path(args.figures_dir) if args.figures_dir else (Path(pipeline_dir) / "_inventory_report" if pipeline_dir else output_html.with_suffix(""))

    out = build_heat_inventory_report(
        inventory_path=Path(inventory),
        singular_path=singular,
        lst_tif=lst_tif,
        output_html=output_html,
        figures_dir=figures_dir,
        municipality_label=args.municipality,
        logo_data_path=Path(args.logo) if args.logo and Path(args.logo).exists() else None,
        report_date=args.report_date,
        activity_label=args.activity or defaults.get("activity", "Actividad 3"),
    )
    print(f"Report written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
