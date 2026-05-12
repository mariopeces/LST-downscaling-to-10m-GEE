"""Build an INFFE urban heat island methodology + results HTML report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lst_downscaling.report import build_heat_report


DEFAULT_PROJECT_DIR = Path("G:/Unidades compartidas/6. Projects/Projects/26.03 INFFE")
DEFAULT_LOGO = Path("c:/tmp/darwin_logo.txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", required=True, help="Directory containing pipeline_outputs.json")
    parser.add_argument("--municipality", required=True, help="Municipality label shown in the report")
    parser.add_argument("--activity", default="Actividad 3", help="INFFE activity label")
    parser.add_argument("--output-html", default=None, help="Destination HTML path")
    parser.add_argument("--logo", default=str(DEFAULT_LOGO))
    parser.add_argument("--report-date", default=None)
    args = parser.parse_args(argv)

    pdir = Path(args.pipeline_dir).resolve()
    if not (pdir / "pipeline_outputs.json").exists():
        raise FileNotFoundError(f"No existe pipeline_outputs.json en {pdir}")

    if args.output_html:
        output_html = Path(args.output_html)
    else:
        safe_name = (
            args.municipality.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("Á", "A")
            .replace("É", "E")
            .replace("Í", "I")
            .replace("Ó", "O")
            .replace("Ú", "U")
            .replace("ñ", "n")
            .replace("Ñ", "N")
            .replace(" ", "_")
        )
        output_html = DEFAULT_PROJECT_DIR / f"{safe_name}_Mapa_Isla_Calor_Metodologia.html"

    out = build_heat_report(
        pipeline_dir=pdir,
        output_html=output_html,
        figures_dir=pdir / "_report",
        logo_data_path=Path(args.logo) if args.logo and Path(args.logo).exists() else None,
        report_date=args.report_date,
        municipality_label=args.municipality,
        activity_label=args.activity,
    )
    print(f"Report written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
