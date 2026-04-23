from __future__ import annotations

import argparse


def _parse_int_list(raw_value: str, *, argument_name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(chunk.strip()) for chunk in raw_value.split(",") if chunk.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{argument_name} debe ser una lista de enteros separados por comas."
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError(f"{argument_name} no puede estar vacio.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline AOI -> Landsat ST_B10 + Sentinel-2 -> downscaling termico -> "
            "LST media estival agregada"
        )
    )
    parser.add_argument("--aoi", required=True, help="Ruta al AOI vectorial (.gpkg, .shp, .geojson, ...)")
    parser.add_argument("--output-dir", required=True, help="Directorio de salida")
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Ano inicial del periodo. Por defecto: 2021",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Ano final del periodo. Por defecto: 2025",
    )
    parser.add_argument(
        "--months",
        default="6,7,8",
        help="Meses a incluir en la agregacion, separados por comas. Por defecto: 6,7,8",
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=20.0,
        help="Cobertura maxima de nubes a nivel de escena. Por defecto: 20",
    )
    parser.add_argument(
        "--max-pair-gap-days",
        type=int,
        default=5,
        help="Maxima diferencia temporal entre Landsat y Sentinel-2. Por defecto: 5 dias",
    )
    parser.add_argument(
        "--scale-m",
        type=float,
        default=10.0,
        help="Resolucion final de salida en metros. Por defecto: 10",
    )
    parser.add_argument(
        "--ee-project",
        default=None,
        help="Proyecto de Google Cloud para inicializar Earth Engine. Si se omite, se usa el entorno activo.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limita el numero de escenas Landsat procesadas tras el emparejamiento. Util para pruebas.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.end_year < args.start_year:
        parser.error("--end-year debe ser mayor o igual que --start-year")
    if args.max_cloud_cover < 0 or args.max_cloud_cover > 100:
        parser.error("--max-cloud-cover debe estar entre 0 y 100")
    if args.max_pair_gap_days < 0:
        parser.error("--max-pair-gap-days no puede ser negativo")
    if args.scale_m <= 0:
        parser.error("--scale-m debe ser mayor que 0")

    months = _parse_int_list(args.months, argument_name="--months")

    from .pipeline import run_pipeline

    outputs = run_pipeline(
        aoi_path=args.aoi,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        months=months,
        max_cloud_cover=args.max_cloud_cover,
        max_pair_gap_days=args.max_pair_gap_days,
        scale_m=args.scale_m,
        ee_project=args.ee_project,
        max_scenes=args.max_scenes,
    )

    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0

