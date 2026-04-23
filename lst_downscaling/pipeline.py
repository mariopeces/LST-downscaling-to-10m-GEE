from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import ee
import geemap
from pyproj import CRS

from .aoi import AOI, read_aoi


LANDSAT_COLLECTIONS = (
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LC09/C02/T1_L2",
)
SENTINEL_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


@dataclass(frozen=True)
class ScenePair:
    landsat_collection: str
    landsat_asset_id: str
    landsat_system_index: str
    landsat_product_id: str
    landsat_spacecraft_id: str
    landsat_acquired_on: str
    landsat_cloud_cover: float
    sentinel_system_indices: tuple[str, ...]
    sentinel_product_ids: tuple[str, ...]
    sentinel_acquired_on: str
    sentinel_image_count: int
    sentinel_cloud_cover_mean: float
    date_gap_days: int


def choose_output_crs(aoi: AOI) -> CRS:
    source = aoi.source_crs
    if source.is_projected:
        axis_info = getattr(source, "axis_info", None) or ()
        if axis_info and all("metre" in (axis.unit_name or "").lower() for axis in axis_info):
            return source
        source_text = source.to_string().upper()
        if source_text.startswith("EPSG:"):
            return source

    zone = int((aoi.centroid_lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if aoi.centroid_lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def initialize_earth_engine(project: str | None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:  # pragma: no cover - depends on local EE auth
        raise RuntimeError(
            "No se ha podido inicializar Earth Engine. Ejecuta `earthengine authenticate` "
            "y, si aplica, pasa --ee-project con tu proyecto de Google Cloud."
        ) from exc


def _ee_geometry_from_aoi(aoi: AOI) -> ee.Geometry:
    geometry = json.loads(aoi.to_feature_collection())["features"][0]["geometry"]
    return ee.Geometry(geometry)


def _mask_landsat(image: ee.Image) -> ee.Image:
    qa_pixel = image.select("QA_PIXEL")
    qa_radsat = image.select("QA_RADSAT")
    clear_mask = (
        qa_pixel.bitwiseAnd(1 << 1).eq(0)
        .And(qa_pixel.bitwiseAnd(1 << 2).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 3).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 4).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 5).eq(0))
        .And(qa_radsat.eq(0))
    )
    return image.updateMask(clear_mask).updateMask(image.select("ST_B10").gt(0))


def _apply_landsat_scale_factors(image: ee.Image) -> ee.Image:
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0).subtract(273.15)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


def _mask_sentinel(image: ee.Image) -> ee.Image:
    qa60 = image.select("QA60")
    cloud_mask = qa60.bitwiseAnd(1 << 10).eq(0).And(qa60.bitwiseAnd(1 << 11).eq(0))
    scl = image.select("SCL")
    scl_mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(cloud_mask.And(scl_mask)).divide(10000)


def _landsat_collection(
    *,
    aoi_geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_cover: float,
) -> ee.ImageCollection:
    def annotate(collection_id: str) -> ee.ImageCollection:
        return (
            ee.ImageCollection(collection_id)
            .filterBounds(aoi_geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_cover))
            .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
            .map(
                lambda image: image.set(
                    {
                        "source_collection": collection_id,
                        "source_asset_id": image.get("system:id"),
                    }
                )
            )
        )

    merged = annotate(LANDSAT_COLLECTIONS[0]).merge(annotate(LANDSAT_COLLECTIONS[1]))
    return merged.sort("system:time_start")


def _sentinel_collection(
    *,
    aoi_geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_cover: float,
) -> ee.ImageCollection:
    return (
        ee.ImageCollection(SENTINEL_COLLECTION)
        .filterBounds(aoi_geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
        .sort("system:time_start")
    )


def _collection_to_records(
    collection: ee.ImageCollection,
    fields: dict[str, str],
) -> list[dict[str, Any]]:
    size = int(collection.size().getInfo())
    if size == 0:
        return []

    values_by_field = {
        key: collection.aggregate_array(source).getInfo()
        for key, source in fields.items()
    }
    acquired_on = (
        ee.List(collection.aggregate_array("system:time_start"))
        .map(lambda value: ee.Date(value).format("YYYY-MM-dd"))
        .getInfo()
    )

    records: list[dict[str, Any]] = []
    for index in range(size):
        record = {key: values[index] for key, values in values_by_field.items()}
        record["acquired_on"] = acquired_on[index]
        records.append(record)
    return records


def _group_sentinel_records(records: list[dict[str, Any]]) -> dict[date, list[dict[str, Any]]]:
    grouped: dict[date, list[dict[str, Any]]] = {}
    for record in records:
        acquired_on = date.fromisoformat(record["acquired_on"])
        grouped.setdefault(acquired_on, []).append(record)
    return grouped


def build_scene_pairs(
    *,
    aoi_geometry: ee.Geometry,
    start_year: int,
    end_year: int,
    months: tuple[int, ...],
    max_cloud_cover: float,
    max_pair_gap_days: int,
    max_scenes: int | None,
) -> list[ScenePair]:
    start_date = f"{start_year:04d}-01-01"
    end_date = f"{end_year + 1:04d}-01-01"

    landsat_records = _collection_to_records(
        _landsat_collection(
            aoi_geometry=aoi_geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
        ),
        fields={
            "source_collection": "source_collection",
            "source_asset_id": "source_asset_id",
            "system_index": "system:index",
            "product_id": "LANDSAT_PRODUCT_ID",
            "spacecraft_id": "SPACECRAFT_ID",
            "cloud_cover": "CLOUD_COVER",
        },
    )
    landsat_records = [
        record
        for record in landsat_records
        if date.fromisoformat(record["acquired_on"]).month in set(months)
    ]

    sentinel_records = _collection_to_records(
        _sentinel_collection(
            aoi_geometry=aoi_geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
        ),
        fields={
            "system_index": "system:index",
            "product_id": "PRODUCT_ID",
            "cloud_cover": "CLOUDY_PIXEL_PERCENTAGE",
        },
    )
    sentinel_records = [
        record
        for record in sentinel_records
        if date.fromisoformat(record["acquired_on"]).month in set(months)
    ]
    sentinel_by_date = _group_sentinel_records(sentinel_records)

    pairs: list[ScenePair] = []
    for landsat_record in landsat_records:
        landsat_date = date.fromisoformat(landsat_record["acquired_on"])
        candidate_days = [
            sentinel_date
            for sentinel_date in sentinel_by_date
            if abs((sentinel_date - landsat_date).days) <= max_pair_gap_days
        ]
        if not candidate_days:
            continue

        def score(candidate: date) -> tuple[int, float, int]:
            scenes = sentinel_by_date[candidate]
            mean_cloud = sum(float(scene["cloud_cover"]) for scene in scenes) / len(scenes)
            return (abs((candidate - landsat_date).days), mean_cloud, -len(scenes))

        best_day = sorted(candidate_days, key=score)[0]
        best_scenes = sentinel_by_date[best_day]

        pairs.append(
            ScenePair(
                landsat_collection=str(landsat_record["source_collection"]),
                landsat_asset_id=str(landsat_record["source_asset_id"]),
                landsat_system_index=str(landsat_record["system_index"]),
                landsat_product_id=str(landsat_record["product_id"]),
                landsat_spacecraft_id=str(landsat_record["spacecraft_id"]),
                landsat_acquired_on=landsat_record["acquired_on"],
                landsat_cloud_cover=float(landsat_record["cloud_cover"]),
                sentinel_system_indices=tuple(str(scene["system_index"]) for scene in best_scenes),
                sentinel_product_ids=tuple(str(scene["product_id"]) for scene in best_scenes),
                sentinel_acquired_on=best_day.isoformat(),
                sentinel_image_count=len(best_scenes),
                sentinel_cloud_cover_mean=sum(float(scene["cloud_cover"]) for scene in best_scenes)
                / len(best_scenes),
                date_gap_days=abs((best_day - landsat_date).days),
            )
        )

    pairs = sorted(pairs, key=lambda pair: (pair.landsat_acquired_on, pair.landsat_product_id))
    if max_scenes is not None:
        pairs = pairs[:max_scenes]
    if not pairs:
        raise RuntimeError(
            "No se han encontrado pares Landsat/Sentinel-2 validos para el AOI y la ventana temporal."
        )
    return pairs


def _get_landsat_image(pair: ScenePair) -> ee.Image:
    return ee.Image(pair.landsat_asset_id)


def _get_sentinel_image(pair: ScenePair, aoi_geometry: ee.Geometry) -> ee.Image:
    collection = (
        ee.ImageCollection(SENTINEL_COLLECTION)
        .filterBounds(aoi_geometry)
        .filter(
            ee.Filter.inList("system:index", ee.List(list(pair.sentinel_system_indices)))
        )
        .map(_mask_sentinel)
    )
    return collection.median().clip(aoi_geometry)


def _s2_indices(s2_image: ee.Image) -> dict[str, ee.Image]:
    nir = s2_image.select("B8")
    red = s2_image.select("B4")
    green = s2_image.select("B3")
    swir = s2_image.select("B11").resample("bilinear")

    return {
        "ndvi": nir.subtract(red).divide(nir.add(red)).rename("S2_NDVI"),
        "ndwi": green.subtract(swir).divide(green.add(swir)).rename("S2_NDWI"),
        "ndbi": swir.subtract(nir).divide(swir.add(nir)).rename("S2_NDBI"),
    }


def _landsat_indices(landsat_image: ee.Image) -> dict[str, ee.Image]:
    return {
        "ndvi": landsat_image.normalizedDifference(["SR_B5", "SR_B4"]).rename("L8_NDVI"),
        "ndwi": landsat_image.normalizedDifference(["SR_B3", "SR_B5"]).rename("L8_NDWI"),
        "ndbi": landsat_image.normalizedDifference(["SR_B6", "SR_B5"]).rename("L8_NDBI"),
    }


def build_downscaled_pair_image(
    *,
    pair: ScenePair,
    aoi_geometry: ee.Geometry,
) -> tuple[ee.Image, ee.Image, dict[str, Any]]:
    landsat_image = _apply_landsat_scale_factors(_mask_landsat(_get_landsat_image(pair))).clip(aoi_geometry)
    sentinel_image = _get_sentinel_image(pair, aoi_geometry)

    landsat_lst = landsat_image.select("ST_B10").rename("L8_LST_30m")
    landsat_indices = _landsat_indices(landsat_image)
    sentinel_indices = _s2_indices(sentinel_image)

    regression_input = (
        ee.Image.constant(1)
        .rename("constant")
        .addBands(
            [
                landsat_indices["ndvi"],
                landsat_indices["ndbi"],
                landsat_indices["ndwi"],
                landsat_lst,
            ]
        )
        .rename(["constant", "ndvi", "ndbi", "ndwi", "lst"])
    )

    regression = regression_input.reduceRegion(
        reducer=ee.Reducer.linearRegression(numX=4, numY=1),
        geometry=aoi_geometry,
        scale=30,
        maxPixels=1_000_000_000_000,
    )
    coefficients = ee.Array(regression.get("coefficients")).getInfo()
    if not coefficients or len(coefficients) != 4:
        raise RuntimeError(
            f"No se han podido calcular coeficientes para la escena {pair.landsat_product_id}."
        )

    intercept = float(coefficients[0][0])
    slope_ndvi = float(coefficients[1][0])
    slope_ndbi = float(coefficients[2][0])
    slope_ndwi = float(coefficients[3][0])

    landsat_model = (
        ee.Image.constant(intercept)
        .add(landsat_indices["ndvi"].multiply(slope_ndvi))
        .add(landsat_indices["ndbi"].multiply(slope_ndbi))
        .add(landsat_indices["ndwi"].multiply(slope_ndwi))
        .rename("L8_LST_MODEL")
    )
    residuals = landsat_lst.subtract(landsat_model).rename("L8_RESIDUALS")
    residuals_smoothed = residuals.resample("bicubic").convolve(
        ee.Kernel.gaussian(radius=1.5, units="pixels")
    )

    downscaled = (
        ee.Image.constant(intercept)
        .add(sentinel_indices["ndvi"].multiply(slope_ndvi))
        .add(sentinel_indices["ndbi"].multiply(slope_ndbi))
        .add(sentinel_indices["ndwi"].multiply(slope_ndwi))
        .add(residuals_smoothed)
        .rename("LST_10m")
        .clip(aoi_geometry)
        .toFloat()
    )

    residual_rmse = (
        residuals.pow(2)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi_geometry,
            scale=30,
            maxPixels=1_000_000_000_000,
        )
        .get("L8_RESIDUALS")
        .getInfo()
    )

    metadata = asdict(pair) | {
        "intercept": intercept,
        "slope_ndvi": slope_ndvi,
        "slope_ndbi": slope_ndbi,
        "slope_ndwi": slope_ndwi,
        "residual_rmse_c": float(residual_rmse) ** 0.5 if residual_rmse is not None else None,
    }
    image = downscaled.set(metadata)
    return image, landsat_lst.toFloat().clip(aoi_geometry).set(metadata), metadata


def _download_image(
    *,
    image: ee.Image,
    output_path: Path,
    aoi_geometry: ee.Geometry,
    crs: CRS,
    scale_m: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geemap.download_ee_image(
        image=image,
        filename=str(output_path),
        region=aoi_geometry,
        crs=crs.to_string(),
        scale=scale_m,
        dtype="float32",
        resampling="bilinear",
        unmask_value=-9999.0,
        overwrite=True,
    )
    return output_path


def run_pipeline(
    *,
    aoi_path: str | Path,
    output_dir: str | Path,
    start_year: int,
    end_year: int,
    months: tuple[int, ...],
    max_cloud_cover: float,
    max_pair_gap_days: int,
    scale_m: float,
    ee_project: str | None,
    max_scenes: int | None,
) -> dict[str, object]:
    aoi = read_aoi(aoi_path)
    initialize_earth_engine(project=ee_project)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    aoi_geometry = _ee_geometry_from_aoi(aoi)
    output_crs = choose_output_crs(aoi)
    scene_pairs = build_scene_pairs(
        aoi_geometry=aoi_geometry,
        start_year=start_year,
        end_year=end_year,
        months=months,
        max_cloud_cover=max_cloud_cover,
        max_pair_gap_days=max_pair_gap_days,
        max_scenes=max_scenes,
    )

    pair_images: list[ee.Image] = []
    landsat_lst_images: list[ee.Image] = []
    pair_metadata: list[dict[str, Any]] = []
    for pair in scene_pairs:
        image, landsat_lst_image, metadata = build_downscaled_pair_image(
            pair=pair,
            aoi_geometry=aoi_geometry,
        )
        pair_images.append(image)
        landsat_lst_images.append(landsat_lst_image)
        pair_metadata.append(metadata)

    if not pair_images or not landsat_lst_images:
        raise RuntimeError("No se ha podido generar ninguna imagen downscaled valida.")

    aggregated = ee.ImageCollection(pair_images).mean().rename("LST_MEAN").clip(aoi_geometry).toFloat()
    aggregated_landsat_lst = (
        ee.ImageCollection(landsat_lst_images).mean().rename("L8_LST_MEAN").clip(aoi_geometry).toFloat()
    )

    months_label = "-".join(f"{month:02d}" for month in months)
    aggregated_path = output_path / f"lst_mean_{months_label}_{start_year}_{end_year}_{int(scale_m)}m.tif"
    aggregated_landsat_path = (
        output_path / f"landsat_lst_mean_{months_label}_{start_year}_{end_year}_30m.tif"
    )
    _download_image(
        image=aggregated,
        output_path=aggregated_path,
        aoi_geometry=aoi_geometry,
        crs=output_crs,
        scale_m=scale_m,
    )
    _download_image(
        image=aggregated_landsat_lst,
        output_path=aggregated_landsat_path,
        aoi_geometry=aoi_geometry,
        crs=output_crs,
        scale_m=30.0,
    )

    scene_pairs_path = output_path / "scene_pairs.json"
    scene_pairs_path.write_text(json.dumps(pair_metadata, indent=2), encoding="utf-8")

    summary = {
        "aoi": str(Path(aoi_path).resolve()),
        "scene_count": len(pair_metadata),
        "scene_pairs": str(scene_pairs_path.resolve()),
        "aggregated_lst": str(aggregated_path.resolve()),
        "aggregated_landsat_lst": str(aggregated_landsat_path.resolve()),
        "output_crs": output_crs.to_string(),
        "scale_m": scale_m,
    }
    (output_path / "pipeline_outputs.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
