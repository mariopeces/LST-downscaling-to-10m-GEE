from pyproj import CRS

from lst_downscaling.aoi import AOI
from lst_downscaling.pipeline import ScenePair, choose_output_crs


def test_choose_output_crs_keeps_projected_metric_source():
    aoi = AOI(
        path=None,  # type: ignore[arg-type]
        source_crs=CRS.from_epsg(25830),
        geometry=None,  # type: ignore[arg-type]
        geometry_4326=None,  # type: ignore[arg-type]
        centroid_lon=-3.7,
        centroid_lat=40.4,
    )

    assert choose_output_crs(aoi).to_epsg() == 25830


def test_scene_pair_serializable_fields():
    pair = ScenePair(
        landsat_collection="LANDSAT/LC08/C02/T1_L2",
        landsat_asset_id="LANDSAT/LC08/C02/T1_L2/LC08_TEST",
        landsat_system_index="LC08_TEST",
        landsat_product_id="LC08_PRODUCT",
        landsat_spacecraft_id="LANDSAT_8",
        landsat_acquired_on="2021-06-18",
        landsat_cloud_cover=5.0,
        sentinel_system_indices=("S2A_TEST_1", "S2A_TEST_2"),
        sentinel_product_ids=("P1", "P2"),
        sentinel_acquired_on="2021-06-18",
        sentinel_image_count=2,
        sentinel_cloud_cover_mean=3.2,
        date_gap_days=0,
    )

    assert pair.sentinel_image_count == 2
