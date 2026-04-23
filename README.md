# LST Downscaling to 10 m in Google Earth Engine

This repository now contains two things:

1. The original JavaScript Google Earth Engine app in `javascript_codes/LST-downscaling_GEE_APP.js` as methodological reference.
2. A new AOI-driven Python pipeline that follows the same operational pattern as `wind_calculator`:
   - AOI stored locally in `data/aoi`
   - execution by CLI
   - deterministic outputs in `outputs/<municipality>`

The new pipeline is oriented to municipal urban heat island mapping using:

- Landsat 8/9 Collection 2 Level-2 `ST_B10` as the base LST source
- Sentinel-2 SR Harmonized for high-resolution predictors
- linear regression plus residual smoothing, following the logic already used in the original GEE script
- temporal aggregation of all valid summer pairs into a final mean LST raster

## What the pipeline produces

For an AOI and a summer multi-year window such as `2021-2025`, the pipeline:

1. Reads the AOI from `data/aoi/<name>.gpkg`
2. Searches Landsat 8/9 `L2SP` scenes with valid `ST_B10`
3. Searches Sentinel-2 scenes close in time
4. Pairs Landsat and Sentinel-2 scenes automatically
5. Builds a downscaled LST image for each valid pair
6. Aggregates all pair outputs into one final raster
7. Downloads the final GeoTIFF locally to `outputs/<name>`

Main outputs:

- `lst_mean_06-07-08_2021_2025_10m.tif`
- `landsat_lst_mean_06-07-08_2021_2025_30m.tif`
- `scene_pairs.json`
- `pipeline_outputs.json`

## Important methodological note

This workflow uses `ST_B10` directly from Landsat Collection 2 Level-2.

That means:

- the Landsat base LST is not recomputed from raw thermal radiance in this repo
- the final `10 m` output is a downscaled product, not a native thermal measurement at `10 m`
- the final product is suitable for urban heat island mapping as an operational surface temperature layer, as long as this is stated clearly in the methodology

## Repository structure

```text
config/
data/
  aoi/
javascript_codes/
lst_downscaling/
outputs/
tests/
```

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Earth Engine authentication

Before running the pipeline, authenticate Earth Engine:

```bash
earthengine authenticate
```

If your account requires an explicit Google Cloud project, pass it at runtime with `--ee-project`.

## Usage

Put the AOI in `data/aoi`, for example:

```text
data/aoi/aranjuez.gpkg
```

Run:

```bash
python -m lst_downscaling ^
  --aoi data/aoi/aranjuez.gpkg ^
  --output-dir outputs/aranjuez ^
  --start-year 2021 ^
  --end-year 2025 ^
  --months 6,7,8 ^
  --max-cloud-cover 20 ^
  --max-pair-gap-days 5 ^
  --scale-m 10
```

On PowerShell you can also run it in one line:

```powershell
python -m lst_downscaling --aoi data/aoi/aranjuez.gpkg --output-dir outputs/aranjuez --start-year 2021 --end-year 2025 --months 6,7,8 --max-cloud-cover 20 --max-pair-gap-days 5 --scale-m 10
```

Useful options:

- `--ee-project <project-id>` to initialize Earth Engine with an explicit project
- `--max-scenes 3` to do a short test run

## Notes on scene selection

The pipeline currently:

- filters Landsat by `CLOUD_COVER < threshold`
- requires `PROCESSING_LEVEL = L2SP`, so `ST_B10` is actually available
- filters Sentinel-2 by `CLOUDY_PIXEL_PERCENTAGE < threshold`
- pairs each Landsat scene with the nearest Sentinel-2 acquisition date within the allowed day gap
- mosaics all Sentinel-2 granules found for the selected day over the AOI

## Notes on the downscaling model

For each valid pair:

- Landsat predictors:
  - `NDVI = (SR_B5 - SR_B4) / (SR_B5 + SR_B4)`
  - `NDWI = (SR_B3 - SR_B5) / (SR_B3 + SR_B5)`
  - `NDBI = (SR_B6 - SR_B5) / (SR_B6 + SR_B5)`
- Sentinel-2 predictors:
  - `NDVI = (B8 - B4) / (B8 + B4)`
  - `NDWI = (B3 - B11) / (B3 + B11)`
  - `NDBI = (B11 - B8) / (B11 + B8)`
- Landsat `ST_B10` is scaled to degrees Celsius
- a multiple linear regression is fitted at Landsat scale
- the regression is applied at Sentinel scale
- smoothed residuals are added back to the final image

## Tests

```bash
python -m pytest -q
```

## Current limitations

- I have not run a full real AOI export from this environment because Earth Engine is not authenticated here
- the pipeline uses scene-level cloud filters plus pixel masks, but it does not yet score pairs by more advanced coverage metrics over the AOI
- the final raster is generated from the valid paired scenes found by Earth Engine at runtime, so results depend on the available archive and masks

## Original JavaScript app

The original interactive script is still available in:

- `javascript_codes/LST-downscaling_GEE_APP.js`

That file remains useful as:

- methodological reference
- quick visual prototyping in the GEE Code Editor
- comparison point against the new AOI pipeline
