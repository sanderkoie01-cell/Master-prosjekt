#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import shapefile
import xarray as xr
from shapely import contains_xy
from shapely.geometry import shape as shapely_shape
from tqdm import tqdm

# Run:
# python scripts/offshore_wind_sites_timeseries_2024.py

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import functions


MONTHS = [
    ("01", "jan"), ("02", "feb"), ("03", "mar"), ("04", "apr"),
    ("05", "may"), ("06", "jun"), ("07", "jul"), ("08", "aug"),
    ("09", "sep"), ("10", "oct"), ("11", "nov"), ("12", "dec"),
]


shape_folder = Path("/Data/gfi/vindenergi/nab015/Wind_data/havvind_identifiserteomrader_2023f") #path to shape files

# -------- Script configuration --------
SHAPEFILE = shape_folder / "havvind_identifiserteomrader_2023f.shp"
ALL_SITES = True
SITE_NAMES = ""
REFERENCE_CAPACITY_MW = 1000.0
CAPACITY_DENSITY_MW_KM2 = 5.0
YEAR = 2024
WEATHER_ROOT = Path("/Data/gfi/vindenergi/nab015/CERRA_multi_level")
OUTPUT = shape_folder / "havvind_identifiserteomrader_output/offshore_all_sites_wind_timeseries_2024_1000MW.nc"


def _read_site_metadata(shp_path: Path) -> list[dict]:
    sf = shapefile.Reader(str(shp_path), encoding="latin1")
    fields = [f[0] for f in sf.fields[1:]]
    records: list[dict] = []
    for sr in sf.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        geom = shapely_shape(sr.shape.__geo_interface__)
        records.append(
            {
                "name": str(rec.get("OMRADENAVN", "")).strip(),
                "type": str(rec.get("ANLEGGTYPE", "")).strip(),
                "area_km2": float(rec.get("OMRADEAREA", 0.0) or 0.0),
                "geometry": geom,
            }
        )
    return records


def _select_sites(site_records: list[dict], site_names_arg: str, all_sites: bool) -> list[dict]:
    if all_sites:
        return site_records

    if site_names_arg.strip():
        wanted = [s.strip() for s in site_names_arg.split(",") if s.strip()]
        by_name = {s["name"]: s for s in site_records}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise ValueError(f"Site names not found in shapefile: {missing}")
        selected = [by_name[name] for name in wanted]
    else:
        exact_bunnfast = [s for s in site_records if s["type"].strip().lower() == "bunnfast"]
        if len(exact_bunnfast) >= 3:
            selected = exact_bunnfast[:3]
        else:
            bunnfast_any = [s for s in site_records if "bunnfast" in s["type"].lower()]
            if len(bunnfast_any) >= 3:
                selected = bunnfast_any[:3]
            elif len(site_records) >= 3:
                selected = site_records[:3]
            else:
                raise ValueError("Shapefile does not contain at least 3 areas.")
    return selected


def _site_grid_cells(ds: xr.Dataset, sites: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    lat2d = ds["latitude"].values
    lon2d = ds["longitude"].values

    site_cells: dict[str, dict[str, np.ndarray]] = {}
    for site in sites:
        geom = site["geometry"]
        minx, miny, maxx, maxy = geom.bounds

        bbox_mask = (
            (lat2d >= miny)
            & (lat2d <= maxy)
            & (lon2d >= minx)
            & (lon2d <= maxx)
        )
        y_all, x_all = np.where(bbox_mask)
        if len(y_all) == 0:
            continue

        lon_candidates = lon2d[y_all, x_all]
        lat_candidates = lat2d[y_all, x_all]
        inside = contains_xy(geom, lon_candidates, lat_candidates)

        y_sel = y_all[inside]
        x_sel = x_all[inside]
        if len(y_sel) == 0:
            continue

        site_cells[site["name"]] = {
            "y_idx": y_sel.astype(int),
            "x_idx": x_sel.astype(int),
            "lat": lat2d[y_sel, x_sel].astype(float),
            "lon": lon2d[y_sel, x_sel].astype(float),
        }

    return site_cells


def estimate_site_timeseries(
    ds: xr.Dataset,
    site: dict,
    cell_info: dict[str, np.ndarray],
    startyear: int,
    prod_year: int,
    capacity_density_mw_km2: float,
    reference_capacity_mw: float | None,
) -> np.ndarray:
    n_cells = len(cell_info["y_idx"])
    if n_cells == 0:
        return np.zeros(ds.sizes["valid_time"], dtype=np.float64)

    if reference_capacity_mw is not None:
        total_capacity_mw = float(reference_capacity_mw)
    else:
        total_capacity_mw = float(site["area_km2"]) * float(capacity_density_mw_km2)

    cell_capacity_mw = total_capacity_mw / n_cells if n_cells > 0 else 0.0

    site_ts = np.zeros(ds.sizes["valid_time"], dtype=np.float64)
    for idx in range(n_cells):
        ts_mw = functions.estimate_wind_power(
            country="Norway",
            lat=float(cell_info["lat"][idx]),
            lon=float(cell_info["lon"][idx]),
            capacity=cell_capacity_mw,
            startyear=startyear,
            prod_year=prod_year,
            status="operating",
            installation_type="offshore",
            xrds=ds,
            y_idx=int(cell_info["y_idx"][idx]),
            x_idx=int(cell_info["x_idx"][idx]),
            wts_smoothing=False,
            power_smoothing=False,
            spatial_interpolation=True,
            wake_loss_factor=1.0,
            single_turb_curve=False,
            enforce_start_year=False,
            verbose=False,
        )
        if ts_mw is not None:
            site_ts += np.asarray(ts_mw, dtype=np.float64)

    return site_ts


def _write_netcdf_atomic(ds_out: xr.Dataset, output: Path) -> None:
    tmp_output = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        ds_out.to_netcdf(tmp_output, engine="h5netcdf")
        tmp_output.replace(output)
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise

def main() -> None:
    sites_raw = _read_site_metadata(SHAPEFILE)
    selected_sites = _select_sites(sites_raw, SITE_NAMES, ALL_SITES)
    output = Path(OUTPUT)

    month0 = MONTHS[0][1]
    ref_file = WEATHER_ROOT / str(YEAR) / f"cerra_{YEAR}_multi_level_{month0}.nc"
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference weather file not found: {ref_file}")

    with xr.open_dataset(ref_file, engine="h5netcdf") as ds_ref:
        site_cells = _site_grid_cells(ds_ref, selected_sites)

    selected_sites = [s for s in selected_sites if s["name"] in site_cells]
    if len(selected_sites) == 0:
        raise RuntimeError("No weather grid cells found inside selected polygons.")

    monthly_dfs: list[pd.DataFrame] = []
    for month_num, month_name in MONTHS:
        weather_file = WEATHER_ROOT / str(YEAR) / f"cerra_{YEAR}_multi_level_{month_name}.nc"
        if not weather_file.exists():
            print(f"Skipping missing weather file: {weather_file}")
            continue

        with xr.open_dataset(weather_file, engine="h5netcdf") as ds:
            ds_wind = ds[["ws", "latitude", "longitude"]].load()
            shifted_time = pd.to_datetime(ds_wind["valid_time"].values) - pd.Timedelta(hours=1)

            month_df = pd.DataFrame(index=shifted_time)
            for site in tqdm(selected_sites, desc=f"{YEAR}-{month_num}", unit="site"):
                site_name = site["name"]
                ts_mw = estimate_site_timeseries(
                    ds=ds_wind,
                    site=site,
                    cell_info=site_cells[site_name],
                    startyear=2024,
                    prod_year=YEAR,
                    capacity_density_mw_km2=CAPACITY_DENSITY_MW_KM2,
                    reference_capacity_mw=REFERENCE_CAPACITY_MW,
                )
                month_df[site_name] = ts_mw

        monthly_dfs.append(month_df)

    if not monthly_dfs:
        raise RuntimeError("No monthly weather files were processed.")

    result_df = pd.concat(monthly_dfs, axis=0).sort_index()
    result_df = result_df[~result_df.index.duplicated(keep="first")]
    result_df["total_mw"] = result_df.sum(axis=1)

    site_names = [s["name"] for s in selected_sites]

    ds_out = xr.Dataset(
        data_vars={
            "wind_power_mw": (("time", "site"), result_df[site_names].to_numpy(dtype=np.float64)),
            "total_wind_power_mw": (("time",), result_df["total_mw"].to_numpy(dtype=np.float64)),
        },
        coords={
            "time": result_df.index.to_numpy(dtype="datetime64[ns]"),
            "site": site_names,
        },
        attrs={
            "year": int(YEAR),
            "startyear_used_in_estimate_wind_power": 2024,
            "all_sites": int(bool(ALL_SITES)),
            "reference_capacity_mw_per_site": (
                float(REFERENCE_CAPACITY_MW) if REFERENCE_CAPACITY_MW is not None else np.nan
            ),
            "capacity_density_mw_per_km2": float(CAPACITY_DENSITY_MW_KM2),
            "shapefile": str(SHAPEFILE),
            "note": "Potential offshore wind generation estimated from polygon-covered CERRA grid cells.",
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_netcdf_atomic(ds_out, output)

    print(f"Saved timeseries to: {output}")
    print("Sites included:")
    for s in selected_sites:
        print(f"  - {s['name']} ({s['area_km2']} km^2)")


if __name__ == "__main__":
    main()