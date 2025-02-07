#!/usr/bin/env python

"""
Script to stitch land and sea surface temperatures using sea fraction and ice
fraction fields.

By J. Siddons 2025-02.
For Python >= 3.11
"""

import os
from itertools import product

import netCDF4 as nc
import numpy as np
import polars as pl
import xarray as xr
import yaml

from glomar_gridding.grid import (
    align_to_grid,
    assign_to_grid,
    grid_from_resolution,
)
from glomar_gridding.io import load_array, load_dataset
from glomar_gridding.utils import days_since_by_month

CONFIG_PATH: str = os.path.join(os.path.dirname(__file__), "config_stitch.yaml")
ENSEMBLES: list[int] = list(range(1, 201))


def _align_sea_area_fraction(
    sea_area_fraction: xr.DataArray,
    grid: xr.DataArray,
) -> pl.DataFrame:
    sf_df = pl.from_pandas(sea_area_fraction.to_dataframe().reset_index())
    return (
        align_to_grid(
            sf_df,
            grid,
            grid_coords=["lat", "lon"],
        )
        .group_by("grid_idx")
        .agg(pl.col("sea_fraction").mean())
    )


def _prep_ice_fraction(
    ice_fraction_path: str,
    year_start: int,
    year_end: int,
    grid: xr.DataArray,
) -> pl.DataFrame:
    if_df = pl.from_pandas(
        load_dataset(ice_fraction_path).to_dataframe().reset_index()
    ).filter(
        pl.col("time").dt.year().is_between(year_start, year_end, closed="both")
    )
    grid_df = (
        if_df.select(["latitude", "longitude"])
        .unique()
        .pipe(
            align_to_grid,
            grid,
            obs_coords=["latitude", "longitude"],
            grid_coords=["lat", "lon"],
        )
    )
    if_df = if_df.join(
        grid_df, on=["latitude", "longitude"], how="left", coalesce=True
    )

    return (
        if_df.group_by(["time", "grid_idx"])
        .agg(pl.col("sic").mean())
        .with_columns(
            (
                pl.when(pl.col("sic").lt(0.15) | pl.col("sic").is_null())
                .then(pl.lit(0))
                .otherwise(pl.col("sic"))
                .alias("sic")
            ),
            pl.col("time").dt.round("1h").name.keep(),
        )
    )


def _initialise_ncfile(
    ncfile: nc.Dataset,
    output_lon: np.ndarray,
    output_lat: np.ndarray,
    output_time: np.ndarray,
    current_year: int,
):
    ncfile.createDimension("lat", len(output_lat))  # latitude axis
    ncfile.createDimension("lon", len(output_lon))  # longitude axis
    ncfile.createDimension("time", len(output_time))  # unlimited axis

    # Define two variables with the same names as dimensions,
    # a conventional way to define "coordinate variables".
    lat = ncfile.createVariable("lat", np.float32, ("lat",))
    lat.units = "degrees_north"
    lat.long_name = "latitude"
    lon = ncfile.createVariable("lon", np.float32, ("lon",))
    lon.units = "degrees_east"
    lon.long_name = "longitude"
    time = ncfile.createVariable("time", np.float32, ("time",))
    time.units = f"days since {current_year}-01-15"
    time.long_name = "time"

    # Define a 3D variable to hold the data
    # note: unlimited dimension is leftmost
    temp_anom = ncfile.createVariable(
        "surface_temperature_anomaly_perturbed",
        np.float32,
        ("time", "lat", "lon"),
    )
    temp_anom.standard_name = "perturbed surface temperature anomaly"
    temp_anom.units = "deg C"  # degrees Kelvin

    return (
        lon,
        lat,
        time,
        temp_anom,
    )


def main() -> None:  # noqa: D103
    with open(CONFIG_PATH, "r") as io:
        config: dict = yaml.safe_load(io)

    ice_fraction_path: str = config.get("ice_fraction_path", "")
    sea_area_fraction_path: str = config.get("sea_fraction_path", "")

    sst_path: str = config.get("sst_path", "")
    lsat_path: str = config.get("lsat_path", "")
    out_path: str = config.get("out_path", "")

    year_start: int = config.get("domain", {}).get("startyear", 1850)
    year_end: int = config.get("domain", {}).get("endyear", 2024)

    sea_area_fraction = load_array(sea_area_fraction_path, "sea_fraction")

    lon_west: float = config.get("domain", {}).get("west", -180.0)
    lon_east: float = config.get("domain", {}).get("east", 180.0)
    lat_south: float = config.get("domain", {}).get("south", -90.0)
    lat_north: float = config.get("domain", {}).get("north", 90.0)

    # Define the output grid (for aligning the sea and ice fraction fields)
    output_grid: xr.DataArray = grid_from_resolution(
        resolution=5.0,
        bounds=[
            (lat_south + 2.5, lat_north + 2.5),
            (lon_west + 2.5, lon_east + 2.5),
        ],
        coord_names=["lat", "lon"],
    )

    sea_area_fraction_df: pl.DataFrame = _align_sea_area_fraction(
        sea_area_fraction, output_grid
    )
    del sea_area_fraction
    print("Prepared Sea Fraction")
    print(sea_area_fraction_df)

    ice_fraction_df: pl.DataFrame = _prep_ice_fraction(
        ice_fraction_path, year_start, year_end, output_grid
    )
    print("Prepared Ice Fraction")
    print(ice_fraction_df)

    # Combine sea fraction and ice fraction
    ice_fraction_df = ice_fraction_df.join(
        sea_area_fraction_df,
        on="grid_idx",
        how="left",
        coalesce=True,
    ).select(
        [
            pl.col("grid_idx"),
            pl.col("time"),
            pl.max_horizontal(
                pl.min_horizontal(
                    1 - pl.col("sic") + pl.col("sea_fraction"),
                    pl.lit(1),
                ),
                pl.lit(0),
            )
            .fill_nan(0)
            .fill_null(0)
            .alias("sea_fraction"),
        ]
    )
    print("Prepared Ice Fraction and Sea Fraction for the input period")
    print(ice_fraction_df)

    for year, ensemble in product(range(year_start, year_end + 1), ENSEMBLES):
        print(f"Doing {year = }, {ensemble = }")
        sst_file = sst_path.format(year=year, ensemble=ensemble)
        lsat_file = lsat_path.format(year=year, ensemble=ensemble)
        out_file = out_path.format(year=year, ensemble=ensemble)

        if not os.path.isdir(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))

        sst = load_array(sst_file, "sst_anomaly_perturbed")
        lsat = load_array(lsat_file, "lsat_anomaly_perturbed")

        output_lat = sst.coords["lat"]
        output_lon = sst.coords["lon"]
        output_time = sst.coords["time"]

        ncfile = nc.Dataset(out_file, mode="w", format="NETCDF4_CLASSIC")

        (lon, lat, time, temp_anom) = _initialise_ncfile(
            ncfile,
            output_lon,
            output_lat,
            output_time,
            year,
        )

        lat[:] = output_lat
        lon[:] = output_lon
        time[:] = days_since_by_month(year, 15)

        times = pl.from_numpy(output_time.values).to_series()

        frac_year_df = ice_fraction_df.filter(pl.col("time").dt.year().eq(year))

        print("  Beginning loop over times")
        for i, t in enumerate(times):
            frac_month_df = frac_year_df.filter(
                pl.col("time").dt.month().eq(t.month)
                & pl.col("time").dt.year().eq(year)
            )
            frac_month = assign_to_grid(
                frac_month_df.get_column("sea_fraction").to_numpy(),
                frac_month_df.get_column("grid_idx").to_numpy(),
                output_grid,
            )
            del frac_month_df

            out = (sst[i, :, :] + frac_month) + (
                lsat[i, :, :] + (1 - frac_month)
            )

            temp_anom[i, :, :] = out
            print(f"  Done with {i = }, {t = }")
            continue

        del frac_year_df

        ncfile.close()

    return None


if __name__ == "__main__":
    main()
