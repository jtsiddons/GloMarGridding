#!/usr/bin/env python

"""
Script to stitch land and sea surface temperatures using weights

By J. Siddons 2025-02.
For Python >= 3.11
"""

import os

import yaml
import polars as pl
import xarray as xr

from glomar_gridding.io import load_array, load_dataset

CONFIG_PATH: str = os.path.join(os.path.dirname(__file__), "config_stitch.yaml")
ENSEMBLES: list[int] = list(range(1, 201))
YEAR_RANGE: tuple[int, int] = (1850, 2024)


def main() -> None:  # noqa: D103
    with open(CONFIG_PATH, "r") as io:
        config: dict = yaml.safe_load(io)

    sst_path: str = config.get("sst_path", "")
    sst_var_name: str = config.get("sst_var_name", "sst_anom")

    lsat_path: str = config.get("lsat_path", "")
    lsat_var_name: str = config.get("lsat_var_name", "lsat_anom")

    out_path: str = config.get("out_path", "")
    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    out_var_name: str = config.get("out_var_name", "combined")

    weights_path: str = config.get("weights_path", "")
    weights_var_name: str = config.get("weights_var_name", "weights")

    weights = (
        pl.from_pandas(
            load_array(weights_path, weights_var_name)
            .to_dataframe()
            .reset_index()
        )
        .rename({"latitude": "lat", "longitude": "lon"})
        .filter(pl.col("time").dt.year().is_between(*YEAR_RANGE, closed="both"))
        .with_columns(pl.col("time").dt.replace(day=15, hour=12).name.keep())
    )

    weight_indicator: str = config.get("weight_indicator", "sst")

    for member in ENSEMBLES:
        print(f"Doing ensemble member: {member}")
        sst_array = (
            pl.from_pandas(
                load_dataset(sst_path, member=member)
                .to_dataframe()
                .reset_index()
            )
            .filter(
                pl.col("time").dt.year().is_between(*YEAR_RANGE, closed="both")
            )
            .with_columns(
                pl.col("time").dt.replace(day=15, hour=12).name.keep()
            )
            .rename({"epsilon": "sst_epsilon", "n_obs": "sst_n_obs"})
        )
        lsat_array = (
            pl.from_pandas(
                load_dataset(lsat_path, member=member)
                .to_dataframe()
                .reset_index()
            )
            .filter(
                pl.col("time").dt.year().is_between(*YEAR_RANGE, closed="both")
            )
            .with_columns(
                pl.col("time").dt.replace(day=15, hour=12).name.keep()
            )
            .rename({"epsilon": "lsat_epsilon", "n_obs": "lsat_n_obs"})
        )
        coords = ["time", "lat", "lon"]

        combined_df = sst_array.join(lsat_array, on=coords, how="left")
        combined_df = combined_df.join(weights, on=coords, how="left")

        out_array_member_path = out_path.format(member=member)

        match weight_indicator:
            case "sst":
                combined_df = combined_df.with_columns(
                    (
                        pl.col(sst_var_name) * pl.col(weights_var_name)
                        + (1 - pl.col(weights_var_name)) * pl.col(lsat_var_name)
                    ).alias(out_var_name)
                )
            case "lsat":
                combined_df = combined_df.with_columns(
                    (
                        pl.col(lsat_var_name) * pl.col(weights_var_name)
                        + (1 - pl.col(weights_var_name)) * pl.col(sst_var_name)
                    ).alias(out_var_name)
                )
            case _:
                raise ValueError(
                    "weight_indicator does not match an array, expect one of "
                    + "'sst' or 'lsat'"
                )

        combined_df = combined_df.sort(coords)

        xr.Dataset.from_dataframe(
            combined_df.to_pandas().set_index(coords)
        ).to_netcdf(out_array_member_path)

    return None


if __name__ == "__main__":
    main()
