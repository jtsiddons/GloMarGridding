#!/usr/bin/env python

"""
Script to stitch land and sea surface temperatures using weights

By J. Siddons 2025-02.
For Python >= 3.11
"""

import os

import yaml

from glomar_gridding.array import merge_by_weights
from glomar_gridding.io import load_array

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

    weights = load_array(weights_path, weights_var_name)
    weights = weights.sel(
        YEAR_RANGE[0] <= weights.time.dt.year <= YEAR_RANGE[1]
    )

    weight_indicator: str = config.get("weight_indicator", "sst")

    for member in ENSEMBLES:
        sst_array = load_array(sst_path, var=sst_var_name, member=member)
        lsat_array = load_array(lsat_path, var=lsat_var_name, member=member)

        out_array_member_path = out_path.format(member=member)

        match weight_indicator:
            case "sst":
                combined = merge_by_weights(
                    left=sst_array,
                    right=lsat_array,
                    weights=weights,
                    output_name=out_var_name,
                )
            case "lsat":
                combined = merge_by_weights(
                    left=lsat_array,
                    right=sst_array,
                    weights=weights,
                    output_name=out_var_name,
                )
            case _:
                raise ValueError(
                    "weight_indicator does not match an array, expect one of "
                    + "'sst' or 'lsat'"
                )

        combined.to_netcdf(out_array_member_path)

    return None


if __name__ == "__main__":
    main()
