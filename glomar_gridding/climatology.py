"""Functions for mapping climatologies and computing anomalies"""

import numpy as np
import polars as pl
import xarray as xr
from glomar_gridding.utils import find_nearest


def join_climatology_by_doy(
    obs_df: pl.DataFrame,
    climatology_365: xr.DataArray,
    lat_col: str = "lat",
    lon_col: str = "lon",
    date_col: str = "date",
    var_col: str = "sst",
    clim_lat: str = "latitude",
    clim_lon: str = "longitude",
    clim_doy: str = "doy",
    clim_var: str = "climatology",
) -> pl.DataFrame:
    """
    Merge a climatology from an xarray.DataArray into a polars.DataFrame using
    the day of year value and position.

    This function accounts for leap years by taking the average of the
    climatology values for 28th Feb and 1st March for observations that were
    made on the 29th of Feb.

    The climatology is merged into the DataFrame and anomaly values are
    computed.

    Parameters
    ----------
    obs_df : polars.DataFrame
        Observational DataFrame.
    climatology_365 : xarray.DataArray
        DataArray containing daily climatology values (for 365 days).
    lat_col : str
        Name of the latitude column in the observational DataFrame.
    lon_col : str
        Name of the longitude column in the observational DataFrame.
    date_col : str
        Name of the datetime column in the observational DataFrame. Day of year
        values are computed from this value.
    var_col : str
        Name of the variable column in the observational DataFrame. The merged
        climatology names will have this name prefixed to "_climatology", the
        anomaly values will have this name prefixed to "_anomaly".
    clim_lat : str
        Name of the latitude coordinate in the climatology DataArray.
    clim_lon : str
        Name of the longitude coordinate in the climatology DataArray.
    clim_doy : str
        Name of the day of year coordinate in the climatology DataArray.
    clim_var : str
        Name of the climatology variable in the climatology DataArray.

    Returns
    -------
    obs_df : polars.DataFrame
        With the climatology merged and anomaly computed. The new columns are
        "_climatology" and "_anomaly" prefixed by the `var_col` value
        respectively.
    """
    obs_lat = obs_df.get_column(lat_col)
    lat_idx, _ = find_nearest(climatology_365.coords[clim_lat], obs_lat)

    obs_lon = obs_df.get_column(lon_col)
    lon_idx, _ = find_nearest(climatology_365.coords[clim_lon], obs_lon)

    mask = (obs_df.get_column(date_col).dt.is_leap_year()) & (
        obs_df.get_column(date_col).dt.ordinal_day().eq(60)
    )
    non_leap_df = obs_df.filter(~mask)
    leap_df = obs_df.filter(mask)
    non_leap_df = non_leap_df.with_columns(
        pl.datetime(pl.lit(2009), pl.col("mo"), pl.col("dy"))
        .dt.ordinal_day()
        .alias("doy")
    )
    doy_idx = non_leap_df.get_column("doy") - 1

    climatology = climatology_365[clim_var].values
    climatology = climatology - 273.15

    # TODO: Dynamic ordering of indices
    selected = climatology[doy_idx, lat_idx, lon_idx]

    non_leap_df["climatology"] = selected
    end_feb = climatology[
        np.repeat(np.array([58]), leap_df.height),
        lat_idx,
        lon_idx,
    ]

    beg_mar = climatology[
        np.repeat(np.array([59]), leap_df.height),
        lat_idx,
        lon_idx,
    ]

    selected2 = [(g + h) / 2 for g, h in zip(end_feb, beg_mar)]

    clim_var_name = f"{var_col}_climatology"
    anom_var_name = f"{var_col}_anomaly"
    leap_df[clim_var_name] = selected2
    obs_df = pl.concat([non_leap_df, leap_df])
    obs_df[anom_var_name] = obs_df[var_col] - obs_df[clim_var_name]

    return obs_df.drop_nulls()
