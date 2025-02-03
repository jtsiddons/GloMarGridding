"""Functions for mapping climatologies and computing anomalies"""

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
    temp_from_kelvin: bool = True,
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
    temp_from_kelvin : bool
        Optionally adjust the climatology from Kelvin to Celcius if the variable
        is a temperature.

    Returns
    -------
    obs_df : polars.DataFrame
        With the climatology merged and anomaly computed. The new columns are
        "_climatology" and "_anomaly" prefixed by the `var_col` value
        respectively.
    """
    # Names of the output columns
    clim_var_name = f"{var_col}_climatology"
    anom_var_name = f"{var_col}_anomaly"

    climatology = pl.from_pandas(
        climatology_365[clim_var].to_dataframe().reset_index(drop=False)
    )
    if temp_from_kelvin:
        climatology = climatology.with_columns(
            (pl.col(clim_var) - 273.15).name.keep()
        )
    climatology = climatology.select([clim_lat, clim_lon, clim_doy, clim_var])

    obs_lat = obs_df.get_column(lat_col)
    _, lat_vals = find_nearest(climatology_365.coords[clim_lat], obs_lat)

    obs_lon = obs_df.get_column(lon_col)
    _, lon_vals = find_nearest(climatology_365.coords[clim_lon], obs_lon)

    obs_df = obs_df.with_columns(
        pl.Series("clim_lat", lat_vals),
        pl.Series("clim_lon", lon_vals),
    )

    mask = (obs_df.get_column(date_col).dt.is_leap_year()) & (
        obs_df.get_column(date_col).dt.ordinal_day().eq(60)
    )

    non_leap_df = (
        obs_df.filter(~mask)
        .with_columns(
            pl.datetime(pl.lit(2009), pl.col("mo"), pl.col("dy"))
            .dt.ordinal_day()
            .alias("doy")
        )
        .join(
            climatology,
            left_on=["clim_lat", "clim_lon", "doy"],
            right_on=[clim_lat, clim_lon, clim_doy],
            how="left",
            coalesce=True,
        )
        .drop(["clim_lat", "clim_lon", "doy"])
    )

    # Take average of 28th Feb and 1st March for 29th Feb
    leap_clim = (
        climatology.filter(pl.col(clim_doy).is_between(59, 60, closed="both"))
        .group_by([clim_lat, clim_lon])
        .agg(pl.col(clim_var).mean())
    )

    leap_df = (
        obs_df.filter(mask)
        .join(
            leap_clim,
            left_on=["clim_lat", "clim_lon"],
            right_on=[clim_lat, clim_lon],
            how="left",
            coalesce=True,
        )
        .drop(["clim_lat", "clim_lon"])
    )

    del climatology, leap_clim

    obs_df = pl.concat([non_leap_df, leap_df])
    obs_df = obs_df.with_columns(
        (pl.col(var_col) - pl.col(clim_var_name)).alias(anom_var_name)
    )

    return obs_df.drop_nulls()
