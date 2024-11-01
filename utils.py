import netCDF4 as nc
import numpy as np
import polars as pl
from datetime import date


def add_empty_layers(nc_variables: list[nc.Variable] | nc.Variable,
                     timestamps: list[int] | int,
                     shape: tuple[int, int],
                     ) -> None:
    empty = np.zeros(shape=shape).astype(np.float32)
    nc_variables = [nc_variables] if not isinstance(nc_variables, list) else nc_variables
    timestamps = [timestamps] if not isinstance(timestamps, list) else timestamps
    for variable in nc_variables:
        for timestamp in timestamps:
            variable[timestamp, :, :] = empty
    return None


def _daterange_by_day(year: int, day: int) -> pl.Series:
    start = date(year, 1, day)
    end = date(year, 12, day)
    dates = pl.date_range(start, end, interval="1mo", eager=True, closed="both")
    return dates


def days_since_by_month(year: int, day: int) -> np.ndarray:
    dates = _daterange_by_day(year, day)
    return (dates - date(year, 1, day)).dt.total_days().to_numpy()
