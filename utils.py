import netCDF4 as nc
import numpy as np
import polars as pl
from datetime import date
from warnings import warn


def add_empty_layers(
    nc_variables: list[nc.Variable] | nc.Variable,
    timestamps: list[int] | int,
    shape: tuple[int, int],
) -> None:
    empty = np.zeros(shape=shape).astype(np.float32)
    nc_variables = (
        [nc_variables] if not isinstance(nc_variables, list) else nc_variables
    )
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


def adjust_small_negative(mat: np.ndarray[float]) -> None:
    """
    Adjusts small negative values (with absolute value < 1e-8)
    in matrix to 0 in-place.

    Raises a warning if any small negative values are detected.

    Parameters
    ----------
    mat : np.ndarray[float]
          Squared uncertainty associated with chosen kriging method
          Derived from the diagonal of the matrix
    """
    small_negative_check = np.logical_and(np.isclose(mat, 0, atol=1e-08), mat < 0.0)
    if small_negative_check.any():
        warn("Small negative vals are detected. Setting to 0.")
        print(mat[small_negative_check])
        mat[small_negative_check] = 0.0
    return None
