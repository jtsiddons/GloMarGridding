"""Utility functions for GlomarGridding"""

from calendar import isleap
from collections import OrderedDict
from collections.abc import Iterable
from datetime import date, timedelta
from enum import IntEnum
import inspect
import logging
from typing import TypeVar
import netCDF4 as nc
import numpy as np
import polars as pl
import re
import xarray as xr
from warnings import warn
from polars._typing import ClosedInterval

_XR_Data = TypeVar("_XR_Data", xr.DataArray, xr.Dataset)


class ColumnNotFoundError(Exception):
    """Error class for Column Not Being Found"""

    pass


class MonthName(IntEnum):
    """Name of month from int"""

    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12


def add_empty_layers(
    nc_variables: Iterable[nc.Variable] | nc.Variable,
    timestamps: Iterable[int] | int,
    shape: tuple[int, int],
) -> None:
    """
    Add empty layers to a netcdf file. This adds a layer of zeros to the netCDF
    file.

    Parameters
    ----------
    nc_variables : Iterable[nc.Variable] | nc.Variable
        Name(s) of the variables to add empty layers to
    timestamps : Iterable[int] | int
        Indices to add empty layers
    shape : tuple[int, int]
        Shape of the layer to add
    """
    empty = np.zeros(shape=shape).astype(np.float32)
    nc_variables = (
        [nc_variables]
        if not isinstance(nc_variables, Iterable)
        else nc_variables
    )
    timestamps = (
        [timestamps] if not isinstance(timestamps, Iterable) else timestamps
    )
    for variable in nc_variables:
        for timestamp in timestamps:
            variable[timestamp, :, :] = empty
    return None


class ConfigParserMultiValues(OrderedDict):
    """Internal Helper Class"""

    def __setitem__(self, key, value):
        if key in self and isinstance(value, list):
            self[key].extend(value)
        else:
            super().__setitem__(key, value)

    @staticmethod
    def getlist(value):  # noqa: D102
        return value.splitlines()


# DELETE: Unused
def match_coord(
    ds: xr.Dataset,
    candidate_names: list[str] | str,
    case_insensitive: bool = True,
) -> str:
    """
    Identify a coordinate from a dataset based on exact match. An error is
    raised if no match is found.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing coordinates to match against.
    candidate_names : list[str] | str
        List of possible values for the coordinate.
    case_insensitive : bool
        Match the coordinate ignoring case. Setting to True will test lowercase
        coordinate names against the candidates. Will return the original
        coordinate name.

    Returns
    -------
    coord : str
        Name of the coordinate matching any candidate.
    """
    if isinstance(candidate_names, str):
        candidate_names = [candidate_names]
    coords: list[str] = [str(c) for c in ds.coords]
    for coord in coords:
        test = coord.lower() if case_insensitive else coord
        if test in candidate_names:
            return coord
    raise ValueError(
        "Cannot find candidate coordinate name, possible coords = '"
        + "', '".join(coords)
        + "'."
    )


# DELETE: Unused
def regex_coord(
    ds: xr.Dataset,
    pattern: re.Pattern,
    case_insensitive: bool = True,
) -> str:
    """
    Identify a coordinate from a dataset based on regex match. An error is
    raised if no match is found.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing coordinates to match against.
    pattern : re.Pattern
        Regular expression to use for coordinate identification.
    case_insensitive : bool
        Match the coordinate ignoring case. Setting to True will test lowercase
        coordinate names against the candidates. Will return the original
        coordinate name.

    Returns
    -------
    coord : str
        Name of the coordinate matching the regex pattern.
    """
    coords: list[str] = [str(c) for c in ds.indexes]
    for coord in coords:
        test = coord.lower() if case_insensitive else coord
        if pattern.match(test):
            return coord
    raise ValueError(
        "Cannot match coordinates to pattern, possible coords = '"
        + "', '".join(coords)
        + "'."
    )


def _daterange_by_day(year: int, day: int) -> pl.Series:
    start = date(year, 1, day)
    end = date(year, 12, day)
    dates = pl.date_range(start, end, interval="1mo", eager=True, closed="both")
    return dates


def days_since_by_month(year: int, day: int) -> np.ndarray:
    """
    Get the number of days since `year`-01-`day` for each month. This is used
    to set the time values in a netCDF file where temporal resolution is monthly
    and the units are days since some date.
    """
    dates = _daterange_by_day(year, day)
    return (dates - date(year, 1, day)).dt.total_days().to_numpy()


def adjust_small_negative(mat: np.ndarray) -> np.ndarray:
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
    small_negative_check = np.logical_and(
        np.isclose(mat, 0, atol=1e-08), mat < 0.0
    )
    # Calls from kriging_ordinary and kriging_simple use np.diag
    # np.diag returns an immutable view of the array; .copy is required. See:
    # https://numpy.org/doc/2.1/reference/generated/numpy.diagonal.html#numpy.diagonal
    ans = mat.copy()
    if small_negative_check.any():
        warn("Small negative vals are detected. Setting to 0.")
        print(mat[small_negative_check])
        ans[small_negative_check] = 0.0
    return ans


def find_nearest(
    array: Iterable,
    values: Iterable,
) -> tuple[list[int], np.ndarray]:
    """
    Get the indices and values from an array that are closest to the input
    values.

    A single index, value pair is returned for each look-up value in the values
    list.

    Parameters
    ----------
    array : Iterable
        The array to search for nearest values.
    values : Iterable
        The values to look-up in the array.

    Returns
    -------
    idx_list : list[int]
        The indices of nearest values
    array_values_list : list
        The list of values in array that are closest to the input values.
    """
    idx_list = [(np.abs(array - value)).argmin() for value in values]
    array_values_list = np.array(array)[idx_list]
    # print(values)
    # print(array_values_list)
    return idx_list, array_values_list


def select_bounds(
    x: _XR_Data,
    bounds: list[tuple[float, float]] = [(-90, 90), (-180, 180)],
    variables: list[str] = ["lat", "lon"],
) -> _XR_Data:
    """
    Filter an xarray.DataArray or xarray.Dataset by a set of bounds.

    Parameters
    ----------
    x : xarray.DataArray | xarray.Dataset
        The data to filter
    bounds : list[tuple[float, float]]
        A list of tuples containing the lower and upper bounds for each
        dimension.
    variables : list[str]
        Names of the dimensions (the order must match the bounds).

    Returns
    -------
    x : xarray.DataArray | xarray.Dataset
        The input data filtered by the bounds.
    """
    bnd_map: dict[str, slice] = {
        b: slice(*v) for b, v in zip(variables, bounds)
    }
    return x.sel(bnd_map)


def intersect_mtlb(a, b):
    """
    Returns data common between two arrays, a and b, in a sorted order and index
    vectors for a and b arrays Reproduces behaviour of Matlab's intersect
    function.

    Parameters
    ----------
    a (array) - 1-D array
    b (array) - 1-D array

    Returns
    -------
    1-D array, c, of common values found in two arrays, a and b, sorted in order
    List of indices, where the common values are located, for array a
    List of indices, where the common values are located, for array b
    """
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


def check_cols(
    df: pl.DataFrame,
    cols: list[str],
) -> None:
    """Check that all columns in a list of columns are in a DataFrame"""
    # Get name of function that is calling this
    calling_func = str(inspect.stack()[1][3])

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ColumnNotFoundError(
            calling_func
            + ": DataFrame is missing required columns: "
            + ", ".join(missing_cols)
        )
    return None


def filter_bounds(
    df: pl.DataFrame,
    bounds: list[tuple[float, float]],
    bound_cols: list[str],
    closed: ClosedInterval | list[ClosedInterval] = "left",
) -> pl.DataFrame:
    """
    Filter a polars DataFrame based on a set of lower and upper bounds.

    Parameters
    ----------
    df : polars.DataFrame
        The data to be filtered by the bounds
    bounds : list[tuple[float, float]]
        A list of tuples containing lower and upper bounds for a column
    bound_cols : list[str]
        A list of column names to be filtered by the bounds, the length of
        the bounds list must equal the length of the bound_cols list.
    closed : str | list[str]
        One of "both", "left", "right", "none" indicating the closedness of
        the bounds. If the input is a single instance then all bounds will have
        that closedness. If it is a list of closed values then its length must
        match the length of the bounds list.
    """
    if len(bounds) != len(bound_cols):
        raise ValueError("Length of 'bounds' must equal length of 'bound_cols'")

    if not isinstance(closed, list):
        closed = [closed for _ in range(len(bounds))]

    if len(closed) != len(bounds):
        raise ValueError(
            "Length of 'closed' must equal length of 'bounds', "
            + "or be a single value."
        )

    check_cols(df, bound_cols)

    # Dynamically build the filter condition
    condition: pl.Expr = pl.col(bound_cols[0]).is_between(
        *bounds[0], closed=closed[0]
    )
    for bound, col, close in zip(bounds[1:], bound_cols[1:], closed[1:]):
        condition = condition & (pl.col(col).is_between(*bound, closed=close))

    return df.filter(condition)


# TODO: get pentad convention
def get_pentad_range(centre_date: date) -> tuple[date, date]:
    """
    Get the start and date of a pentad centred at a centre date. If the
    pentad includes the leap date of 29th Feb then the pentad will include
    6 days. This follows the ***** pentad convention.

    The start and end date are first calculated from a non-leap year.

    If the centre date value is 29th Feb then the pentad will be a pentad
    starting on 27th Feb and ending on 2nd March.

    Parameters
    ----------
    centre_date : datetime.date
        The centre date of the pentad. The start date will be 2 days before this
        date, and the end date will be 2 days after.

    Returns
    -------
    start_date : datetime.date
        Two days before centre_date
    end_date : datetime.date
        Two days after centre_date
    """
    centre_year = centre_date.year
    if isleap(centre_year) and not (
        centre_date.month == 2 and centre_date.day == 29
    ):
        fake_non_leap_year = 2003
        current_date = centre_date.replace(year=fake_non_leap_year)
        start_date = (current_date - timedelta(days=2)).replace(
            year=centre_year
        )
        end_date = (current_date + timedelta(days=2)).replace(year=centre_year)
    else:
        start_date = centre_date - timedelta(days=2)
        end_date = centre_date + timedelta(days=2)
    return start_date, end_date


def init_logging(file: str | None = None) -> None:
    """
    Initialise the logger

    Parameters
    ----------
    file : str | None
        File to send log messages to. If set to None (default) then print log
        messages to STDout
    """
    from importlib import reload

    reload(logging)  # Clear the logging from cdm_reader_mapper
    if file is None:
        logging.basicConfig(
            format="\n%(levelname)s at %(asctime)s : %(message)s\n",
            level=logging.INFO,
        )
    else:
        logging.basicConfig(
            filename=file,
            filemode="a",
            encoding="utf-8",
            format="%(levelname)s at %(asctime)s : %(message)s",
            level=logging.INFO,
        )

    logging.captureWarnings(True)
    return None
