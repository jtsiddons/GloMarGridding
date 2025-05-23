r"""Utility functions for `glomar_gridding`"""

from calendar import isleap
from collections import OrderedDict
from collections.abc import Iterable
from datetime import date, timedelta
from enum import IntEnum
import inspect
from itertools import islice
import logging
from typing import Any, TypeVar
import netCDF4 as nc
import numpy as np
import polars as pl
import xarray as xr
from warnings import warn
from polars._typing import ClosedInterval


from glomar_gridding.constants import (
    KM_TO_NM,
    NM_PER_LAT,
)

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
    ret = mat.copy()
    if small_negative_check.any():
        warn("Small negative vals are detected. Setting to 0.")
        print(mat[small_negative_check])
        ret[small_negative_check] = 0.0
    return ret


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


def _get_logging_level(level: str) -> int:
    match level.lower():
        case "debug":
            level_i = 10
        case "info":
            level_i = 20
        case "warn":
            level_i = 30
        case "error":
            level_i = 40
        case "critical":
            level_i = 50
        case _:
            raise ValueError(f"Unknown logging level: {level}")
    return level_i


def init_logging(
    file: str | None = None,
    level: str = "DEBUG",
) -> None:
    """
    Initialise the logger

    Parameters
    ----------
    file : str
        File to send log messages to. If set to None (default) then print log
        messages to STDout
    level : str
        Level of logging, one of: "debug", "info", "warn", "error", "critical".

    Returns
    -------
    None
    """
    from importlib import reload

    level_i: int = _get_logging_level(level)

    reload(logging)  # Clear the logging from cdm_reader_mapper
    logging.basicConfig(
        filename=file,
        filemode="a",
        encoding="utf-8",
        format="%(levelname)s at %(asctime)s : %(message)s",
        level=level_i,
    )
    logging.captureWarnings(True)
    return None


def get_date_index(year: int, month: int, start_year: int) -> int:
    """
    Get the index of a given year-month in a monthly sequence of dates
    starting from month 1 in a specific start year

    Parameters
    ----------
    year : int
        The year for the date to find the index of.
    month : int
        The month for the date to find the index of.
    start_year : int
        The start year of the date series, the result assumes that the date
        time series starts in the first month of this year.

    Returns
    -------
    index : int
        The index of the input date in the monthly datetime series starting from
        the first month of year `start_year`.
    """
    return 12 * (year - start_year) + (month - 1)


def deg_to_nm(deg: float) -> float:
    """
    deg: float (degrees)
    Convert degree latitude change to nautical miles
    """
    return NM_PER_LAT * deg


def deg_to_km(deg: float) -> float:
    """
    deg: float (degrees)
    Convert degree latitude change to km
    """
    return KM_TO_NM * deg_to_nm(deg)


def km_to_deg(km: float) -> float:
    """
    km: float (km)
    Convert meridonal km change to degree latitude
    """
    return (km / KM_TO_NM) / NM_PER_LAT


def is_iter(val: Any) -> bool:
    """Determine if a value is an iterable"""
    try:
        iter(val)
        return True
    except TypeError:
        return False


def uncompress_masked(
    compressed_array: np.ndarray,
    mask: np.ndarray,
    fill_value: Any = 0.0,
    apply_mask: bool = False,
    dtype: type | None = None,
) -> np.ndarray | np.ma.MaskedArray:
    """
    Un-compress a compressed array using a mask.

    Parameters
    ----------
    compressed_array : numpy.ndarray
        The compressed array, originally compressed by the mask
    mask : numpy.ndarray
        The mask - a boolean numpy array
    fill_value : Any
        The value to fill masked points. If `apply_mask` is set, then this will
        be removed in the output.
    apply_mask : bool
        Apply the mask to the result, returning a MaskedArray rather than a
        ndarray.
    dtype : type | None
        Optionally set a dtype for the returned array, if not set then the
        dtype of the compressed_array is used.

    Returns
    -------
    uncompressed : numpy.ndarray | numpy.ma.MaskedArray
        The uncompressed array, masked points are filled with the fill_value if
        apply_mask is False. If apply_mask is True, then the result is an
        instance of numpy.ma.MaskedArray with the mask applied to the
        uncompressed result.
    """
    not_mask = np.logical_not(mask)
    if np.sum(not_mask) != len(compressed_array):
        raise ValueError("Length of compressed_array does not align with mask")

    dtype = dtype or compressed_array.dtype

    uncompressed = np.empty_like(mask, dtype=dtype)
    np.place(uncompressed, not_mask, compressed_array)

    if apply_mask:
        return np.ma.masked_where(mask, uncompressed)

    np.place(uncompressed, mask, fill_value)
    return uncompressed


def cor_2_cov(
    cor: np.ndarray,
    variances: np.ndarray,
    rounding: int | None = None,
) -> np.ndarray:
    """
    Compute covariance matrix from correlation matrix and variances

    Parameters
    ----------
    cor : numpy.ndarray
        Correlation Matrix
    variances : numpy.ndarray
        Variances to scale the correlation matrix.
    rounding : int
        round the values of the output
    """
    stdevs = np.sqrt(variances)
    normalisation = np.outer(stdevs, stdevs)
    cov = cor * normalisation
    cov[cor == 0] = 0
    if rounding is not None:
        cov = np.round(cov, rounding)
    return cov


def cov_2_cor(
    cov: np.ndarray,
    rounding: int | None = None,
) -> np.ndarray:
    """
    Normalises the covariance matrices within the class instance
    and return correlation matrices
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix
    rounding : int
        round the values of the output
    """
    stdevs = np.sqrt(np.diag(cov))
    normalisation = np.outer(stdevs, stdevs)
    cor = cov / normalisation
    if not np.all(np.diag(cor) == 1.0):
        print(np.max(np.abs(np.diag(cor) - 1.0)))
        if np.max(np.abs(np.diag(cor) - 1.0)) > 1e-6:
            raise ValueError(
                "Correlation Diagonal contains values not close to 1."
            )  # This should never get flagged:
        print(
            "Numerical error correction applied to correlation matrix diagonal"
        )
        np.fill_diagonal(cor, 1.0)
    cor[cov == 0] = 0
    if rounding is not None:
        cor = np.round(cor, rounding)
    return cor


def mask_array(arr: np.ndarray) -> np.ma.MaskedArray:
    """
    Forces numpy array to be an instance of np.ma.MaskedArray

    Parameters
    ----------
    arr : np.ndarray
        Can be masked or not masked

    Returns
    -------
    arr : np.ndarray
        array is now an instance of np.ma.MaskedArray
    """
    if isinstance(arr, np.ma.MaskedArray):
        return arr
    if isinstance(arr, np.ndarray):
        logging.info("Ad hoc conversion to np.ma.MaskedArray")
        arr = np.ma.MaskedArray(arr)
        return arr
    raise TypeError("Input is not a numpy array.")


def batched(iterable: Iterable, n: int, *, strict: bool = False):
    """
    Implementation of itertools.batched for use if python version is < 3.12.

    Examples
    --------
    >>> list(batched("ABCDEFG", 3))
    [("A", "B", "C"), ("D", "E", "F"), ("G", )]
    """
    if n < 1:
        raise ValueError("'n' must be >= 1")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def get_month_midpoint(dates: pl.Series) -> pl.Series:
    """
    Get the month midpoint for a series of datetimes.

    The midpoint of a month is the exact half-way point between the start and
    end of the month.

    For example, the midpoint of January 1990 is 1990-01-16 12:00.
    """
    if not dates.dtype.is_temporal():
        raise TypeError("Input is not a datetime series")

    month_len = (
        dates.dt.month_end().dt.date()
        - dates.dt.month_start().dt.date().dt.offset_by("1d")
    )
    dates = dates.dt.month_start() + (month_len.cast(pl.Int64) / 2).cast(
        pl.Duration("ms")
    )

    return dates


def sizeof_fmt(num: float, suffix="B") -> str:
    """
    Convert numbers to kilo/mega... bytes, for interactive printing of code
    progress.
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_spatial_mean(
    grid_obs: np.ndarray,
    covx: np.ndarray,
) -> float:
    """
    Compute the spatial mean accounting for auto-correlation. See [Cornell]_

    Parameters
    ----------
    grid_obs : np.ndarray
        Vector containing observations
    covx : np.ndarray
        Observation covariance matrix

    Returns
    -------
    spatial_mean : float
        The spatial mean defined as (1^T x C^{-1} x 1)^{-1} * (1^T x C^{-1} x z)

    References
    ----------
    [Cornell]_ https://www.css.cornell.edu/faculty/dgr2/_static/files/distance_ed_geostats/ov5.pdf
    """
    n = len(grid_obs)
    ones = np.ones(n)
    invcov = ones.T @ np.linalg.inv(covx)

    return float(1 / (invcov @ ones) * (invcov @ grid_obs))
