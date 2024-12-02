from collections import OrderedDict
from enum import IntEnum
import netCDF4 as nc
import numpy as np
import re
import xarray as xr
from warnings import warn


class MonthName(IntEnum):
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
    nc_variables: list[nc.Variable] | nc.Variable,
    timestamps: list[int] | int,
    shape: tuple[int, int],
) -> None:
    empty = np.zeros(shape=shape).astype(np.float32)
    nc_variables = (
        [nc_variables] if not isinstance(nc_variables, list) else nc_variables
    )
    timestamps = (
        [timestamps] if not isinstance(timestamps, list) else timestamps
    )
    for variable in nc_variables:
        for timestamp in timestamps:
            variable[timestamp, :, :] = empty
    return None


class ConfigParserMultiValues(OrderedDict):
    def __setitem__(self, key, value):
        if key in self and isinstance(value, list):
            self[key].extend(value)
        else:
            super().__setitem__(key, value)

    @staticmethod
    def getlist(value):
        return value.splitlines()


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
    coords: list[str] = [str(c) for c in ds.coords]
    for coord in coords:
        test = coord.lower() if case_insensitive else coord
        if pattern.match(test):
            return coord
    raise ValueError(
        "Cannot match coordinates to pattern, possible coords = '"
        + "', '".join(coords)
        + "'."
    )


def adjust_small_negative(mat: np.ndarray) -> None:
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
    if small_negative_check.any():
        warn("Small negative vals are detected. Setting to 0.")
        print(mat[small_negative_check])
        mat[small_negative_check] = 0.0
    return None
