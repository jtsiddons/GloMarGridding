"""
Functions for loading data from a netCDF file using format strings to find the
file.
"""

import os
import xarray as xr


def load_dataset(
    path,
    **kwargs,
) -> xr.Dataset:
    """
    Load an xarray.Dataset from a netCDF file. Can input a filename or a
    string to format with keyword arguments.

    Parameters
    ----------
    path : str
        Full filename (including path), or filename with replacements using
        str.format with named replacements. For example:
            /path/to/global_covariance_{month:02d}.nc
    **kwargs
        Keywords arguments matching the replacements in the input path.

    Returns
    -------
    arr : xarray.Dataset
        The netcdf dataset as an xarray.Dataset.
    """
    if os.path.isfile(path):
        filename = path
    elif kwargs:
        if not os.path.isdir(os.path.dirname(path)):
            raise FileNotFoundError(f"Covariance path: {path} not found")
        filename = path.format(**kwargs)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Covariance file: {filename} not found")
    else:
        raise FileNotFoundError("Cannot determine filename")

    return xr.open_dataset(filename, engine="netcdf4")


def load_array(
    path: str,
    var: str = "covariance",
    **kwargs,
) -> xr.DataArray:
    """
    Load an xarray.DataArray from a netCDF file. Can input a filename or a
    string to format with keyword arguments.

    Parameters
    ----------
    path : str
        Full filename (including path), or filename with replacements using
        str.format with named replacements. For example:
            /path/to/global_covariance_{month:02d}.nc
    var : str
        Name of the variable to select from the input file
    **kwargs
        Keywords arguments matching the replacements in the input path.

    Returns
    -------
    arr : xarray.DataArray
        An array containing the values of the variable specified by var
    """
    return load_dataset(path, **kwargs)[var]
