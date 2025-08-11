# Copyright 2025 National Oceanography Centre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
`glomar_gridding` includes functionality for loading datasets or arrays from
`netCDF` files using python format strings. This can be useful for loading
pre-computed inputs for the Kriging process, for example covariance matrices or
observations. The allowance for passing a string containing format components
(e.g. python t-string) allows for dynamic configuration if processing a series
of monthly inputs for example.

Also included is a function for recursively getting sub-keys from a python
`dict` style object. This can be useful for working with `yaml` formatting
configuration files for instance.
"""

import os
from typing import Any

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
    dirname = os.path.dirname(path) or "."
    if os.path.isfile(path):
        filename = path
    elif kwargs:
        filename = path.format(**kwargs)
        if not os.path.isdir(dirname):
            raise FileNotFoundError(f"Array path: {path} not found")
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Array file: {filename} not found")
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


def get_recurse(config: dict, *keys, default: Any = None) -> Any:
    """
    Recursively get sub keys from a python dict object.

    If a dictionary object contains keys whose values are themselves
    dictionaries, get a value from a sub dictionary by specifying the key-path
    to get to the desired value.

    Equivalent to:

    .. code-block:: python

        config[keys[0]][keys[1]]...[keys[n]]

    Or:

    .. code-block:: python

        config.get(keys[0]).get(keys[1])...get(keys[n])

    Parameters
    ----------
    config : dict
        The layered dictionary containing sub dictionaries.
    *keys
        The sequence of keys to recurse through to get the final value. If any
        key in the sequence is not found, or is not a dictionary (and is not
        the final key), then the default value is returned.
    default : Any
        The default value, returned if the sequence of keys cannot be completed
        or the final key is not present.

    Returns
    -------
    The value associated with the final key, or the default value if the final
    key cannot be reached.
    """
    if len(keys) == 1:
        return config.get(keys[0], default)
    new_config = config.get(keys[0])
    if new_config is None or not isinstance(new_config, dict):
        return default
    return get_recurse(new_config, *keys[1:], default=default)
