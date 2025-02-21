"""
Interpolation Covariance
------------------------

Functions for computing (components of) the interpolation covariance matrix
used for the interpolation step.
"""

import numpy as np

from glomar_gridding.io import load_array


def load_covariance(
    path: str, cov_var_name: str = "covariance", **kwargs
) -> np.ndarray:
    """
    Load a covariance matrix from a netCDF file. Can input a filename or a
    string to format with keyword arguments.

    Parameters
    ----------
    path : str
        Full filename (including path), or filename with replacements using
        str.format with named replacements. For example:
        /path/to/global_covariance_{month:02d}.nc
    cov_var_name : str
        Name of the variable for the covariance matrix
    **kwargs
        Keywords arguments matching the replacements in the input path.

    Returns
    -------
    covariance : numpy.ndarray
        A numpy matrix containing the covariance matrix loaded from the netCDF
        file determined by the input arguments.
    """
    return load_array(path, cov_var_name, **kwargs).values
