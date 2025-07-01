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

"""I/O functionality for loading a covariance matrix from disk."""

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
