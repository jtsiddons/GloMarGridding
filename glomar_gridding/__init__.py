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
The NOC Surface Processes library for interpolating ungridded or point
observational data to in-filled gridded fields. Typically this will make use
of Kriging as the inteprolation method.
"""

from .error_covariance import (
    correlated_components,
    dist_weight,
    get_weights,
    uncorrelated_components,
)
from .grid import map_to_grid
from .variogram import (
    ExponentialVariogram,
    GaussianVariogram,
    MaternVariogram,
    SphericalVariogram,
)

__all__ = [
    "ExponentialVariogram",
    "GaussianVariogram",
    "MaternVariogram",
    "SphericalVariogram",
    "correlated_components",
    "dist_weight",
    "get_weights",
    "map_to_grid",
    "uncorrelated_components",
]

__version__ = "1.1.0"
