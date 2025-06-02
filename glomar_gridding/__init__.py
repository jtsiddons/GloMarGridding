"""
The NOC Surface Processes library for interpolating ungridded or point
observational data to in-filled gridded fields. Typically this will make use
of Kriging as the inteprolation method.
"""

from .grid import map_to_grid
from .error_covariance import (
    dist_weight,
    get_weights,
    uncorrelated_components,
    correlated_components,
)
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

__version__ = "1.0.0-rc.1"
