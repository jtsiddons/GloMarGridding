"""
The NOC Surface Processes library for interpolating ungridded or point
observational data to in-filled gridded fields. Typically this will make use
of Kriging as the inteprolation method.
"""

from .grid import align_to_grid
from .error_covariance import (
    dist_weight,
    get_weights,
    uncorrelated_components,
    correlated_components,
)
from .kriging import kriging
from .variogram import (
    ExponentialVariogram,
    GaussianVariogram,
    LinearVariogram,
    MaternVariogram,
    PowerVariogram,
)

__all__ = [
    "ExponentialVariogram",
    "GaussianVariogram",
    "LinearVariogram",
    "MaternVariogram",
    "PowerVariogram",
    "align_to_grid",
    "correlated_components",
    "dist_weight",
    "get_weights",
    "kriging",
    "uncorrelated_components",
]
