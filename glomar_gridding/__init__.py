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

__citation__ = """
Cornes, R. C., S. C.Chan, A.Cable, et al. 2026. “GloMarGridding: A Python
Toolkit for Flexible Spatial Interpolation in Climate Applications.” Geoscience
Data Journal, 13 (2): e70064. https://doi.org/10.1002/gdj3.70064.
"""

__bibtex__ = """
@article{https://doi.org/10.1002/gdj3.70064,
author = {Cornes, Richard C. and Chan, Steven C. and Cable, Archie and Chan, Duo and Faulkner, Agnieszka and Kent, Elizabeth C. and Siddons, Joseph T.},
title = {GloMarGridding: A Python Toolkit for Flexible Spatial Interpolation in Climate Applications},
journal = {Geoscience Data Journal},
volume = {13},
number = {2},
pages = {e70064},
doi = {https://doi.org/10.1002/gdj3.70064},
url = {https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/gdj3.70064},
eprint = {https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/gdj3.70064},
note = {e70064 GDJ-2025-08-0064},
abstract = {ABSTRACT Global surface temperature datasets are constructed through processing chains that inherently introduce structural uncertainty, arising from choices made both in the processing of input observations and in the spatial interpolation methods employed. Because these steps are often tightly integrated, it is difficult to isolate their individual contributions to uncertainty. Here, we introduce GloMarGridding, a Python package designed to support the evaluation of the component of structural uncertainty arising specifically from spatial interpolation. It provides tools to apply Gaussian Process Regression Modelling (GPRM), widely used in the production of global temperature datasets, enabling the generation of spatially complete temperature fields from grid-box average and point observations, along with estimation of uncertainty in those fields. GloMarGridding currently supports three spatial covariance parametrizations: fixed isotropic variograms, ellipse-based anisotropic, and empirically derived covariance matrices. It also allows for uncertainty propagation via error covariance matrices and conditional simulation from input ensembles. By decoupling spatial interpolation from earlier stages of dataset development—such as homogenization, quality control, and aggregation—this framework enables independent assessment of upstream processing choices and their impacts on gridded outputs.},
year = {2026}
}
"""  # noqa: E501
