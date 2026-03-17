Kriging
-------

The `glomar_gridding.kriging` module contains classes and functions for interpolation via Kriging.
Two methods of Kriging are supported by `glomar_gridding`:

- Simple Kriging
- Ordinary Kriging

For each Kriging method there is a class and a function. The recommended approach is to use the
classes, the functions will be deprecated in a future version of `glomar_gridding`. The classes
require the full grid spatial covariance structure, the observation values, and the grid index of
each observation, and an optional error covariance matrix as inputs. Each grid index should be a
single index value, and represents the flattened index, and connects directly to the corresponding
index of the covariance matrices. If an error covariance matrix is provided, the covariance matrix
will be automatically subset to the grid index values, if the resulting matrix contains `nan` or `0`
values on the diagonal, then the observation values and indices are filtered to exclude these
points, and the error covariance matrix is subset again. Just initialising the class does not solve
the system, this requires the `solve` method to be called.

Preparation
===========

`glomar_gridding` provides functionality for preparing your data for the interpolation. The `grid`
module has functionality for defining the output grid
(:py:func:`glomar_gridding.grid.grid_from_resolution`) which allows the user to create a coordinate
system for the output, that can easily be mapped to a covariance matrix. The grid object is an
`xarray.DataArray` object, with a coordinate system. Once the grid is defined, the observations can
be mapped to the grid. This creates a 1-dimensional index value that should match to the covariance
matrices used in the interpolation.

.. autofunction:: glomar_gridding.grid.map_to_grid

For Kriging, the interpolation requires at most a single observation value in each grid box. If the
data contains multiple values in a single grid cell then these need to be combined.

.. autofunction:: glomar_gridding.kriging.prep_obs_for_kriging

Simple Kriging
==============

.. autoclass:: glomar_gridding.kriging.SimpleKriging
   :members:

.. autofunction:: glomar_gridding.kriging.kriging_simple

Ordinary Kriging
================

.. autoclass:: glomar_gridding.kriging.OrdinaryKriging
   :members:

.. autofunction:: glomar_gridding.kriging.kriging_ordinary

Perturbed Gridded Fields
========================

An additional two-stage combined Kriging class is provided in the `stochastic` module. In this case,
the `solve` method has an additional optional `simulated_state` argument, if this is set, then the
value is used as the simulated system state used as the base of the perturbed field (from which
observations are simulated), otherwise the value is computed. This allows for pre-computation of a
sequence of simulated states as part of ensemble processing, for example.

.. autoclass:: glomar_gridding.stochastic.StochasticKriging
   :members:

.. autofunction:: glomar_gridding.stochastic.draw_from_cov

Outputs
=======

The outputs to the solvers, :py:func:`glomar_gridding.SimpleKriging.solve` for example will be
vectors, they should be re-shaped to the grid.

Outputs can also be re-mapped to the grid

.. autofunction:: glomar_gridding.grid.assign_to_grid
