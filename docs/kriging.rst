Kriging
-------

The `glomar_gridding.kriging` module contains classes and functions for interpolation via Kriging.
Two methods of Kriging are supported by `glomar_gridding`:

- Simple Kriging
- Ordinary Kriging

For each Kriging method there is a class and a function. The recommended approach is to use the
classes, the functions will be deprecated in a future version of `glomar_gridding`. The classes
require the full grid spatial covariance structure as an input. Each class contains a `solve` method
that requires the observation values and the 1-dimensional grid index of each observation,
optionally an error covariance matrix can be provided. The grid index values are used to index into
the covariance matrix to obtain the inputs for the Kriging equations.

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

An additional two-stage combined Kriging class is provided in the `stochastic` module.

.. autoclass:: glomar_gridding.stochastic.StochasticKriging
   :members:

.. autofunction:: glomar_gridding.stochastic.scipy_mv_normal_draw

Outputs
=======

The outputs to the solvers, :py:func:`glomar_gridding.SimpleKriging.solve` for example will be
vectors, they should be re-shaped to the grid.

Outputs can also be re-mapped to the grid

.. autofunction:: glomar_gridding.grid.assign_to_grid
