Example Workflow
----------------

Here we present a simple example, where we use Ordinary Kriging to interpolate observational data.

A full example can be found in the notebooks directory of the repository.

Load Observations
=================

In this hypothetical example, we are working with data from a `csv` file. This is assumed to be
point-observation data.

.. code-block:: python

   obs = pl.read_csv("/path/to/obs.csv", ...)

Output Grid
===========

The first step is to create the output grid, using
:py:func:`glomar_gridding.grid.grid_from_resolution` to specify a global grid with a fixed
resolution. Here, the grid has a 5-degree resolution.

The grid is essentially an empty `xarray.DataArray` with a coordinate system defined by the input
parameters. This can be used to map to the observations, and to create a covariance matrix with
consistent coordinates.

.. code-block:: python

   grid = grid_from_resolution(
       resolution=5,
       bounds=[(-87.5, 90), (-177.5, 180)],
       coord_names=["latitude", "longitude"],
   )

Align Observations
==================

The input observations may not be located at grid-box locations, i.e. they may be located somewhere
between grid-box centres. :py:func:`glomar_gridding.grid.map_to_grid` can be used to map each point
observation to a grid-box by identifying the nearest grid-box position from an input grid.

This adds an index column to the observational data-frame, which is the 1-dimensional index value of
the coordinate. This allows for easy mapping to the indices of covariance matrices, etc.

.. code-block:: python

   obs = map_to_grid(
       obs=obs,
       grid=grid,
   )

In this example, it is assumed that the observations are such that at most one observation is
associated with a grid-box. If this is not the case, the observations can be combined into a
grid-box *super*-observation with a weighting using
:py:func:`glomar_gridding.kriging.prep_obs_for_kriging`.

Extract the observation values, and the grid-box index for each observation. In this example, the
observation value is stored in the `"val"` column.

.. code-block:: python

   grid_obs = obs["val"]
   grid_idx = obs["grid_idx"]

Create or Load Spatial Covariance
=================================

The grid can be converted in to a distance matrix, and finally to a covariance matrix using a
:py:class:`glomar_gridding.variogram.Variogram` object, for example
:py:class:`glomar_gridding.variogram.GaussianVariogram`.

.. code-block:: python

   dist = grid_to_distance_matrix(
       grid=grid,
       lat_coord="latitude",
       lon_coord="longitude",
   )

   variogram = GaussianVariogram(
       range=1200,
       psill=1.2,
       nugget=0.0,
   ).fit(dist)

   covariance = variogram_to_covariance(variogram, variance=1.2)

Alternatively, the covariance matrix can be loaded from disk. A non-stationary (varying parameter)
covariance matrix can be estimated using ellipse-based models. See
:py:class:`glomar_gridding.ellipse.EllipseModel`.

Optionally Load Error Covariance
================================

In this example an error covariance matrix is loaded from a netCDF file on disk, using
:py:func:`glomar_gridding.io.load_array`

.. code-block:: python

   error_cov = load_array("/path/to/error_cov.nc", var="error_covariance")

Alternatively, an error covariance matrix can be computed component wise.

Ordinary Kriging
================

In this example, we will infill the observations using Ordinary Kriging. For this, we use
:py:class:`glomar_gridding.kriging.OrdinaryKriging`, which requires a spatial covariance matrix,
observation grid indices, observation values, and (optionally) error covariance as inputs.

.. code-block:: python

   ok = OrdinaryKriging(
       covariance.values,
       idx=grid_idx,
       obs=grid_obs,
       error_cov=error_cov,
   )

We can now use this class-instance to solve the system, using the `solve` method.

.. code-block:: python

   result = ok.solve()

Finally, the output can be mapped back on to the grid using
:py:func:`glomar_gridding.grid.assign_to_grid`

.. code-block:: python

   gridded_result = assign_to_grid(
        values=result,
        grid_idx=np.arange(grid.size),
        grid=grid,
   )
