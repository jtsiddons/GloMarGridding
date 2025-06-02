Stationary Interpolation Covariance
-----------------------------------

Outside of the observation values and positions, the spatial covariance structure is the most
important component for Kriging. The Kriging classes in this library all require this matrix as the
input to the class. This covariance matrix can be provided as a pre-computed matrix, loaded into the
environment with :py:func:`glomar_gridding.interpolation_covariance.load_covariance`. Alternatively
the covariance structure can be estimated using functions and classes contained within
`glomar_gridding`.

A commonly used approach is to compute a stationary covariance structure, using a *Variogram* with
fixed scales. The use of *stationary* here suggests that the range of covariance is constant across
all positions - each position has the same influence over a fixed distance.

.. code-block:: python

    from glomar_gridding.grid import grid_from_resolution, grid_to_distance_matrix
    from glomar_gridding.variogram import GaussianVariogram, variogram_to_covariance


    # Initialise a grid
    grid = grid_from_resolution(
        resolution=5,
        bounds=[(-87.5, 90), (-177.5, 180)],
        coord_names=["latitude", "longitude"],
    )

    # Compute a distance matrix
    dist = grid_to_distance_matrix(
        grid=grid,
        lat_coord="latitude",
        lon_coord="longitude",
    )

    # Define and compute a Variogram
    variogram = GaussianVariogram(
        nugget=0.0,
        psill=1.2,
        range=1300,
    ).fit(dist)

    # Convert to covariance
    covariance = variogram_to_covariance(variogram, sill=1.2)


Grid
====

.. autofunction:: glomar_gridding.grid.grid_from_resolution

.. autofunction:: glomar_gridding.grid.grid_to_distance_matrix

Variograms
==========

.. autoclass:: glomar_gridding.variogram.ExponentialVariogram
   :members:

.. autoclass:: glomar_gridding.variogram.SphericalVariogram
   :members:

.. autoclass:: glomar_gridding.variogram.GaussianVariogram
   :members:

.. autoclass:: glomar_gridding.variogram.MaternVariogram
   :members:

.. autofunction:: glomar_gridding.variogram.variogram_to_covariance

Covariance
==========

.. automodule:: glomar_gridding.interpolation_covariance
   :members:
