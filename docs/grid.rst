Grid Class
----------

An alternative workflow is to use the `Grid` class added in version 1.1.0. The `Grid` class can be
used to run the full Kriging process from defining an output grid to interpolating the observational
data.

This class allows for easy masking of positions, so that masked positions are not infilled.

A new notebook was added ("Mask_example.ipynb") to demonstrate the full process.

.. code-block:: python

   from glomar_gridding.grid import Grid

   # Initialise, like with grid_from_resolution
   grid = Grid.from_resolution(
       resolution=5,
       bounds=[(-90, 90), (-180, 180)],
       coord_names=["latitude", "longitude"],
       definition="left",
   )

   # Can now easily add a mask and only used unmasked positions on the grid
   mask = ...
   grid.add_mask(mask)

   # Add distance matrix and covariance matrix
   grid.distance_matrix()
   grid.covariance_matrix(
       "matern",
       range=1300,
       psill=1.2,
       nu=1.5,
       nugget=0.0,
       method="sklearn",
   )

   # Add observations
   obs = ...
   grid.map_observations(obs, ...)

   # Can ensure an external error covariance matrix is aligned to the grid
   error_cov = ...
   error_cov = grid.prep_covariance(error_cov)

   # Perform Kriging
   grid.kriging("ordinary", error_cov)


.. autoclass:: glomar_gridding.grid.Grid
   :members:
