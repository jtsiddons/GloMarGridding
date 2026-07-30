Spline Interpolation
--------------------

The `spline` module brings an alternative approach to interpolation using thin-plate-spline
methodology to perform interpolation. We provide a standard thin-plate-spline interpolator using
the traditional thin-plate-spline kernel:

.. math::
   r^2 \times log(r)

where :math:`r` is the distance, as well as a spline method that applies to spherical geometry
following the approach of [Wahba_Sphere]_.

The `Spline` classes all have the following methods:

- `fit` : To fit the spline to the training data (used to initialise the class instance). A
  smoothing parameter can be supplied as the `lam` argument, or it can be estimated using GCV
  minimisation.
- `predict` : To predict at new positions (or at the training positions) using the result from
  fitting the spline. The standard error can also be returned.
- `standard_error` : To compute the standard error of the prediction (following [Nychka]_).

On using the `fit` method with `set_results = True` a `SplineResult` class is set to the `Spline` as
the `result` attribute. This contains details about the fit, and optionally statistics about the
fit.

Example Workflow
================

.. code-block:: python

   import numpy as np
   import polars as pl

   from glomar_gridding.spline import SphericalThinPlateSpline

   # Load observations (latitude, longitude, tos)
   obs = pl.read_csv("/path/to/obs.csv", ...)

   # Get the training data from the observations
   X_train = obs.select(["latitude", "longitude"]).to_numpy()
   y_train = obs.get_column("tos").to_numpy()

   # Define output grid (testing data) - in this example it
   # is a 5 degree global (un-masked) grid
   out_grid = np.mgrid[-87.5:90:5, -177.5:180:5]
   X_test = out_grid.reshape(2, -1).T

   # Initialise the SphericalThinPlateSpline class with the
   # training data
   stps = SphericalThinPlateSpline(X_train, y_train)

   # Fit with lambda estimation using GCV minimisation
   stps.fit(set_results=True, compute_stats=True)

   # Predict at new points and get the standard_error
   y_pred, std_stps = stps.predict(X_test)

Spline
======

This is the abstract class form of the `Spline`, and should not be used directly and will raise an
error if used. Instead one should use either the `ThinPlateSpline` or `SphericalThinPlateSpline`
classes.

The documentation for this abstract class is included as it includes the documentation for several
shared components between the `Spline` classes.

.. autoclass:: glomar_gridding.spline.Spline
   :members:

Thin-Plate-Splines
==================

.. autoclass:: glomar_gridding.spline.ThinPlateSpline
   :members:

Spherical Thin-Plate-Splines
============================

.. autoclass:: glomar_gridding.spline.SphericalThinPlateSpline
   :members:

Spline Result
=============

.. autoclass:: glomar_gridding.spline.SplineResult
   :members:
