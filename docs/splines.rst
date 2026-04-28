Spline Interpolation
--------------------

The `spline` module brings an alternative approach to interpolation using thin-plate-spline
methodology to perform interpolation. We provide a standard thin-plate-spline interpolator using
the traditional kernel:

.. math::
   r^2 * log(r)

where :math:`r` is the distance, as well as a spline method that applies to spherical geometry
following the approach of [Wahba_Sphere]_.

Further, spline-based approaches to Kriging are available. This allows for use of position as input
to the interpolation, removing the need to align observations to a grid.

Thin-Plate-Splines
==================

.. autoclass:: glomar_gridding.spline.ThinPlateSpline
   :members:

Spherical Thin-Plate-Splines
============================

.. autoclass:: glomar_gridding.spline.SphericalThinPlateSpline
   :members:

Spline Based Kriging
====================

Ordinary Kriging
^^^^^^^^^^^^^^^^

.. autoclass:: glomar_gridding.spline.OrdinaryKriging
   :members:

Universal Kriging
^^^^^^^^^^^^^^^^^

.. autoclass:: glomar_gridding.spline.UniversalKriging
   :members:

Variogram Kriging
^^^^^^^^^^^^^^^^^

.. autoclass:: glomar_gridding.spline.VariogramKriging
   :members:
