Spline Interpolation
--------------------

The `spline` module brings an alternative approach to interpolation using thin-plate-spline
methodology to perform interpolation. We provide a standard thin-plate-spline interpolator using
the traditional thin-plate-spline kernel:

.. math::
   r^2 \times log(r)

where :math:`r` is the distance, as well as a spline method that applies to spherical geometry
following the approach of [Wahba_Sphere]_.

Thin-Plate-Splines
==================

.. autoclass:: glomar_gridding.spline.ThinPlateSpline
   :members:

Spherical Thin-Plate-Splines
============================

.. autoclass:: glomar_gridding.spline.SphericalThinPlateSpline
   :members:
