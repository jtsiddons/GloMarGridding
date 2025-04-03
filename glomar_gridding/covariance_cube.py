"""
Requires numpy, scipy, sklearm
iris needs to be installed (it is required by other modules within this package
xarray cubes should work via iris interface)
"""

import math
from typing import Any

import xarray as xr
import numpy as np

from sklearn.metrics.pairwise import haversine_distances

# from astropy.constants import R_earth
from glomar_gridding.constants import (
    DEFAULT_N_JOBS,
    RADIUS_OF_EARTH_KM,
)
from glomar_gridding.ellipse import EllipseModel
from glomar_gridding.distances import displacements
from glomar_gridding.types import DELTA_X_METHOD
from glomar_gridding.utils import cov_2_cor, mask_array


class EllipseBuilder:
    """
    Class to build spatial covariance and correlation matricies
    Interacts with MaternEllipseModel to build covariance models

    Parameters
    ----------
    data_cube : instance of iris.cube.Cube
        Training data stored within iris cube

        Some modification probably needed to work with instances
        of xa.dataarray
        The biggest hurdle is that iris cube handle masked data differently
        than xarray

        While both iris.cube.Cube.data and xa.DataArray.data are instances
        to numpy.ndarray...

        A masked array under iris.cube.Cube.data is always an instance of
        np.ma.MaskedArray
        In xarray, xa.DataArray.data is always unmasked.
    """

    def __init__(
        self,
        data_array: xr.DataArray,
    ) -> None:
        # Defining the input data
        self.data = mask_array(data_array.values)
        self.coords = data_array.coords
        self.xy_shape = self.data[0].shape
        if len(self.xy_shape) != 2:
            raise ValueError(
                "Time slice maps should be 2D; check extra dims (ensemble?)"
            )
        self.big_covar_size = np.prod(self.xy_shape)

        self._parse_coords()
        self._detect_mask()

        # Calculate the actual covariance and correlation matrix:
        self.calc_cov()

        return None

    def _parse_coords(self) -> None:
        # Check input data_cube is actually usable
        self.tcoord_pos = -1
        self.xycoords_pos = []
        self.xycoords_name = []
        for i, coord in enumerate(self.coords):
            if coord == "time":
                self.tcoord_pos = i
            if coord in ["latitude", "longitude"]:
                self.xycoords_pos.append(i)
                self.xycoords_name.append(coord)
        if self.tcoord_pos == -1:
            raise ValueError("Input cube needs a time dimension")
        if self.tcoord_pos != 0:
            raise ValueError("Input cube time dimension not at 0")
        if len(self.xycoords_pos) != 2:
            raise ValueError(
                "Input cube need two spatial dimensions "
                + "('latitude' and 'longitude')"
            )
        self.xycoords_pos = tuple(self.xycoords_pos)

        # Look-up table of the coordinates
        self.xx, self.yy = np.meshgrid(
            self.coords["longitude"].points,
            self.coords["latitude"].points,
        )
        self.xi, self.yi = np.meshgrid(
            np.arange(self.coords["longitude"].points),
            np.arange(self.coords["latitude"].points),
        )
        # Length of time dimension
        self.time_n = len(self.coords["time"].points)

        return None

    def _detect_mask(self) -> None:
        # Detect data mask and determine dimension of array without masked data
        # Almost certain True near the coast
        self.data_has_mask = np.ma.is_masked(self.data)
        if self.data_has_mask:
            # Depending on dataset, the mask might not be invariant
            # (like sea ice)
            # xys with time varying mask are currently discarded.
            # If analysis is conducted seasonally
            # this should normally not a problem unless in high latitudes
            self.mask = np.any(self.data.mask, axis=0)
            self.mask_1D = self.mask.flatten()
            self._self_mask()
            self.small_covar_size = np.sum(np.logical_not(self.mask))
        else:
            self.mask = np.zeros_like(self.data[0], dtype=bool)
            self.mask_1D = self.mask.flatten()
            self.small_covar_size = self.big_covar_size
        self.x_masked = np.ma.masked_where(self.mask, self.xx)
        self.y_masked = np.ma.masked_where(self.mask, self.yy)
        self.xi_masked = np.ma.masked_where(self.mask, self.xi)
        self.yi_masked = np.ma.masked_where(self.mask, self.yi)
        self.xy_masked = np.column_stack(
            [self.x_masked.compressed(), self.y_masked.compressed()]
        )
        self.xy_full = np.column_stack(
            [self.x_masked.flatten(), self.y_masked.flatten()]
        )

        return None

    def calc_cov(
        self,
        rounding: int | None = None,
    ) -> None:
        """
        Calculate covariance and correlation matrices.

        Parameters
        ----------
        rounding : int
            round the values of the output
        """
        # Reshape data to (t, xy),
        # get rid of mask values -- cannot caculate cov for such data
        xyflatten_data = self.data.reshape((self.time_n, self.big_covar_size))
        xyflatten_data = np.ma.compress_rowcols(xyflatten_data, -1)
        # Remove mean --
        # even data that says "SST anomalies" don't have zero mean (?!)
        xy_mean = np.mean(xyflatten_data, axis=0, keepdims=True)
        xyflatten_data = xyflatten_data - xy_mean
        self.cov = np.matmul(np.transpose(xyflatten_data), xyflatten_data)
        # xy_cov = _AT_A(xyflatten_data)

        if rounding is not None:
            self.cov = np.round(self.cov, rounding)

        self.cor = cov_2_cor(self.cov, rounding=rounding)

        return None

    def _self_mask(self) -> None:
        """Broadcast cube_mask to all observations"""
        broadcasted_mask = np.broadcast_to(self.mask, self.data.shape)
        self.data = np.ma.masked_where(broadcasted_mask, self.data)

    def _reverse_mask_from_compress_1d(
        self,
        compressed_1D_vector: np.ndarray,
        fill_value: float = 0.0,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        Since there are lot of flatten and compressing going on for observations
        and fitted parameters, this reverses the 1D array to the original 2D map

        DANGER WARNING, use different fill_value depending on situation
        This affects how signal and image processing module interacts
        with missing and masked values.
        They don't ignore them, so a fill_value like 0 may be sensible
        for covariance (-999.99 will do funny things if it finds its way to a
        convolution and filter) iris doesn't care, but should use something like
        -999.99 or something.

        Parameters
        ----------
        compressed_1D_vector : np.ndarray (1D) shape = (NUM,)
            1D vector that is a compressed/flatten version of a 2D map
        fill_value: float
            Fill value for masked point
        dtype: valid numpy float type
            The dtype for 2D array.

        Returns
        -------
        ans : np.ndarray (2D) shape = (NUM,NUM)
            The 2D map array
        """
        compressed_counter = 0
        ans = np.zeros_like(self.mask_1D, dtype=dtype)
        for i in range(len(self.mask_1D)):
            if not self.mask_1D[i]:
                ans[i] = compressed_1D_vector[compressed_counter]
                compressed_counter += 1
        np.ma.set_fill_value(ans, fill_value)
        ans = np.ma.masked_where(self.mask_1D, ans)
        return ans

    def fit_ellipse_model(
        self,
        xy_point: int,
        matern_ellipse: EllipseModel,
        max_distance: float = 20.0,
        min_distance: float = 0.3,
        delta_x_method: DELTA_X_METHOD | None = "Modified_Met_Office",
        guesses: list[float] | None = None,
        bounds: list[tuple[float, ...]] | None = None,
        opt_method: str = "Nelder-Mead",
        tol: float = 0.001,
        estimate_SE: str | None = None,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> dict[str, Any]:
        """
        Fit ellipses/covariance models using adhoc local covariances

        the form of the covariance model depends on the "fform" attribute of the
        Ellipse model:
            isotropic (radial distance only)
            anistropic (x and y are different, but not rotated)
            anistropic_rotated (rotated)

        If the "fform" attribued ends with _pd then phyiscal distances are used
        instead of degrees

        range is defined max_distance (either in km and degrees)
        default is in degrees, but needs to be km if fform is from _pd series
        <--- likely to be wrong: max_distance should only be in degrees

        there is also a min_distance in which values,
        matern function is not defined at the origin, so the 0.0 needs to
        removed

        v = matern covariance function shape parameter
        Karspeck et al and Paciorek and Schervish use 3 and 4
        but 0.5 and 1.5 are popular
        0.5 gives an exponential decay
        lim v-->inf, Gaussian shape

        delta_x_method: only meaningful for _pd fits
            "Met_Office": Cylindrical Earth delta_x = 6400km x delta_lon
            (in radians)
            "Modified_Met_Office": uses the average zonal dist at different lat

        Parameters
        ----------
        xy_point : int
            The index point where ellipses will be fitted to

        max_distance : float
            Maximum seperation in distance unit that data will be fed
            into parameter fitting
            Units depend on fform (it is usually either degrees or km)

        min_distance: float
            Minimum seperation in distance unit that data
            will be fed into parameter fitting
            Units depend on fform (it is usually either degrees or km)
            Note: Due to the way we compute the Matern function,
            it is undefined at dist == 0 even if the limit -> zero is obvious.

        delta_x_method="Modified_Met_Office": str
            How to compute distances between grid points
            For istropic variogram/covariances, this is a trival problem;
            you can just take the haversine or
            Euclidian ("tunnel") distance as they are non-directional.

            But it is non trival for anistropic cases,
            you have to define a set of orthogonal space. In HadSST4,
            Earth is assumed to be cyclindrical "tin can" Earth,
            so you can just define the orthogonal space by
            lines of constant lat and lon (delta_x_method="Met_Office").

            The modified "Modified_Met_Office" is a variation to that,
            but allow the tin can get squished at the poles.
            (Sinusoidal projection). This does results in a problem:
            the zonal displacement now depends in which latitude
            you compute on (at the beginning latitude or at the end latitude).
            Here we take the average of the two.

        guesses=None: tuple of floats; None uses default guess values
            Initial guess values that get feeds in the optimizer for MLE.
            In scipy, you are required to do so (but R often doesn't).
            You should anyway; sometimes they do funny things
            if you don't (per recommendation of David Stephenson)

        bounds=None: tuple of floats; None uses default bounds values
            This is essentially a Bayesian "uniformative prior"
            that forces convergence if the optimizer hits the bound.
            For lower resolution fitting, this is rarely a problem.
            For higher resolution fits, this often interacts with
            the limit of the data you can put into the fit the optimizer
            may fail to converge if the input data is very smooth (aka ENSO
            region, where anomalies are smooth over very large (~10000km)
            scales).

        opt_method='Nelder-Mead': str
            scipy optimizer method. Nelder-Mead is the one used by Karspeck.
            See https://docs.scipy.org/doc/scipy/tutorial/optimize.html
            for valid options

        tol=0.001: float
            set convergence tolerance for scipy optimize.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

            Note on new tol kwarg:
            For N-M, this sets the value to both xatol and fatol
            Default is 1E-4 (?)
            Since it affects accuracy of all values including rotation
            rotation angle 0.001 rad ~ 0.05 deg

        estimate_SE=None : str | None
            The code can estimate the standard error if the Matern parameters.
            This is not usually used or discussed for the purpose of kriging.
            Certain opt_method (gradient descent) can do this automatically
            using Fisher Info for certain covariance funciton,
            but is not possible for some nasty functions (aka Bessel
            func) gets involved nor it is possible for some optimisers
            (such as Nelder-Mead).
            The code does it using bootstrapping.

        n_jobs=DEFAULT_N_JOBS: int
            If parallel processing, number of threads to use.

        Returns
        -------
        ans : dictionary
            Dictionary with results of the fit
            and the observed correlation matrix
        """
        R2 = self.data[0].copy()
        R2x = self._reverse_mask_from_compress_1d(self.cor[xy_point, :])
        R2.data = R2x.reshape(self.xy_shape)
        R2.units = "1"

        X_train, y_train = self._get_train_data(
            xy_point=xy_point,
            min_distance=min_distance,
            max_distance=max_distance,
            anisotropic=matern_ellipse.anisotropic,
            delta_x_method=delta_x_method,
        )

        results, _, bounds = matern_ellipse.fit(
            X_train,
            y_train,
            guesses=guesses,
            bounds=bounds,
            opt_method=opt_method,
            tol=tol,
            estimate_SE=estimate_SE,
            n_jobs=n_jobs,
        )

        model_params = results.x.tolist()
        stdev = None
        if not matern_ellipse.unit_sigma:
            stdev = model_params.pop()

        # Meaning of fit_success (int)
        # 0: success
        # 1: success but with one parameter reaching lower bounadries
        # 2: success but with one parameter reaching upper bounadries
        # 3: success with multiple parameters reaching the boundaries
        #    (aka both Lx and Ly), can be both at lower or upper boundaries
        # 9: fail, probably due to running out of maxiter
        #    (see scipy.optimize.minimize kwargs "options)"
        if results.success:
            fit_success = _get_fit_score(model_params, bounds, results.nit)
            print("RMSE of multivariate norm fit = ", stdev)
        else:
            print("Convergence fail after ", results.nit, " iterations.")
            print(model_params)
            fit_success = 9
        print("QC flag = ", fit_success)

        return {
            "Correlation": R2,
            "Results": results,
            "ModelParams": model_params,
            "Success": fit_success,
            "StandardDeviation": np.sqrt(
                self.cov[xy_point, xy_point] / self.time_n
            ),
        }

    def _get_train_data(
        self,
        xy_point: int,
        min_distance: float,
        max_distance: float,
        anisotropic: bool,
        delta_x_method: DELTA_X_METHOD | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lonlat = self.xy_masked[xy_point]
        y = self.cor[xy_point, :]
        disp_y, disp_x = displacements(
            self.xy_masked[:, 1],
            self.xy_masked[:, 0],
            lonlat[1],
            lonlat[0],
            delta_x_method=delta_x_method,
        )
        if delta_x_method is None:
            # disp_y and disp_x are in degrees
            deg_distance = np.linalg.norm(
                np.column_stack([disp_x, disp_y]), axis=1
            )
            valid_dist_idx = np.where(
                (deg_distance <= max_distance)
                & (deg_distance >= min_distance)
                # Delete the origin (can't have dx = dy = 0)
                & (deg_distance != 0)
            )
            y_train = y[valid_dist_idx]
            if anisotropic:
                X_train = np.column_stack([disp_x, disp_y])[valid_dist_idx, :]
                return X_train, y_train
            return deg_distance[valid_dist_idx], y_train

        # disp_y and disp_x are in radians
        latlons = np.radians(
            np.column_stack([self.xy_masked[:, 1], self.xy_masked[:, 0]])
        )
        latlon = np.radians(np.array([lonlat[1], lonlat[0]])).reshape(1, -1)
        distance = haversine_distances(latlon, latlons)[0] * RADIUS_OF_EARTH_KM
        valid_dist_idx = np.where(
            (distance <= max_distance)
            & (distance >= min_distance)
            # Delete the origin (can't have dx = dy = 0)
            & (distance != 0)
        )
        y_train = y[valid_dist_idx]
        if anisotropic:
            X_train = np.column_stack([disp_x, disp_y])[valid_dist_idx, :]
            return RADIUS_OF_EARTH_KM * X_train, y_train
        return distance[valid_dist_idx], y_train

    def find_nearest_xy_index_in_cov_matrix(
        self,
        lonlat: list[float],
        use_full: bool = False,
    ) -> tuple[int, np.ndarray]:
        """
        Find the nearest column/row index of the covariance
        that corresponds to a specific lat lon
        """
        lon, lat, *_ = lonlat

        a = self.xy_full if use_full else self.xy_masked
        idx = int(((a[:, 0] - lon) ** 2.0 + (a[:, 1] - lat) ** 2.0).argmin())
        return idx, a[idx, :]

    def _xy_2_xy_full_index(self, xy_point: int) -> int:
        """
        Given xy index in that corresponding to a latlon
        in the covariance (masked value np.ma.MaskedArray compressed),
        what is its index with masked values (i.e. ndarray flatten)
        """
        return int(
            np.argwhere(
                np.all(
                    (self.xy_full - self.xy_masked[xy_point, :]) == 0, axis=1
                )
            )[0]
        )

    def __str__(self):
        return str(self.__class__)
        # return str(self.__class__) + ": " + str(self.__dict__)


def _get_fit_score(model_params, bounds, niter) -> int:
    fit_success: int = 0
    for model_param, bb in zip(model_params, bounds):
        left_check = math.isclose(model_param, bb[0], rel_tol=0.01)
        right_check = math.isclose(model_param, bb[1], rel_tol=0.01)
        left_advisory = "near_left_bnd" if left_check else "not_near_left_bnd"
        right_advisory = "near_right_bnd" if right_check else "not_near_rgt_bnd"
        print(
            "Convergence success after ",
            niter,
            " iterations: ",
            model_param,
            bb[0],
            bb[1],
            left_advisory,
            right_advisory,
        )
        if left_check:
            fit_success = 1 if fit_success == 0 else 3
        if right_check:
            fit_success = 2 if fit_success == 0 else 3
    return fit_success
