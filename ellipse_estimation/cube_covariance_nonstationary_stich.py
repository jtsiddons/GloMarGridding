"""
Requires numpy, scipy, sklearn
iris needs to be installed (it is required by other modules within this package
xarray cubes should work via iris interface)
"""

import datetime

import logging
import numbers
import sys
import tracemalloc

from joblib import Parallel, delayed  # Developmental
from itertools import product, combinations

# from iris import analysis as ia
import iris
import numpy as np
from numpy import ma
from numpy import linalg
from scipy.special import kv as modified_bessel_2nd
from scipy.special import gamma
from statsmodels.stats import correlation_tools

from glomar_gridding.distances import (
    sigma_rot_func,
    mahal_dist_func,
    tau_dist,
)
from glomar_gridding.constants import DEFAULT_N_JOBS, DEFAULT_BACKEND
from ellipse_estimation.distance_util import scalar_cube_great_circle_distance
from glomar_gridding.types import DELTA_X_METHOD
from glomar_gridding.utils import uncompress_masked

# Below is in theory redudant, but the view/controller bits of the code
# has not been integrated to the package; for now, keeping this in case
# of breaking other code

# _MAX_DEG_Kar = 20.0  # Karspeck et al distance threshold in degrees latlon
# _MAX_DIST_Kar = cube_cov._deg2km(_MAX_DEG_Kar)  # to km @ lat = 0.0 (2222km)

# _MAX_DIST_UKMO = 10000.0  # UKMO uses 10000km range to fit the non-rot ellipse
# _MAX_DEG_UKMO = cube_cov._km2deg(_MAX_DIST_UKMO)

MAX_DIST_COMPROMISE: float = 6000.0  # Compromise _MAX_DIST_Kar &_MAX_DIST_UKMO
# _MAX_DEG_compromise = cube_cov._km2deg(_MAX_DIST_compromise)

# _MIN_CORR_Threshold = 0.5 / np.e


def mask_cube(cube: iris.cube.Cube) -> iris.cube.Cube:
    """
    Forces cube.data to be an instance of np.ma.MaskedArray

    Parameters
    ----------
    cube : iris.cube.Cube
        Can be masked or not masked

    Returns
    -------
    cube : iris.cube.Cube
        'data' attribute within the cube is now an instance of np.ma.MaskedArray
    """
    if isinstance(cube.data, np.ma.MaskedArray):
        return cube
    if isinstance(cube.data, np.ndarray):
        logging.info("Ad hoc conversion to np.ma.MaskedArray")
        cube.data = np.ma.MaskedArray(cube.data)
        return cube
    raise TypeError("Input cube is not a numpy array.")


def sizeof_fmt(num: float, suffix="B") -> str:
    """
    Convert numbers to kilo/mega... bytes,
    for interactive printing of code progress
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def c_ij_anistropic_rotated_nonstationary(
    v: float,
    sdev_i: float,
    sdev_j: float,
    x_i: np.ndarray,
    x_j: np.ndarray,
    sigma_parms_i: list[float],
    sigma_parms_j: list[float],
    decompose: bool = False,
) -> float:
    """
    Compute the nonstationary spatially-varying covariance between
    point i and j with covariance model parameters sigma_parms_i, sigma_parms_j
    and local standard deviations of sdev_i, sdev_j

    x_i = zonal displacement (NOT COORDINATES OF point i)
    x_j = meridonal displacement (NOT COORDINATES OF point j)
    Use scalar_cube_great_circle_distance to compute displacement and distance

    ans of scalar_cube_great_circle_distance: x_i = ans[2] and x_j = ans[1]
    (ans[0] is the great circle dist)

    sigma_parms_i = (Lx_i, Ly_i, theta_i)
    sigma_parms_j = (Lx_j, Ly_j, theta_j)

    original equation:
    1) Paciorek and Schevrish 2006 Equation 8 https://doi.org/10.1002/env.785
    2) Karspeck et al 2012 Equation 17 https://doi.org/10.1002/qj.900

    Parameters
    ----------
    v : float
        Matern shape parameter
    sdev_i, sdev_j: float
        Standard deviations of the two points
    x_i, x_j: float
        Components of the vector displacement between the two points
    sigma_parms_i, sigma_parms_j: iterable
        iterable with a length of 3 each, that states Lx, Ly, theta
        Dimensions of Lx, Ly, x_i and x_j must be the same (i.e. km with km
        not km with degrees)
    decompose: bool
        Optionally decompose sigma_bar.

    Returns
    -------
    c_ij : float
        the covariance between the two points
    """
    # Compute sigma_bar
    sigma_i = sigma_rot_func(
        sigma_parms_i[0], sigma_parms_i[1], sigma_parms_i[2]
    )
    sigma_j = sigma_rot_func(
        sigma_parms_j[0], sigma_parms_j[1], sigma_parms_j[2]
    )
    sigma_bar = 0.5 * (sigma_i + sigma_j)

    logging.debug(f"{sigma_bar = }")

    # sigma_bar can be broken down to new sigma parameters
    # aka a new Lx, Ly and theta using eigenvalue decomposition
    # Sigma_bar = R(theta_bar) @ [[Lx_bar**2 0][0 Ly_bar**2]] @ R(theta_bar)^-1
    # In which eigenvalues to Sigma Bar forms
    # the diagonal matrix of Lx_bar Ly_bar
    # and eigenvectors are rotation matrix R(theta_bar)
    # if sigma_bar is nearly circle,
    # there are a possibility of floating point issue
    # i.e. eigenvalues and eigenvectors become "complex"
    # when off-diagonal are a very small float (1E-10)

    # This gets annoying if running istropic unit tests...
    # print('Adjustments made to small off diagonal terms of sigma_bar')
    sigma_bar[np.isclose(sigma_bar, 0.0)] = 0.0

    if decompose:
        # Decomposing sigma_bar
        # sigma_bar = R_bar x [( Lx_bar**2 0 ) (0 Ly_bar**2)] x R_bar_transpose
        sigma_bar_eigval, sigma_bar_eigvec = linalg.eig(sigma_bar)

        # If numerical instability is detected, resulting in complex number,
        # take the real part
        if np.any(np.iscomplex(sigma_bar_eigval)) or np.any(
            np.iscomplex(sigma_bar_eigvec)
        ):
            logging.warning("Complex eigenvalues detected!")
            logging.warning(f"{sigma_bar_eigval = }")
            logging.warning(f"{sigma_bar_eigvec = }")
            sigma_bar_eigval = sigma_bar_eigval.real
            sigma_bar_eigvec = sigma_bar_eigvec.real
        # Actual Lx, Ly are square roots of the eigenvalues
        Lx_bar = np.sqrt(sigma_bar_eigval[0])
        Ly_bar = np.sqrt(sigma_bar_eigval[1])

        # https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix

        logging.debug(f"Eigvals of sigma_bar = {sigma_bar_eigval}")
        logging.debug(f"{(Lx_bar, Ly_bar) = }")
        logging.debug(f"Eigvec of sigma_bar  = {sigma_bar_eigvec}")
        # Use arctan2 to compute rotation angle '''
        theta_bar = np.arctan2(sigma_bar_eigvec[1, 0], sigma_bar_eigvec[0, 0])
        # Below should show the same angle in radians
        v_ans0 = (
            np.arccos(sigma_bar_eigvec[0, 0]),
            np.rad2deg(np.arccos(sigma_bar_eigvec[0, 0])),
        )
        v_ans1 = (
            np.arcsin(sigma_bar_eigvec[0, 1]),
            np.rad2deg(np.arcsin(sigma_bar_eigvec[0, 1])),
        )
        v_ans2 = (
            -np.arcsin(sigma_bar_eigvec[1, 0]),
            np.rad2deg(-np.arcsin(sigma_bar_eigvec[1, 0])),
        )
        v_ans3 = (theta_bar, np.rad2deg(theta_bar))
        logging.debug(
            f"arccos  R[0,0][1, 1] = {v_ans0} (>0 for ang within +/- pi/2)"
        )
        logging.debug(f"arcsin  R[0,1]       = {v_ans1}")
        logging.debug(f"-arccos R[1,0]       = {v_ans2}")
        logging.debug(f"theta_bar            = {v_ans3} (using arctan2)")
        tau_bar = mahal_dist_func(x_i, x_j, Lx_bar, Ly_bar, theta=theta_bar)
    else:
        # Direct computation without decomposition (faster)
        # This is direct use of right part of Equation 18 in Karspeck et al 2012
        # This is behind else, so one won't be computing stuff twice
        tau_bar = tau_dist(x_i, x_j, sigma_bar)
    logging.debug(f"{(x_i, x_j) = }")
    logging.debug(f"{tau_bar = }")

    # Eq 17 in Karspeck et al 2012
    # ans = first_term x second_term x third_term x fourth_term '''
    first_term = (sdev_i * sdev_j) / (gamma(v) * (2.0 ** (v - 1)))

    # second_term_u = root4(linalg.det(sigma_i))*root4(linalg.det(sigma_j))
    # second_term_d = np.sqrt(linalg.det(sigma_bar))
    second_term_num = root4(det22(sigma_i)) * root4(det22(sigma_j))
    second_term_denom = np.sqrt(det22(sigma_bar))
    second_term = second_term_num / second_term_denom
    third_term = (2.0 * tau_bar * np.sqrt(v)) ** v
    forth_term = modified_bessel_2nd(v, 2.0 * tau_bar * np.sqrt(v))

    logging.debug(f"Check: {first_term = }")
    logging.debug(f"Check: {second_term = }")
    logging.debug(f"Check: {third_term = }")
    logging.debug(f"Check: {forth_term = }")

    c_ij = first_term * second_term * third_term * forth_term
    logging.debug(f"Check: sdev_i * sdev_j = {sdev_i * sdev_j}")
    logging.debug(f"Check: ans (cov)       = {c_ij}")
    logging.debug(f"Check: ans (cor)       = {c_ij / (sdev_i * sdev_j)}")
    # Don't know why I added the below, I have never seen it triggered...
    # in which you are not supposed to as it will indicate bug elsewhere
    # the original check is much more
    if c_ij > (sdev_i * sdev_j):
        raise ValueError("sdev_i * sdev_j should always be smaller than ans")
    return c_ij


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """Helper function for perturb_sym_matrix_2_positive_definite"""
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def perturb_sym_matrix_2_positive_definite(
    square_sym_matrix: np.ndarray,
) -> np.ndarray:
    """
    On the fly eigenvalue clipping, this is based statsmodels code
    statsmodels.stats.correlation_tools.cov_nearest
    statsmodels.stats.correlation_tools.corr_nearest

    Use repair_damaged_covariance instead, it is more complete

    Other methods exist:
    https://nhigham.com/2021/02/16/diagonally-perturbing-a-symmetric-matrix-to-make-it-positive-definite/
    https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/
    https://academic.oup.com/imajna/article/22/3/329/708688
    """
    matrix_dim = square_sym_matrix.shape
    if (
        (len(matrix_dim) != 2)
        or (matrix_dim[0] != matrix_dim[1])
        or not check_symmetric(square_sym_matrix)
    ):
        raise ValueError("Matrix is not square and/or symmetric.")

    eigenvalues = linalg.eigvalsh(square_sym_matrix)
    min_eigen = np.min(eigenvalues)
    max_eigen = np.max(eigenvalues)
    n_negatives = np.sum(eigenvalues < 0.0)
    print("Number of eigenvalues = ", len(eigenvalues))
    print("Number of negative eigenvalues = ", n_negatives)
    print("Largest eigenvalue  = ", max_eigen)
    print("Smallest eigenvalue = ", min_eigen)
    if min_eigen >= 0.0:
        print("Matrix is already positive (semi-)definite.")
        return square_sym_matrix
    ans = correlation_tools.cov_nearest(square_sym_matrix, return_all=False)
    if not isinstance(ans, np.ndarray):
        raise TypeError(
            "Output of correlation_tools.cov_nearest is not a numpy array"
        )

    eigenvalues_adj = linalg.eigvalsh(ans)
    min_eigen_adj = np.min(eigenvalues_adj)
    max_eigen_adj = np.max(eigenvalues_adj)
    n_negatives_adj = np.sum(eigenvalues_adj < 0.0)
    print("Post adjustments:")
    print("Number of negative eigenvalues (post_adj) = ", n_negatives_adj)
    print("Largest eigenvalue (post_adj)  = ", max_eigen_adj)
    print("Smallest eigenvalue (post_adj) = ", min_eigen_adj)
    return ans


# def seaice_anti_hubris_field(cube, land_mask, ice_fill_value=None):
#     '''
#     We have other ways to fill in data gaps now,
#     like using HadCRUT5 parameters...
#     but this function can be resurrected in the future
#
#     assuming sea == 1 and land == 0 in land_mask
#     replace where xor(land_mask, original_cube) (aka masked sea points)
#     with ice_fill value
#     default for ice_fill_value are applicable
#
#     WARNING:
#     Inserting same values over multiple rows and columns
#     lead to degenerate matrices!!!
#
#     Possible solution:
#     Instead of simple infilling, add a random pertubation on top
#     '''
#     cube2 = cube.copy()
#     cube2.data.mask = False
#     cube2.data = np.ma.masked_where(land_mask.data < 0.95, cube2.data)
#     where_are_the_ice = np.logical_xor(cube.data.mask, cube2.data.mask)
#     if ice_fill_value is None:
#         if cube.units == 'km':
#             ice_fill_value = 100.0
#         elif cube.units == 'radians':
#             ice_fill_value = 0.0
#         else:
#             err_msg = 'ice_fill_value not provided, no defaults for cube units'  # noqa: E501
#             raise ValueError(err_msg)
#     ##
#     if isinstance(ice_fill_value, numbers.Number):
#         print('Replacing all xor(land_mask, cube) points with ', ice_fill_value)  # noqa: E501
#         cube2.data[where_are_the_ice] = ice_fill_value
#         return cube2
#     elif np.ndim(ice_fill_value) != 0:
#         ''' This allow a fillin by user-provided vector/list, length needs to match '''  # noqa: E501
#         ice_fill_value2 = np.array(ice_fill_value)
#         ice_fill_value2 = ice_fill_value2[:, np.newaxis]
#         where_are_the_ice2 = where_are_the_ice.astype(float)
#         fill_in_matrix = np.multiply(ice_fill_value2, where_are_the_ice2)
#         cube2.data[where_are_the_ice] = 0.0
#         cube2.data = cube2.data + fill_in_matrix
#         print('Replacing all xor(land_mask, cube) points with ',ice_fill_value[0],' ... ', ice_fill_value[-1])  # noqa: E501
#         return cube2
#     elif isinstance(ice_fill_value, str):
#         '''
#         Variations of mean sub in imputation
#         for distances imputation by minimum (anti-hubris value)
#         for angles, "minimum" approach would be unrealistic, so we can use median (or angular mean, but that isn't implemented in iris)  # noqa: E501
#         '''
#         if ice_fill_value == 'zonal_min_substitution':
#             zonal_mean = cube.collapsed('longitude', ia.MIN)
#             zonal_mean_val = zonal_mean.data
#             zonal_mean_val = zonal_mean_val[:, np.newaxis]
#             cube2.data[where_are_the_ice] = 0.0
#             fill_in_matrix = np.multiply(zonal_mean_val, where_are_the_ice.astype(float))  # noqa: E501
#             cube2.data = cube2.data + fill_in_matrix
#             print('Replacing all xor(land_mask, cube) points with ',zonal_mean.data[0],' ... ', zonal_mean.data[-1])  # noqa: E501
#             return cube2
#         elif ice_fill_value == 'zonal_median_substitution':
#             zonal_mean = cube.collapsed('longitude', ia.MEDIAN)
#             zonal_mean_val = zonal_mean.data
#             zonal_mean_val = zonal_mean_val[:, np.newaxis]
#             cube2.data[where_are_the_ice] = 0.0
#             fill_in_matrix = np.multiply(zonal_mean_val, where_are_the_ice.astype(float))  # noqa: E501
#             cube2.data = cube2.data + fill_in_matrix
#             print('Replacing all xor(land_mask, cube) points with ', zonal_mean.data[0], ' ... ', zonal_mean.data[-1])  # noqa: E501
#             return cube2
#         elif ice_fill_value == 'zonal_mean_substitution':
#             zonal_mean = cube.collapsed('longitude', ia.MEAN)
#             zonal_mean_val = zonal_mean.data
#             zonal_mean_val = zonal_mean_val[:, np.newaxis]
#             cube2.data[where_are_the_ice] = 0.0
#             fill_in_matrix = np.multiply(zonal_mean_val,
#                                          where_are_the_ice.astype(float))
#             cube2.data = cube2.data + fill_in_matrix
#             print('Replacing all xor(land_mask, cube) points with ',
#                   zonal_mean.data[0],
#                   ' ... ',
#                   zonal_mean.data[-1])
#             return cube2
#         else:
#             raise ValueError('Unknown string input for ice_fill_value')
#     else:
#         raise ValueError('Unknown input for ice_fill_value')


class CovarianceCube_PreStichedLocalEstimates:
    """
    The class that takes multiple iris cubes of
    non-stationary variogram parameters to build
    and save covariance matrices

    v = Matern covariance shape parameter

    As not shown on TV (Karspeck et al)...
    Lx - an iris (or xarray2iris-converted?) cube of horizontal length scales (
    Ly - an iris (or xarray2iris-converted?) cube of meridonal length scales
    theta - an iris cube of rotation angles (RADIANS ONLY)

    sdev - standard deviation -- right now it just takes a number cube
    if you have multiple contribution to sdev (uncertainities derived from
    different sources), you need to put them into one cube

    Rules:
    Valid (ocean) point:
    1) cov_ns and cor_ns are computed out to max_dist; out of range = 0.0
    2) Masked points are ignored

    Invalid (masked) points:
    1) Skipped over

    max_dist:
    float (km) or (degrees if you want to work in degrees), default 6000km
    if you want infinite distance, just set it to some stupidly large number,
    fun numbers to use:
    1.5E8 (i.e. ~1 astronomical unit (Earth-Sun distance))
    5.0E9 (average distance between Earth and not-a-planet-anymore Pluto)

    Parameters
    ----------
    Lx_cube, Ly_cube, theta_cube, sdev_cube: instances to iris.cube.Cube
        cubes with non-stationary parameters
    v=3: float
        Matern shape parameter
    output_floatprecision :
        Float point precision of the output covariance
        numpy defaults to float64,
        noting that float32 halves the storage and halves the memory to use
    max_dist : float
        If the Haversine distance between 2 points exceed max_dist,
        covariance is set to 0
    degree_dist : bool
        Distances are based on degrees
    delta_x_method : str
        How are displacements computed between points
        The default is the same as in cube_covariance "Modified_Met_Office"

    check_positive_definite : bool
        For production this should be False
        but for unit testing it should be True,
        if True a quick on the fly eigenvalue clipping
        will be conducted, if constructed covariance is not
        positive (semi)definite.

    use_joblib : bool
        Should joblib parallel processing be used

    n_jobs : int
        Number of parallel thread, only matter of use_joblib is
        true. Otherwise numpy will uses its own parallelisation

    backend : str
        backend of joblib

    nolazy : bool
        Manually forces computation to occur

    use_sklearn_haversine: bool
        sklearn has haversine function, but its preformance
        is inconsistent between different machines, and can
        cause a major slow down.

    verbose : bool
           More stdout stuff!
    """

    def __init__(
        self,
        Lx_cube: iris.cube.Cube,
        Ly_cube: iris.cube.Cube,
        theta_cube: iris.cube.Cube,
        sdev_cube: iris.cube.Cube,
        v: float = 3,
        delta_x_method: DELTA_X_METHOD = "Modified_Met_Office",
        max_dist: float = MAX_DIST_COMPROMISE,
        output_floatprecision=np.float64,
        degree_dist: bool = False,
        check_positive_definite: bool = False,
        use_joblib: bool = False,
        nolazy: bool = False,
        use_sklearn_haversine: bool = False,
        verbose: bool = False,
        backend: str = DEFAULT_BACKEND,
        n_jobs: int = DEFAULT_N_JOBS,
    ):
        tracemalloc.start()
        ove_start_time = datetime.datetime.now()
        logging.info(
            "Overhead processing start: ",
            ove_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        n_jobs = n_jobs if use_joblib else 1

        if not isinstance(max_dist, numbers.Number):
            raise ValueError("max_dist must be a number")

        # Defining the input data
        self.v = v  # Matern covariance shape parameter
        self.Lx_local_estimates = mask_cube(Lx_cube)
        self.Ly_local_estimates = mask_cube(Ly_cube)
        self.theta_local_estimates = mask_cube(theta_cube)
        self.sdev_local_estimates = mask_cube(sdev_cube)
        self.max_dist = max_dist
        self.degree_dist = degree_dist
        self.delta_x_method: DELTA_X_METHOD = delta_x_method
        self.check_positive_definite = check_positive_definite
        self.use_sklearn_haversine = use_sklearn_haversine

        # print('Fortran ordering check')
        # for selfcubes in [self.Lx_local_estimates,
        #                   self.Ly_local_estimates,
        #                   self.theta_local_estimates,
        #                   self.sdev_local_estimates]:
        #     print(repr(selfcubes), np.isfortran(selfcubes.data))

        if nolazy:
            for selfcubes in [
                self.Lx_local_estimates,
                self.Ly_local_estimates,
                self.theta_local_estimates,
                self.sdev_local_estimates,
            ]:
                self._no_lazy_data(selfcubes)

        # The cov and corr matrix will be sq matrix of this
        self.xy_shape = self.Lx_local_estimates.shape
        self.n_elements = np.prod(self.xy_shape)

        self._get_mask()

        self._init_dummy_arrays(output_floatprecision)

        ove_end_time = datetime.datetime.now()
        print(
            "Overhead processing ended: ",
            ove_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        print("Time ellipsed: ", ove_end_time - ove_start_time)

        # Fill in the covariance matrix by looping over grid points
        # with different sigma matricies
        ii_index = range(self.covar_size)
        cov_start_time = datetime.datetime.now()
        print(
            "Covariance processing start: ",
            cov_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        for ii in ii_index:
            curr, peak = tracemalloc.get_traced_memory()
            rt_row_info = (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "
            )
            rt_row_info += "Row " + str(ii) + "/" + str(ii_index[-1]) + "; "
            rt_row_info += f"current & peak mem: {curr / (1024 * 1024)} {peak / (1024 * 1024)} MB"  # noqa: E501
            rt_row_info += "; n_jobs = " + str(n_jobs)
            print(rt_row_info)
            self.cov_ns[ii, ii] = (
                self.sdev_local_estimates_compressed[ii] ** 2.0
            )
            self.lat_mat_i[ii, ii] = self.lat_mat_j[ii, ii] = (
                self.lat_grid_compressed[ii]
            )
            self.lon_mat_i[ii, ii] = self.lon_mat_j[ii, ii] = (
                self.lon_grid_compressed[ii]
            )
            jj_index = range(ii + 1, self.covar_size)

            # New -- moved this outside the loop,
            # may make things slightly faster
            # (code will at least look cleaner)
            sdev_i = self.sdev_local_estimates_compressed[ii].copy()
            sigma_parms_i = [
                self.Lx_local_estimates_compressed[ii].copy(),
                self.Ly_local_estimates_compressed[ii].copy(),
                self.theta_local_estimates_compressed[ii].copy(),
            ]
            lat_grid_compressed_i = self.lat_grid_compressed[ii].copy()
            lon_grid_compressed_i = self.lon_grid_compressed[ii].copy()
            logging.debug(
                ii,
                sdev_i,
                sigma_parms_i,
                lat_grid_compressed_i,
                lon_grid_compressed_i,
            )  # For checking use

            # Potential for improvement --
            # it should be possible to make this faster
            # as it is is embarrassingly parallel
            # Nevertheless it doesn't run slow.
            # Estimating the individual local sigma (cube_covariance.py)
            # is much slower

            # interface info: _single_cell_process_parallel(self, ii, jj, verbose)  # noqa: E501
            if not use_joblib:
                # Serial mode (fastest currently)
                for jj in jj_index:
                    if verbose:
                        print(ii, jj)
                    self.lat_mat_i[ii, jj] = self.lat_mat_i[jj, ii] = (
                        lat_grid_compressed_i
                    )
                    self.lon_mat_i[ii, jj] = self.lon_mat_i[jj, ii] = (
                        lon_grid_compressed_i
                    )
                    self.lat_mat_j[ii, jj] = self.lat_mat_j[jj, ii] = (
                        self.lat_grid_compressed[jj]
                    )
                    self.lon_mat_j[ii, jj] = self.lon_mat_j[jj, ii] = (
                        self.lon_grid_compressed[jj]
                    )
                    abs_x, x_j, x_i = scalar_cube_great_circle_distance(
                        lat_grid_compressed_i,
                        lon_grid_compressed_i,
                        self.lat_grid_compressed[jj],
                        self.lon_grid_compressed[jj],
                        degree_dist=self.degree_dist,
                        delta_x_method=self.delta_x_method,
                        use_sklearn_haversine=self.use_sklearn_haversine,
                    )
                    if abs_x > self.max_dist:
                        self.cov_ns[ii, jj] = self.cov_ns[jj, ii] = 0.0
                    else:
                        sdev_j = self.sdev_local_estimates_compressed[jj]
                        sigma_parms_j = [
                            self.Lx_local_estimates_compressed[jj],
                            self.Ly_local_estimates_compressed[jj],
                            self.theta_local_estimates_compressed[jj],
                        ]
                        logging.debug(ii, jj, x_i, x_j)
                        logging.debug(
                            ii,
                            sdev_i,
                            sigma_parms_i,
                            np.rad2deg(sigma_parms_i[-1]),
                        )
                        logging.debug(
                            jj,
                            sdev_j,
                            sigma_parms_j,
                            np.rad2deg(sigma_parms_j[-1]),
                        )
                        # Compute eq 17 in Karspeck et al at each grid point
                        cov_bar = c_ij_anistropic_rotated_nonstationary(
                            v,
                            sdev_i,
                            sdev_j,
                            np.asarray(x_i),
                            np.asarray(x_j),
                            sigma_parms_i,
                            sigma_parms_j,
                            decompose=verbose,
                        )
                        # Fill in symmetric matrix
                        self.cov_ns[ii, jj] = self.cov_ns[jj, ii] = cov_bar
            else:

                def spnip(jj):
                    return self._single_cell_process_notinplace(
                        ii,
                        jj,
                        sdev_i,
                        sigma_parms_i,
                        lat_grid_compressed_i,
                        lon_grid_compressed_i,
                        verbose,
                    )

                parallel_kwargs = {"n_jobs": n_jobs, "backend": backend}
                cov_bars = Parallel(**parallel_kwargs)(
                    delayed(spnip)(jj) for jj in jj_index
                )
                # Fill in symmetric matrix
                logging.debug(
                    len(jj_index),
                    [cov_bars[jjjj] for jjjj in range(10)],
                    cov_bars[-1],
                )  # For checking use
                for j_index, jj in enumerate(jj_index):
                    self.lat_mat_i[ii, jj] = self.lat_mat_i[jj, ii] = (
                        self.lat_grid_compressed[ii]
                    )
                    self.lon_mat_i[ii, jj] = self.lon_mat_i[jj, ii] = (
                        self.lon_grid_compressed[ii]
                    )
                    self.lat_mat_j[ii, jj] = self.lat_mat_j[jj, ii] = (
                        self.lat_grid_compressed[jj]
                    )
                    self.lon_mat_j[ii, jj] = self.lon_mat_j[jj, ii] = (
                        self.lon_grid_compressed[jj]
                    )
                    self.cov_ns[ii, jj] = cov_bars[j_index]
                    self.cov_ns[jj, ii] = cov_bars[j_index]

        cov_end_time = datetime.datetime.now()
        logging.info(
            "Cov processing ended: ", cov_end_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        logging.info("Time ellipsed: ", cov_end_time - cov_start_time)
        logging.info(
            "Mem used by cov mat = ", sizeof_fmt(sys.getsizeof(self.cov_ns))
        )

        # Code now reports eigvals and determinant of the constructed matrix
        self.cov_eig = np.sort(linalg.eigvalsh(self.cov_ns))
        self.cov_det = linalg.det(self.cov_ns)
        if self.check_positive_definite:
            # The purpose of this bit been replaced by
            # repair_damaged_covariance.py
            # in production runs, but is still useful for unit tests
            # Perturb cov matrix to positive semi-definite if needed
            # Tests shows small negative eigval (most neg ~ -0.3 K**2) possible
            logging.info("positive_definite_check is enabled")
            logging.debug("FYI, determinant = ", self.cov_det)
            logging.debug("FYI, eigenvalues sorted (first 10, last 10):")
            logging.debug(self.cov_eig[:10], "...", self.cov_eig[-10:])
            if np.min(self.cov_eig) < 0:
                # On the fly eigenvalue clipping
                logging.warning(
                    "Negative eigval detected; corrections will be applied."
                )
                self.positive_definite_check()
            else:
                logging.debug("Corrections are not needed.")
            logging.info("Positive (semi-)definite checks complete.")
        else:
            logging.info("positive_definite_check not enabled")
            logging.debug("FYI, determinant = ", self.cov_det)
            logging.debug("FYI, eigenvalues sorted (first 10, last 10):")
            logging.debug(self.cov_eig[:10], "...", self.cov_eig[-10:])

        # Compute correlation matrix
        logging.info("Getting reciprocal of covariance diagonal")
        sigma_inverse = np.diag(np.reciprocal(np.sqrt(np.diag(self.cov_ns))))
        logging.info("Computing correlation matrix")
        self.cor_ns = sigma_inverse @ self.cov_ns @ sigma_inverse
        # Check for numerical errors
        print("Checking non-1 values in diagonal of correlation")
        diag_values = np.diag(self.cor_ns)
        where_not_one = diag_values != 1.0
        if np.any(where_not_one):
            largest_weird_value = np.max(np.abs(diag_values[where_not_one]))
            print("Ad hoc fix to numerical issues to corr matrix diag != 1.0")
            print("Largest error = ", largest_weird_value)
            np.fill_diagonal(self.cor_ns, 1.0)

    def _get_mask(self) -> None:
        self.data_has_mask = ma.is_masked(self.Lx_local_estimates.data)
        if self.data_has_mask:
            print("Masked pixels detected in input files")
            self.cube_mask = self.Lx_local_estimates.data.mask
            self.cube_mask_1D = self.cube_mask.flatten()
            self.covar_size = np.sum(np.logical_not(self.cube_mask))
        else:
            print("No masked pixels")
            self.cube_mask = np.zeros_like(
                self.Lx_local_estimates.data.data, dtype=bool
            )
            self.cube_mask_1D = self.cube_mask.flatten()
            self.covar_size = self.n_elements

        print("Compressing (masked) array to 1D")
        self.Lx_local_estimates_compressed = (
            self.Lx_local_estimates.data.compressed()
        )
        self.Ly_local_estimates_compressed = (
            self.Ly_local_estimates.data.compressed()
        )
        self.theta_local_estimates_compressed = (
            self.theta_local_estimates.data.compressed()
        )
        self.sdev_local_estimates_compressed = (
            self.sdev_local_estimates.data.compressed()
        )

        self.xx, self.yy = np.meshgrid(
            self.Lx_local_estimates.coord("longitude").points,
            self.Lx_local_estimates.coord("latitude").points,
        )
        self.xm = np.ma.masked_where(self.cube_mask, self.xx)
        self.ym = np.ma.masked_where(self.cube_mask, self.yy)
        self.lat_grid_compressed = self.ym.compressed()
        self.lon_grid_compressed = self.xm.compressed()
        self.xy = np.column_stack(
            [self.lon_grid_compressed, self.lat_grid_compressed]
        )
        self.xy_full = np.column_stack([self.xm.flatten(), self.ym.flatten()])
        return None

    def _init_dummy_arrays(self, output_floatprecision) -> None:
        # Prepare matricies
        # i and j represent data pairs, the original point follows _i,
        # remote point is _j
        logging.info("Creating dummy arrays")
        self.cov_ns = np.zeros(
            (self.covar_size, self.covar_size), dtype=output_floatprecision
        )
        self.lat_mat_i = np.zeros(
            (self.covar_size, self.covar_size), dtype=np.float16
        )
        self.lon_mat_i = np.zeros(
            (self.covar_size, self.covar_size), dtype=np.float16
        )
        self.lat_mat_j = np.zeros(
            (self.covar_size, self.covar_size), dtype=np.float16
        )
        self.lon_mat_j = np.zeros(
            (self.covar_size, self.covar_size), dtype=np.float16
        )
        return None

    def _single_cell_process_notinplace(
        self,
        ii,
        jj,
        sdev_i,
        sigma_parms_i,
        lat_grid_compressed_i,
        lon_grid_compressed_i,
        decompose: bool = False,
    ):
        """Standard safe way to do the covariance computation"""
        abs_x, x_j, x_i = scalar_cube_great_circle_distance(
            lat_grid_compressed_i,
            lon_grid_compressed_i,
            self.lat_grid_compressed[jj],
            self.lon_grid_compressed[jj],
            degree_dist=self.degree_dist,
            delta_x_method=self.delta_x_method,
            use_sklearn_haversine=self.use_sklearn_haversine,
        )
        if abs_x > self.max_dist:
            cov_bar = 0.0
        else:
            sdev_j = self.sdev_local_estimates_compressed[jj]
            sigma_parms_j = [
                self.Lx_local_estimates_compressed[jj],
                self.Ly_local_estimates_compressed[jj],
                self.theta_local_estimates_compressed[jj],
            ]
            logging.debug(ii, jj, x_i, x_j)
            logging.debug(
                ii, sdev_i, sigma_parms_i, np.rad2deg(sigma_parms_i[-1])
            )
            logging.debug(
                jj, sdev_j, sigma_parms_j, np.rad2deg(sigma_parms_j[-1])
            )
            # Compute eq 17 in Karspeck et al at each grid point
            cov_bar = c_ij_anistropic_rotated_nonstationary(
                self.v,
                sdev_i,
                sdev_j,
                np.asarray(x_i),
                np.asarray(x_j),
                sigma_parms_i,
                sigma_parms_j,
                decompose=decompose,
            )
        return cov_bar

    def _no_lazy_data(self, cube):
        """Disable iris cube lazy data"""
        if cube.has_lazy_data():
            cube.data  # pylint: disable=pointless-statement
        for coord in ["latitude", "longitude"]:
            if cube.coord(coord).has_lazy_points():
                cube.coord(coord).points  # pylint: disable=expression-not-assigned

    def remap_one_point_2_map(
        self,
        compressed_vector: np.ndarray,
        cube_name: str = "stuff",
        cube_unit: str = "1",
    ):
        """
        Reverse one row/column of the covariance/correlation matrix to a
        plottable iris cube, using mask defined in class.
        """
        dummy_cube = self.Lx_local_estimates.copy()
        masked_vector = uncompress_masked(
            compressed_vector,
            mask=self.cube_mask_1D,
            apply_mask=True,
        )
        dummy_cube.data = masked_vector.reshape(self.xy_shape)
        dummy_cube.rename(cube_name)
        dummy_cube.units = cube_unit
        return dummy_cube

    def positive_definite_check(self):
        """On the fly checking positive semidefinite and eigenvalue clipping"""
        self.cov_ns = perturb_sym_matrix_2_positive_definite(self.cov_ns)
        self.cov_eig = np.sort(linalg.eigvalsh(self.cov_ns))
        self.cov_det = linalg.det(self.cov_ns)


def det22(m22):
    """Explict computation of determinant of 2x2 matrix"""
    m22[np.isclose(m22, 0)] = 0
    return m22[0, 0] * m22[1, 1] - m22[0, 1] * m22[1, 0]


def root4(val):
    """4th root"""
    return np.sqrt(np.sqrt(val))


def c_ij_batched(
    v: float,
    Lxs: np.ndarray,
    Lys: np.ndarray,
    thetas: np.ndarray,
    x_is: np.ndarray,
    x_js: np.ndarray,
    stdevs: np.ndarray,
) -> np.ndarray:
    """DOCUMENTATION"""
    stdev_prod = np.asarray(
        [stdev_i * stdev_j for stdev_i, stdev_j in combinations(stdevs, 2)]
    )
    c_ij = np.divide(stdev_prod, gamma(v) * (2 ** (v - 1)))

    sigmas = [
        sigma_rot_func(Lx, Ly, theta) for Lx, Ly, theta in zip(Lxs, Lys, thetas)
    ]
    sigma_dets = np.asarray([det22(sigma) for sigma in sigmas])
    sqrt_dets = np.sqrt(sigma_dets)
    sqrt_dets = np.asarray([d1 * d2 for d1, d2 in combinations(sqrt_dets, 2)])

    sigma_bars = [
        0.5 * (sigma_i + sigma_j)
        for sigma_i, sigma_j in combinations(sigmas, 2)
    ]
    sigma_bar_dets = np.asarray([det22(sigma) for sigma in sigma_bars])
    taus = np.asarray(
        [
            tau_dist(x_i, x_j, sigma)
            for x_i, x_j, sigma in zip(x_is, x_js, sigma_bars)
        ]
    )
    del sigma_bars

    c_ij = c_ij * np.sqrt(np.divide(sqrt_dets, sigma_bar_dets))

    inner = 2.0 * np.sqrt(v) * taus
    c_ij = c_ij * np.pow(inner, v)
    c_ij = c_ij * modified_bessel_2nd(v, inner)
    del inner

    if np.any(c_ij > stdev_prod):
        raise ValueError("c_ij must always be smaller than sdev_i * sdev_j")

    return c_ij
