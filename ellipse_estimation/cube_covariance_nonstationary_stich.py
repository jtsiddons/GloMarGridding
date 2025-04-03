"""Class to compute covariance matrix from ellipse parameters and positions."""

import datetime

import logging
import numbers
import sys
import tracemalloc

from itertools import combinations

from sklearn.metrics.pairwise import haversine_distances

import numpy as np
from numpy import ma
from numpy import linalg
from scipy.special import kv as modified_bessel_2nd
from scipy.special import gamma
from statsmodels.stats import correlation_tools

from glomar_gridding.distances import (
    displacements,
    sigma_rot_func,
    tau_dist,
)
from glomar_gridding.constants import (
    RADIUS_OF_EARTH_KM,
)
from glomar_gridding.types import DELTA_X_METHOD

MAX_DIST_COMPROMISE: float = 6000.0  # Compromise _MAX_DIST_Kar &_MAX_DIST_UKMO


def mask_array(arr: np.ndarray) -> np.ma.MaskedArray:
    """
    Forces numpy array to be an instance of np.ma.MaskedArray

    Parameters
    ----------
    arr : np.ndarray
        Can be masked or not masked

    Returns
    -------
    arr : np.ndarray
        array is now an instance of np.ma.MaskedArray
    """
    if isinstance(arr, np.ma.MaskedArray):
        return arr
    if isinstance(arr, np.ndarray):
        logging.info("Ad hoc conversion to np.ma.MaskedArray")
        arr = np.ma.MaskedArray(arr)
        return arr
    raise TypeError("Input is not a numpy array.")


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


class EllipseCovarianceBuilder:
    """
    Compute covariance from Ellipse parameters and positions.

    v = Matern covariance shape parameter

    Lx - an numpy array of horizontal length scales (
    Ly - an numpy array of meridonal length scales
    theta - an numpy array of rotation angles (RADIANS ONLY)

    sdev - standard deviation -- right now it just takes a numeric array
    if you have multiple contribution to sdev (uncertainities derived from
    different sources), you need to put them into one array

    Rules:
    Valid (ocean) point:
    1) cov_ns and cor_ns are computed out to max_dist; out of range = 0.0
    2) Masked points are ignored

    Invalid (masked) points:
    1) Skipped over

    max_dist:
    float (km) or (degrees if you want to work in degrees), default 6000km
    if you want infinite distance, just set it to a large number, some fun
    numbers to use:
        1.5E8 (i.e. ~1 astronomical unit (Earth-Sun distance))
        5.0E9 (average distance between Earth and not-a-planet-anymore Pluto)

    Parameters
    ----------
    Lx, Ly, theta, stdev: numpy.ndarray
        arrays with non-stationary parameters
    lats, lons : numpy.ndarray
        arrays containing the latitude and longitude values
    v=3: float
        Matern shape parameter
    delta_x_method : str
        How are displacements computed between points
    max_dist : float
        If the Haversine distance between 2 points exceed max_dist,
        covariance is set to 0
    output_floatprecision :
        Float point precision of the output covariance
        numpy defaults to float64,
        noting that float32 halves the storage and halves the memory to use
    check_positive_definite : bool
        For production this should be False
        but for unit testing it should be True,
        if True a quick on the fly eigenvalue clipping
        will be conducted, if constructed covariance is not
        positive (semi)definite.
    """

    def __init__(
        self,
        Lx: np.ndarray,
        Ly: np.ndarray,
        theta: np.ndarray,
        stdev: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        v: float = 3.0,
        delta_x_method: DELTA_X_METHOD | None = "Modified_Met_Office",
        max_dist: float = MAX_DIST_COMPROMISE,
        output_floatprecision=np.float64,
        check_positive_definite: bool = False,
    ):
        tracemalloc.start()
        ove_start_time = datetime.datetime.now()
        logging.info(
            "Overhead processing start: ",
            ove_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if not isinstance(max_dist, numbers.Number):
            raise ValueError("max_dist must be a number")

        # Defining the input data
        self.v = v  # Matern covariance shape parameter
        self.Lx = mask_array(Lx)
        self.Ly = mask_array(Ly)
        self.theta = mask_array(theta)
        self.stdev = mask_array(stdev)
        self.max_dist = max_dist
        self.delta_x_method: DELTA_X_METHOD | None = delta_x_method
        self.check_positive_definite = check_positive_definite
        self.lats = lats
        self.lons = lons

        # The cov and corr matrix will be sq matrix of this
        self.xy_shape = self.Lx.shape
        self.n_elements = np.prod(self.xy_shape)

        self._get_mask()

        ove_end_time = datetime.datetime.now()
        logging.info(
            "Overhead processing ended: ",
            ove_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        logging.info("Time ellipsed: ", ove_end_time - ove_start_time)

        cov_start_time = datetime.datetime.now()
        self.calculate_covariance(output_floatprecision)

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
        self.data_has_mask = ma.is_masked(self.Lx.data)
        if self.data_has_mask:
            print("Masked pixels detected in input files")
            self.data_mask = self.Lx.mask
            self.covar_size = np.sum(np.logical_not(self.data_mask))
        else:
            print("No masked pixels")
            self.data_mask = np.zeros_like(self.Lx.data.data, dtype=bool)
            self.covar_size = self.n_elements

        print("Compressing (masked) array to 1D")
        self.Lx_compressed = self.Lx.data.compressed()
        self.Ly_compressed = self.Ly.data.compressed()
        self.theta_compressed = self.theta.data.compressed()
        self.stdev_compressed = self.stdev.data.compressed()

        self.x_grid, self.y_grid = np.meshgrid(self.lons, self.lats)
        self.x_mask = np.ma.masked_where(self.data_mask, self.x_grid)
        self.y_mask = np.ma.masked_where(self.data_mask, self.y_grid)
        self.lat_grid_compressed = self.y_mask.compressed()
        self.lon_grid_compressed = self.x_mask.compressed()

        self.xy_compressed = np.column_stack(
            [self.lon_grid_compressed, self.lat_grid_compressed]
        )
        self.xy_full = np.column_stack(
            [self.x_mask.flatten(), self.y_mask.flatten()]
        )
        return None

    def calculate_covariance(self, output_floatprecision: type) -> None:
        """Calculate the covariance matrix from the ellipse parameters"""
        # Calculate distances & Displacements
        disp_y, disp_x = displacements(
            self.lat_grid_compressed,
            self.lon_grid_compressed,
            delta_x_method=self.delta_x_method,
            to_radians=True,
        )
        dists = haversine_distances(
            np.radians(
                np.column_stack(
                    [self.lat_grid_compressed, self.lon_grid_compressed]
                )
            )
        )
        disp_y = RADIUS_OF_EARTH_KM * disp_y
        disp_x = RADIUS_OF_EARTH_KM * disp_x
        dists = RADIUS_OF_EARTH_KM * dists

        # Initialise Covariance
        self.cov_ns = np.zeros_like(dists, dtype=output_floatprecision)

        # Mask to upper triangular (exclude diagonal)
        tri_mask = np.triu(np.ones_like(dists), 1) == 0
        disp_y_comp = np.ma.masked_where(tri_mask, disp_y)
        disp_x_comp = np.ma.masked_where(tri_mask, disp_x)
        dists_comp = np.ma.masked_where(tri_mask, dists)

        # Calculate covariance values
        cij = c_ij_anisotropic_array(
            v=self.v,
            Lxs=self.Lx_compressed,
            Lys=self.Ly_compressed,
            thetas=self.theta_compressed,
            x_is=disp_x_comp,
            x_js=disp_y_comp,
            stdevs=self.stdev_compressed,
        )
        cij[dists_comp > self.max_dist] = 0.0

        # Re-populate upper triangular
        np.place(self.cor_ns, ~tri_mask, cij)

        # Add transpose
        self.cov_ns = self.cov_ns + self.cor_ns.T

        # Set diagonal elements
        self.cov_nx = self.cov_ns + np.diag(self.stdev_compressed**2)
        return None

    def positive_definite_check(self):
        """On the fly checking positive semidefinite and eigenvalue clipping"""
        self.cov_ns = perturb_sym_matrix_2_positive_definite(self.cov_ns)
        self.cov_eig = np.sort(linalg.eigvalsh(self.cov_ns))
        self.cov_det = linalg.det(self.cov_ns)


def det22(m22):
    """Explict computation of determinant of 2x2 matrix"""
    m22[np.isclose(m22, 0)] = 0
    return m22[0, 0] * m22[1, 1] - m22[0, 1] * m22[1, 0]


def c_ij_anisotropic_array(
    v: float,
    Lxs: np.ndarray,
    Lys: np.ndarray,
    thetas: np.ndarray,
    x_is: np.ndarray,
    x_js: np.ndarray,
    stdevs: np.ndarray,
) -> np.ndarray:
    """
    Compute the covariances between pairs of ellipses, at displacements.

    Each ellipse is defined by values from Lxs, Lys, and thetas, with standard
    deviation in stdevs.

    The displacements between each pair of ellipses are x_is and x_js.

    For N ellipses, the number of displacements should be 1/2 * N * (N - 1),
    i.e. the displacement between each pair combination of ellipses. This
    function will return the upper triangular values of the covariance
    matrix (excluding the diagonal).

    `itertools.combinations` is used to handle ordering, so the displacements
    must be ordered in the same way.

    Reference
    ---------
    1) Paciorek and Schevrish 2006 Equation 8 https://doi.org/10.1002/env.785
    2) Karspeck et al 2012 Equation 17 https://doi.org/10.1002/qj.900

    Parameters
    ----------
    v : float
        Matern shape parameter
    Lxs : numpy.ndarray
        A vector containing the Lx values - the ellipse semi-major axis length
        scales. These are the values for each ellipse (not duplicated).
    Lys : numpy.ndarray
        A vector containing the Ly values - the ellipse semi-minor axis length
        scales. These are the values for each ellipse (not duplicated).
    thetas : numpy.ndarray
        A vector containing the theta values - the ellipse angles (in radians).
        These are the values for each ellipse (not duplicated).
    x_is : np.ndarray
        A vector containing the east-west displacements. It is expected that the
        size of this array is 0.5 * N * (N - 1) where N is the length of the Lxs
        vector. These are the displacements between each pair of ellipses, and
        must follow the ordering one would achieve with
        `itertools.combinations`. Note these values should be distances, and not
        degree displacements.
    x_js : np.ndarray
        A vector containing the north-south displacements. It is expected that
        the size of this array is 0.5 * N * (N - 1) where N is the length of the
        Lxs vector. These are the displacements between each pair of ellipses,
        and must follow the ordering one would achieve with
        `itertools.combinations`. Note these values should be distances, and not
        degree displacements.
    stdevs : np.ndarray
        A vector containing the standard deviation for each ellipse.

    Returns
    -------
    c_ij : numpy.ndarray
        A vector containing the covariance values between each pair of ellipses.
        This will return the components of the upper triangle of the covariance
        matrix as a vector (excluding the diagonal).
    """
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
