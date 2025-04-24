"""Class to compute covariance matrix from ellipse parameters and positions."""

import datetime
import logging
import numbers
import sys
import tracemalloc

from itertools import combinations
from warnings import warn

import numpy as np
import polars as pl
from scipy.special import gamma
from scipy.special import kv as modified_bessel_2nd
from sklearn.metrics.pairwise import haversine_distances
from statsmodels.stats import correlation_tools


from glomar_gridding.constants import RADIUS_OF_EARTH_KM
from glomar_gridding.distances import displacements
from glomar_gridding.types import CovarianceMethod, DeltaXMethod
from glomar_gridding.utils import cov_2_cor, mask_array

if sys.version_info.minor >= 12:
    from itertools import batched
else:
    from glomar_gridding.utils import batched

MAX_DIST_COMPROMISE: float = 6000.0  # Compromise _MAX_DIST_Kar &_MAX_DIST_UKMO
TWO_PI = 2 * np.pi


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

    eigenvalues = np.linalg.eigvalsh(square_sym_matrix)
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
    perturbed = correlation_tools.cov_nearest(
        square_sym_matrix, return_all=False
    )
    if not isinstance(perturbed, np.ndarray):
        raise TypeError(
            "Output of correlation_tools.cov_nearest is not a numpy array"
        )

    eigenvalues_adj = np.linalg.eigvalsh(perturbed)
    min_eigen_adj = np.min(eigenvalues_adj)
    max_eigen_adj = np.max(eigenvalues_adj)
    n_negatives_adj = np.sum(eigenvalues_adj < 0.0)
    print("Post adjustments:")
    print("Number of negative eigenvalues (post_adj) = ", n_negatives_adj)
    print("Largest eigenvalue (post_adj)  = ", max_eigen_adj)
    print("Smallest eigenvalue (post_adj) = ", min_eigen_adj)
    return perturbed


class EllipseCovarianceBuilder:
    """
    Compute covariance from Ellipse parameters and positions.

    v = Matern covariance shape parameter

    Lx - an numpy array of horizontal length scales (
    Ly - an numpy array of meridonal length scales
    theta - an numpy array of rotation angles (RADIANS ONLY)

    sdev - standard deviation -- right now it just takes a numeric array
    if you have multiple contribution to sdev (uncertainties derived from
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
        Arrays with non-stationary parameters
    lats, lons : numpy.ndarray
        Arrays containing the latitude and longitude values
    v=3: float
        Matern shape parameter
    delta_x_method : str
        How are displacements computed between points
    max_dist : float
        If the Haversine distance between 2 points exceed max_dist,
        covariance is set to 0
    output_floatprecision : type
        Float point precision of the output covariance numpy defaults to float64
        Noting that float32 halves the storage and halves the memory to use
    check_positive_definite : bool
        For production this should be False
        but for unit testing it should be True,
        if True a quick on the fly eigenvalue clipping
        will be conducted, if constructed covariance is not
        positive (semi)definite.
    low-memory : bool
        Use a slower, but more memory efficient loop to construct the covariance
        matrix. The more memory efficient approach will be used in all cases if
        the number of unmasked grid-points exceeds 10_000 (this is the number of
        latitudes x the number of longitudes).
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
        delta_x_method: DeltaXMethod | None = "Modified_Met_Office",
        max_dist: float = MAX_DIST_COMPROMISE,
        output_float_precision=np.float32,
        check_positive_definite: bool = False,
        covariance_method: CovarianceMethod = "array",
        batch_size: int | None = None,
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
        self.Lx = mask_array(Lx.astype(output_float_precision))
        self.Ly = mask_array(Ly.astype(output_float_precision))
        self.theta = mask_array(theta.astype(output_float_precision))
        self.stdev = mask_array(stdev.astype(output_float_precision))
        self.max_dist = max_dist
        self.delta_x_method: DeltaXMethod | None = delta_x_method
        self.check_positive_definite = check_positive_definite
        self.lats = lats
        self.lons = lons

        # The cov and corr matrix will be sq matrix of this
        self.xy_shape = self.Lx.shape
        self.n_elements = np.prod(self.xy_shape)

        self._get_mask()

        ove_end_time = datetime.datetime.now()
        print(
            "Overhead processing ended: ",
            ove_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        print("Time elapsed: ", ove_end_time - ove_start_time)
        self._calulate_covariance(
            covariance_method, output_float_precision, batch_size
        )

        # Code now reports eigvals and determinant of the constructed matrix
        self.cov_eig = np.sort(np.linalg.eigvalsh(self.cov_ns))
        self.cov_det = np.linalg.det(self.cov_ns)
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
        self.cor_ns = cov_2_cor(self.cov_ns)

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
        self.data_has_mask = np.ma.is_masked(self.Lx)
        if self.data_has_mask:
            logging.info("Masked pixels detected in input files")
            self.data_mask = self.Lx.mask
            self.covar_size = np.sum(np.logical_not(self.data_mask))
        else:
            logging.info("No masked pixels")
            self.data_mask = np.zeros_like(self.Lx, dtype=bool)
            self.covar_size = self.n_elements

        logging.info("Compressing (masked) array to 1D")
        self.Lx_compressed = self.Lx.compressed()
        self.Ly_compressed = self.Ly.compressed()
        self.theta_compressed = self.theta.compressed()
        self.stdev_compressed = self.stdev.compressed()

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

    def _calulate_covariance(
        self,
        covariance_method: CovarianceMethod,
        output_float_precision: type,
        batch_size: int | None,
    ) -> None:
        cov_start_time = datetime.datetime.now()
        if len(self.Lx_compressed) > 10_000 and covariance_method == "array":
            warn(
                "Number of grid-points > 10_000, setting to low-memory mode "
                + f"(num grid-points = {len(self.Lx_compressed)}"
            )
            covariance_method = "low_memory"
        match covariance_method:
            case "low_memory":
                self.calculate_covariance_loop(output_float_precision)
            case "array":
                self.calculate_covariance_array(output_float_precision)
            case "batched":
                if batch_size is None:
                    raise ValueError(
                        "batch_size must be set if using 'batched' method"
                    )
                self.calculate_covariance_batched(
                    batch_size, output_float_precision
                )
            case _:
                raise ValueError(
                    f"Unknown covariance_method: {covariance_method}"
                )

        cov_end_time = datetime.datetime.now()
        logging.info(
            "Cov processing ended: ", cov_end_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        print("Time elapsed: ", cov_end_time - cov_start_time)
        logging.info(
            "Mem used by cov mat = ", sizeof_fmt(sys.getsizeof(self.cov_ns))
        )
        return None

    def calculate_covariance_array(self, precision: type) -> None:
        """Calculate the covariance matrix from the ellipse parameters"""
        # Calculate distances & Displacements
        dists = haversine_distances(
            np.radians(
                np.column_stack(
                    [self.lat_grid_compressed, self.lon_grid_compressed]
                )
            )
        )
        disp_y, disp_x = displacements(
            self.lat_grid_compressed,
            self.lon_grid_compressed,
            delta_x_method=self.delta_x_method,
        )

        # Initialise Covariance
        self.cov_ns = np.zeros_like(dists, dtype=precision)

        # Mask to upper triangular (exclude diagonal)
        tri_mask = np.triu(np.ones_like(dists), 1) == 0
        disp_y = np.ma.masked_where(tri_mask, disp_y).compressed()
        disp_x = np.ma.masked_where(tri_mask, disp_x).compressed()
        dists = np.ma.masked_where(tri_mask, dists).compressed()

        # Earth scale and set to float32
        disp_y = RADIUS_OF_EARTH_KM * disp_y.astype(np.float32)
        disp_x = RADIUS_OF_EARTH_KM * disp_x.astype(np.float32)
        dists = RADIUS_OF_EARTH_KM * dists.astype(np.float32)

        # Calculate covariance values
        cij = c_ij_anisotropic_array(
            v=self.v,
            Lxs=self.Lx_compressed,
            Lys=self.Ly_compressed,
            thetas=self.theta_compressed,
            delta_x=disp_x,
            delta_y=disp_y,
            stdevs=self.stdev_compressed,
        ).astype(precision)
        cij[dists > self.max_dist] = 0.0

        # Re-populate upper triangular
        np.place(self.cov_ns, ~tri_mask, cij)

        # Add transpose
        self.cov_ns = self.cov_ns + self.cov_ns.T

        # Set diagonal elements
        self.cov_ns = self.cov_ns + np.diag(self.stdev_compressed**2)
        return None

    def calculate_covariance_loop(
        self,
        precision: type,
    ) -> None:
        """
        Compute the covariance matrix from ellipse parameters, using a loop.
        This approach is more memory safe and appropriate for low-memory
        operations, but is significantly slower than self.calculate_covariance
        which uses a lot of pre-computation and a vectorised approach.

        Each ellipse is defined by values from Lxs, Lys, and thetas, with
        standard deviation in stdevs.

        Reference
        ---------
        1) Paciorek and Schevrish 2006 Equation 8 https://doi.org/10.1002/env.785
        2) Karspeck et al 2012 Equation 17 https://doi.org/10.1002/qj.900
        """
        match self.delta_x_method:
            case "Modified_Met_Office":
                disp_fn = _mod_mo_disp_single
            case "Met_Office":
                disp_fn = _mo_disp_single
            case _:
                raise ValueError(
                    f"Unknown 'delta_x_method' value: {self.delta_x_method}"
                )

        # Precomupte common terms
        # Note, these are 1x4 rather than 2x2 for convenience
        sigmas = _sigma_rot_func_multi(
            self.Lx_compressed, self.Ly_compressed, self.theta_compressed
        )
        sqrt_dets = np.sqrt(_det_22_multi(sigmas))
        gamma_v_term = gamma(self.v) * (2 ** (self.v - 1))
        sqrt_v_term = np.sqrt(self.v) * 2

        # Precompute to radians for convenience
        lats = np.deg2rad(self.lat_grid_compressed)
        lons = np.deg2rad(self.lon_grid_compressed)

        # Initialise empty matrix
        n = len(self.Ly_compressed)
        self.cov_ns = np.diag(self.stdev_compressed**2).astype(precision)

        for i, j in combinations(range(n), 2):
            # Leave as zero if too far away
            if (
                _haversine_single(lats[i], lons[i], lats[j], lons[j])
                > self.max_dist
            ):
                continue

            sigma_bar = 0.5 * (sigmas[i] + sigmas[j])
            sigma_bar_det = _det_22(sigma_bar)
            # Leave as zero if cannot invert the sigma_bar matrix
            if sigma_bar_det == 0:
                continue

            stdev_prod = self.stdev_compressed[i] * self.stdev_compressed[j]
            c_ij = stdev_prod / gamma_v_term
            c_ij *= np.sqrt(
                np.divide((sqrt_dets[i] * sqrt_dets[j]), sigma_bar_det)
            )

            # Get displacements
            delta_y, delta_x = disp_fn(lats[i], lons[i], lats[j], lons[j])

            tau = np.sqrt(
                (
                    delta_x * (delta_x * sigma_bar[3] - delta_y * sigma_bar[1])
                    + delta_y
                    * (-delta_x * sigma_bar[2] + delta_y * sigma_bar[0])
                )
                / sigma_bar_det
            )

            inner = sqrt_v_term * tau
            c_ij *= np.pow(inner, self.v)
            c_ij *= modified_bessel_2nd(self.v, inner)
            # if res > stdev_prod:
            #     raise ValueError(
            #         "c_ij must always be smaller than sdev_i * sdev_j"
            #     )
            # Assign and mirror
            self.cov_ns[i, j] = self.cov_ns[j, i] = precision(c_ij)

        return None

    def calculate_covariance_batched(
        self,
        batch_size: int,
        precision: type,
    ) -> None:
        """Batched version"""
        match self.delta_x_method:
            case "Modified_Met_Office":
                disp_fn = _mod_mo_disp_multi
            case "Met_Office":
                disp_fn = _mo_disp_multi
            case _:
                raise ValueError(
                    f"Unknown 'delta_x_method' value: {self.delta_x_method}"
                )
        # Precomupte common terms
        # Note, these are 1x4 rather than 2x2 for convenience
        N = len(self.Lx_compressed)
        sigmas = _sigma_rot_func_multi(
            self.Lx_compressed, self.Ly_compressed, self.theta_compressed
        ).astype(precision)
        sqrt_dets = np.sqrt(_det_22_multi(sigmas))
        gamma_v_term = gamma(self.v) * (2 ** (self.v - 1))
        sqrt_v_term = np.sqrt(self.v) * 2

        # Precompute to radians for convenience
        lats = np.deg2rad(self.lat_grid_compressed)
        lons = np.deg2rad(self.lon_grid_compressed)

        self.cov_ns = np.zeros((N, N), dtype=precision)

        for batch in batched(combinations(range(N), 2), batch_size):
            i_s, j_s = np.asarray(batch).T
            lats_i = lats[i_s]
            lons_i = lons[i_s]
            lats_j = lats[j_s]
            lons_j = lons[j_s]

            # Mask large distances
            dists = _haversine_multi(lats_i, lons_i, lats_j, lons_j)
            mask = dists > self.max_dist
            i_s = i_s.compress(~mask)
            j_s = j_s.compress(~mask)
            lats_i = lats[i_s]
            lons_i = lons[i_s]
            lats_j = lats[j_s]
            lons_j = lons[j_s]
            dy, dx = disp_fn(lats_i, lons_i, lats_j, lons_j)

            loop_c_ij = (
                self.stdev_compressed[i_s] * self.stdev_compressed[j_s]
            ) / gamma_v_term

            sigma_bars = 0.5 * (sigmas[i_s] + sigmas[j_s])
            sigma_bar_dets = _det_22_multi(sigma_bars)
            loop_c_ij *= np.sqrt(
                (sqrt_dets[i_s] * sqrt_dets[j_s]) / sigma_bar_dets
            )
            taus = np.sqrt(
                (
                    dx * (dx * sigma_bars[:, 3] - dy * sigma_bars[:, 1])
                    + dy * (-dx * sigma_bars[:, 2] + dy * sigma_bars[:, 0])
                )
                / sigma_bar_dets
            )
            del sigma_bars, sigma_bar_dets
            inner = sqrt_v_term * taus
            loop_c_ij *= np.power(inner, self.v)
            loop_c_ij *= modified_bessel_2nd(self.v, inner)
            self.cov_ns[i_s, j_s] = loop_c_ij.astype(precision)

        self.cov_ns += self.cov_ns.T
        self.cov_ns += np.diag(
            np.power(self.stdev_compressed, 2).astype(precision)
        )

        return None

    def positive_definite_check(self):
        """On the fly checking positive semidefinite and eigenvalue clipping"""
        self.cov_ns = perturb_sym_matrix_2_positive_definite(self.cov_ns)
        self.cov_eig = np.sort(np.linalg.eigvalsh(self.cov_ns))
        self.cov_det = np.linalg.det(self.cov_ns)


def c_ij_anisotropic_array(
    v: float,
    Lxs: np.ndarray,
    Lys: np.ndarray,
    thetas: np.ndarray,
    delta_x: np.ndarray,
    delta_y: np.ndarray,
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
    delta_x : np.ndarray
        A vector containing the east-west displacements. It is expected that the
        size of this array is 0.5 * N * (N - 1) where N is the length of the Lxs
        vector. These are the displacements between each pair of ellipses, and
        must follow the ordering one would achieve with
        `itertools.combinations`. Note these values should be distances, and not
        degree displacements.
    delta_y : np.ndarray
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

    # Note 1x4 rather than 2x2 for convenience
    sigmas = _sigma_rot_func_multi(Lxs, Lys, thetas)
    sqrt_dets = np.sqrt(_det_22_multi(sigmas))

    sqrt_dets = np.asarray([d1 * d2 for d1, d2 in combinations(sqrt_dets, 2)])

    sigma_bars = (
        pl.from_numpy(
            # Turn the sigma_bar values into a polars frame
            np.asarray(
                [
                    0.5 * (sigma_i + sigma_j)
                    for sigma_i, sigma_j in combinations(sigmas, 2)
                ]
            ),
            schema=["a", "b", "c", "d"],
        )
        .with_columns(
            (pl.col("a") * pl.col("d") - pl.col("b") * pl.col("c")).alias(
                "det"
            ),
            pl.Series("dx", delta_x),
            pl.Series("dy", delta_y),
        )
        # Compute Tau directly from displacements and matrix values
        # [dx, dy] * inv(sigma) * [dx, dy].T
        # inv([[a, b], [c, d]]) = 1/det [[d, -b], [-c, a]]
        .select(
            pl.col("det"),
            (
                (
                    pl.col("dx")
                    * (pl.col("dx") * pl.col("d") - pl.col("dy") * pl.col("b"))
                    + pl.col("dy")
                    * (-pl.col("dx") * pl.col("c") + pl.col("dy") * pl.col("a"))
                )
                / pl.col("det")
            )
            .sqrt()
            .alias("tau"),
        )
    )
    sigma_bar_dets = sigma_bars.get_column("det").to_numpy()
    taus = sigma_bars.get_column("tau").to_numpy()
    del sigma_bars

    c_ij = c_ij * np.sqrt(np.divide(sqrt_dets, sigma_bar_dets))

    inner = 2.0 * np.sqrt(v) * taus
    c_ij = c_ij * np.pow(inner, v)
    c_ij = c_ij * modified_bessel_2nd(v, inner)
    del inner

    if np.any(c_ij > stdev_prod):
        raise ValueError("c_ij must always be smaller than sdev_i * sdev_j")

    return c_ij


def _sigma_rot_func_multi(
    Lx: np.ndarray,
    Ly: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    ct = np.cos(theta)
    st = np.sin(theta)
    c2 = np.power(ct, 2)
    s2 = np.power(st, 2)
    cs = np.multiply(ct, st)
    Lx2 = np.pow(Lx, 2)
    Ly2 = np.pow(Ly, 2)
    del ct, st
    return np.column_stack(
        [
            np.multiply(c2, Lx2) + np.multiply(s2, Ly2),
            np.multiply(cs, Lx2 - Ly2),
            np.multiply(cs, Lx2 - Ly2),
            np.multiply(s2, Lx2) + np.multiply(c2, Ly2),
        ]
    )


def _det_22(
    mats: np.ndarray,
) -> np.ndarray:
    return mats[0] * mats[3] - mats[1] * mats[2]


def _det_22_multi(
    mats: np.ndarray,
) -> np.ndarray:
    return mats[:, 0] * mats[:, 3] - mats[:, 1] * mats[:, 2]


def _haversine_single(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> float:
    dlon = lon0 - lon1
    dlat = lat0 - lat1

    if abs(dlon) < 1e-6 and abs(dlat) < 1e-6:
        return 0

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return RADIUS_OF_EARTH_KM * c


def _haversine_multi(
    lat0: np.ndarray,
    lon0: np.ndarray,
    lat1: np.ndarray,
    lon1: np.ndarray,
) -> np.ndarray:
    dlon = lon0 - lon1
    dlat = lat0 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return RADIUS_OF_EARTH_KM * c


def _mod_mo_disp_single(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> tuple[float, float]:
    dy = lat0 - lat1
    dx = lon0 - lon1
    dx = dx - TWO_PI if dx > np.pi else dx
    dx = dx + TWO_PI if dx < -np.pi else dx

    y_cos_mean = 0.5 * (np.cos(lat0) + np.cos(lat1))
    dx *= y_cos_mean

    return RADIUS_OF_EARTH_KM * dy, RADIUS_OF_EARTH_KM * dx


def _mo_disp_single(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> tuple[float, float]:
    dy = lat0 - lat1
    dx = lon0 - lon1
    dx = dx - TWO_PI if dx > np.pi else dx
    dx = dx + TWO_PI if dx < -np.pi else dx

    return RADIUS_OF_EARTH_KM * dy, RADIUS_OF_EARTH_KM * dx


def _mod_mo_disp_multi(
    lat0: np.ndarray,
    lon0: np.ndarray,
    lat1: np.ndarray,
    lon1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dy = lat0 - lat1
    dx = lon0 - lon1
    dx[dx > np.pi] -= TWO_PI
    dx[dx < -np.pi] += TWO_PI

    y_cos_mean = 0.5 * (np.cos(lat0) + np.cos(lat1))
    dx *= y_cos_mean

    return RADIUS_OF_EARTH_KM * dy, RADIUS_OF_EARTH_KM * dx


def _mo_disp_multi(
    lat0: np.ndarray,
    lon0: np.ndarray,
    lat1: np.ndarray,
    lon1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dy = lat0 - lat1
    dx = lon0 - lon1
    dx[dx > np.pi] -= TWO_PI
    dx[dx < np.pi] += TWO_PI

    return RADIUS_OF_EARTH_KM * dy, RADIUS_OF_EARTH_KM * dx
