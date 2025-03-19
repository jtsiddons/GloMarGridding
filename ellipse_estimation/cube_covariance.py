"""
Requires numpy, scipy, sklearm
iris needs to be installed (it is required by other modules within this package
xarray cubes should work via iris interface)
"""

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal, get_args, cast
import math
import warnings
import logging

import iris
import iris.coords as icoords
import iris.util as iutil
from cf_units import Unit
from joblib import Parallel, delayed
import numpy as np
from numpy import ma
from numpy import linalg
from scipy.special import kv as modified_bessel_2nd
from scipy.special import gamma
from scipy.spatial.transform import Rotation as R
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics.pairwise import haversine_distances, euclidean_distances
# from astropy.constants import R_earth

from glomar_gridding.constants import RADIUS_OF_EARTH_KM
from glomar_gridding.utils import deg_to_km


DEFAULT_N_JOBS: int = 4
DEFAULT_BACKEND: str = "loky"  # loky appears to be fastest

MODEL_TYPE = Literal[
    "ps2006_kks2011_iso",
    "ps2006_kks2011_ani",
    "ps2006_kks2011_ani_r",
    "ps2006_kks2011_iso_pd",
    "ps2006_kks2011_ani_pd",
    "ps2006_kks2011_ani_r_pd",
]

FFORM = Literal[
    "anistropic_rotated",
    "anistropic",
    "isotropic",
    "anistropic_rotated_pd",
    "anistropic_pd",
    "isotropic_pd",
]

SUPERCATEGORY = Literal[
    "1_param_matern",
    "2_param_matern",
    "3_param_matern",
    "1_param_matern_pd",
    "2_param_matern_pd",
    "3_param_matern_pd",
]

MODEL_TYPE_TO_SUPERCATEGORY: dict[MODEL_TYPE, SUPERCATEGORY] = {
    "ps2006_kks2011_iso": "1_param_matern",
    "ps2006_kks2011_ani": "2_param_matern",
    "ps2006_kks2011_ani_r": "3_param_matern",
    "ps2006_kks2011_iso_pd": "1_param_matern_pd",
    "ps2006_kks2011_ani_pd": "2_param_matern_pd",
    "ps2006_kks2011_ani_r_pd": "3_param_matern_pd",
}

FFORM_TO_MODELTYPE: dict[FFORM, MODEL_TYPE] = {
    "anistropic_rotated": "ps2006_kks2011_ani_r",
    "anistropic": "ps2006_kks2011_ani",
    "isotropic": "ps2006_kks2011_iso",
    "anistropic_rotated_pd": "ps2006_kks2011_ani_r_pd",
    "anistropic_pd": "ps2006_kks2011_ani_pd",
    "isotropic_pd": "ps2006_kks2011_iso_pd",
}

SUPERCATEGORY_PARAMS: dict[SUPERCATEGORY, OrderedDict[str, Unit]] = {
    "3_param_matern": OrderedDict(
        [
            ("Lx", Unit("degrees")),
            ("Ly", Unit("degrees")),
            ("theta", Unit("radians")),
            ("standard_deviation", Unit("K")),
            ("qc_code", Unit("1")),
            ("number_of_iterations", Unit("1")),
        ]
    ),
    "2_param_matern": OrderedDict(
        [
            ("Lx", Unit("degrees")),
            ("Ly", Unit("degrees")),
            ("standard_deviation", Unit("K")),
            ("qc_code", Unit("1")),
            ("number_of_iterations", Unit("1")),
        ]
    ),
    "1_param_matern": OrderedDict(
        [
            ("R", Unit("degrees")),
            ("standard_deviation", Unit("K")),
            ("qc_code", Unit("1")),
            ("number_of_iterations", Unit("1")),
        ]
    ),
    "3_param_matern_pd": OrderedDict(
        [
            ("Lx", Unit("km")),
            ("Ly", Unit("km")),
            ("theta", Unit("radians")),
            ("standard_deviation", Unit("K")),
            ("qc_code", Unit("1")),
            ("number_of_iterations", Unit("1")),
        ]
    ),
    "2_param_matern_pd": OrderedDict(
        [
            ("Lx", Unit("km")),
            ("Ly", Unit("km")),
            ("standard_deviation", Unit("K")),
            ("qc_code", Unit("1")),
            ("number_of_iterations", Unit("1")),
        ]
    ),
    "1_param_matern_pd": OrderedDict(
        [
            ("R", Unit("km")),
            ("standard_deviation", Unit("K")),
            ("qc_code", Unit("1")),
            ("number_of_iterations", Unit("1")),
        ]
    ),
}


class MaternEllipseModel:
    """
    The class that contains variogram/ellipse fitting methods and parameters

    This class assumes your input to be a standardised correlation matrix
    They are easier to handle because stdevs in the covariance function become 1

    Parameters
    ----------
    anisotropic : bool
        Should the output be an ellipse? Set to False for circle.
    rotated : bool
        Can the ellipse be rotated. If anisotropic is False this value cannot
        be True.
    physical_distance : bool
        Use physical distances rather than lat/lon distance.
    v : float
        Matern Shape Parameter. Must be > 0.0.
    unit_sigma=True: bool
        When MLE fitting the Matern parameters,
        assuming the Matern parameters themselves
        are normally distributed,
        there is standard deviation within the log likelihood function.

        See Wikipedia entry for Maxmimum Likelihood under:
        - Continuous distribution, continuous parameter space

        Its actual value is not important
        to the best (MLE) estimate of the Matern parameters.
        If one assumes the parameters are normally distributed,
        the mean (best estimate) is independent of its variance.
        In fact in Karspeck et al 2012, it is simply set to 1 (Eq B1).
        This value can however be computed. It serves a similar purpose as
        the original standard deviation:
        in this case, how the actual observed semivariance disperses
        around the fitted variogram.

        Here it defaults to 1 just as Karspeck.
    """

    def __init__(
        self,
        anisotropic: bool,
        rotated: bool,
        physical_distance: bool,
        v: float,
        unit_sigma: bool = False,
    ) -> None:
        if v <= 0:
            raise ValueError("'v' must be > 0")
        self.anisotropic = anisotropic
        self.rotated = rotated
        self.physical_distance = physical_distance
        self.v = v
        self.unit_sigma = unit_sigma

        self._get_model_names()
        self.supercategory_params = SUPERCATEGORY_PARAMS[self.supercategory]
        self.supercategory_n_params = len(self.supercategory_params)

        self._get_defaults()

        return None

    def _get_model_names(self) -> None:
        """
        Determine the fform, model type, and supercategory.

        Returns
        -------
        None
        """
        if self.rotated and not self.anisotropic:
            raise ValueError("Cannot have an isotropic rotated fform")

        fform_builder: list[str] = (
            ["anisotropic"] if self.anisotropic else ["istropic"]
        )
        if self.rotated:
            fform_builder.append("rotated")
        if self.physical_distance:
            fform_builder.append("pd")

        fform_str: str = "_".join(fform_builder)
        if fform_str not in get_args(FFORM):
            raise ValueError("Could not compute fform value from inputs")

        self.fform: FFORM = cast(FFORM, fform_str)
        self.model_type: MODEL_TYPE = FFORM_TO_MODELTYPE[self.fform]
        self.supercategory: SUPERCATEGORY = MODEL_TYPE_TO_SUPERCATEGORY[
            self.model_type
        ]

        return None

    def _get_defaults(self) -> None:
        """Get default values for the MaternEllipseModel."""
        match self.fform:
            case "isotropic":
                self.n_params = 1
                self.default_guesses = [7.0]
                self.default_bounds = [(0.5, 50.0)]
                self.c_ij = lambda X, R: c_ij_istropic(self.v, 1, X, R)
            case "isotropic_pd":
                self.n_params = 1
                self.default_guesses = [deg_to_km(7.0)]
                self.default_bounds = [
                    (deg_to_km(0.5), deg_to_km(50)),
                ]
                self.c_ij = lambda X, R: c_ij_istropic(self.v, 1, X, R)
            case "anistropic":
                self.n_params = 2
                self.default_guesses = [7.0, 7.0]
                self.default_bounds = [(0.5, 50.0), (0.5, 30.0)]
                self.c_ij = lambda X, Lx, Ly: c_ij_anistropic_unrotated(
                    self.v, 1, X[0], X[1], Lx, Ly
                )
            case "anistropic_pd":
                self.n_params = 2
                self.default_guesses = [deg_to_km(7.0), deg_to_km(7.0)]
                self.default_bounds = [
                    (deg_to_km(0.5), deg_to_km(50)),
                    (deg_to_km(0.5), deg_to_km(30)),
                ]
                self.c_ij = lambda X, Lx, Ly: c_ij_anistropic_unrotated(
                    self.v, 1, X[0], X[1], Lx, Ly
                )
            case "anistropic_rotated":
                self.n_params = 3
                self.default_guesses = [7.0, 7.0, 0.0]
                self.default_bounds = [
                    (0.5, 50.0),
                    (0.5, 30.0),
                    (-2.0 * np.pi, 2.0 * np.pi),
                ]
                self.c_ij = lambda X, Lx, Ly, theta: c_ij_anistropic_rotated(
                    self.v, 1, X[0], X[1], Lx, Ly, theta
                )
            case "anistropic_rotated_pd":
                self.n_params = 3
                self.default_guesses = [deg_to_km(7.0), deg_to_km(7.0), 0.0]
                self.default_bounds = [
                    (deg_to_km(0.5), deg_to_km(50)),
                    (deg_to_km(0.5), deg_to_km(30)),
                    (-2.0 * math.pi, 2.0 * math.pi),
                ]
                self.c_ij = lambda X, Lx, Ly, theta: c_ij_anistropic_rotated(
                    self.v, 1, X[0], X[1], Lx, Ly, theta
                )

    def negative_log_likelihood(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: tuple[float, ...],
        arctanh_transform: bool = True,
        backend: str = DEFAULT_BACKEND,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> float:
        """
        Compute the negative log-likelihood given observed X independent
        observations (displacements) and y dependent variable (the observed
        correlation), and Matern parameters params. Namely does the Matern
        covariance function using params, how close it explains the observed
        displacements and correlations.

        log(LL) = SUM (f (y,x|params) )
        params = Maximise (log(LL))
        params = Minimise (-log(LL)) which is how usually the computer solves it
        assuming errors of params are normally distributed

        There is a hidden scale/standard deviation in
        stats.norm.logpdf(scale, which defaults to 1)
        but since we have scaled our values to covariance to correlation (and
        even used Fisher transform) as part of the function, it can be dropped

        Otherwise, you need to have stdev as the last value of params, and
        should be set to the scale parameter

        Parameters
        ----------
        X : np.ndarray
            Observed displacements
        y : np.ndarray
            Observed correlation
        params : tuple of Matern parameters
            (in the current optimize iteration) or if you want to
            compute the actual negative log-likelihood
        arctanh_transform : bool
            Should the Fisher (arctanh) transform be used
            This is usually option, but it does make the computation
            more stable if they are close to 1 (or -1; doesn't apply here)
        backend : str
            joblib backend
            See https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        n_jobs : int
            Number of threads/parallel computation

        Returns
        -------
        nLL : float
            The negative log likelihood
        """
        sigma = 1
        if not self.unit_sigma:
            if len(params) > self.n_params:
                raise ValueError("Cannot get sigma from params")
            sigma = params[self.n_params]

        match self.n_params:
            case 1:  # Circle
                R = params[0]  # Radius
                y_LL = self.c_ij(X, R)
            case 2:  # Un-rotated Ellipse
                Lx = params[0]
                Ly = params[1]
                y_LL = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(self.c_ij)(X[n_x_j, :], Lx, Ly)
                    for n_x_j in range(X.shape[0])
                )
                # y_LL = []
                # for n_x_j in range(X.shape[0]):
                #    y_LL.append(self.c_ij(X[n_x_j, :], Lx, Ly))
            case 3:  # Rotated Ellipse
                Lx = params[0]
                Ly = params[1]
                theta = params[2]
                y_LL = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(self.c_ij)(X[n_x_j, :], Lx, Ly, theta)
                    for n_x_j in range(X.shape[0])
                )
                # y_LL = []
                # for n_x_j in range(X.shape[0]):
                #    y_LL.append(self.c_ij(X[n_x_j,:], Lx, Ly, theta))
            case _:
                raise ValueError("Unexpected length of self.n_params.")

        y_LL = np.array(y_LL)
        # if y is correlation,
        # it might be useful to Fisher transform them before plugging into
        # norm.logpdf this affects values close to 1 and -1
        # imposing better behavior to the differences at the tail

        if arctanh_transform:
            # Warning against arctanh(abs(y) > 1); (TODO: Add correction later)
            arctanh_threshold = 0.999999
            # arctanh_threshold = 1.0
            max_abs_y = np.max(np.abs(y))
            max_abs_yLL = np.max(np.abs(y_LL))
            if max_abs_y >= arctanh_threshold:
                warn_msg = "abs(y) >= " + str(arctanh_threshold) + " detected; "
                warn_msg += "fudged to threshold; max(abs(y))=" + str(max_abs_y)
                warnings.warn(warn_msg, RuntimeWarning)
                y[np.abs(y) > arctanh_threshold] = (
                    np.sign(y[np.abs(y) > arctanh_threshold])
                    * arctanh_threshold
                )
                # y[np.abs(y) > 1] = np.sign(y[np.abs(y) > 1]) * 0.9999

            # if np.any(np.isclose(np.abs(y), 1.0)):
            #     warn_msg = (
            #         "abs(y) is close to 1; max(abs(y))="
            #         + str(max_abs_y)
            #     )
            #     warnings.warn(warn_msg, RuntimeWarning)
            #     y[np.isclose(np.abs(y), 1.0)] = (
            #         np.sign(y[np.isclose(np.abs(y), 1.0)]) * 0.9999
            #     )

            if max_abs_yLL >= 1:
                warn_msg = (
                    "abs(y_LL) >= " + str(arctanh_threshold) + " detected; "
                )
                warn_msg += "fudged to threshold; max(abs(y_LL))=" + str(
                    max_abs_yLL
                )
                warnings.warn(warn_msg, RuntimeWarning)
                y_LL[np.abs(y_LL) > arctanh_threshold] = (
                    np.sign(y_LL[np.abs(y_LL) > arctanh_threshold])
                    * arctanh_threshold
                )
                # y_LL[np.abs(y_LL) > 1] = (
                #     np.sign(y_LL[np.abs(y_LL) > 1]) * 0.9999
                # )

            # if np.any(np.isclose(np.abs(y_LL), 1.0)):
            #     warn_msg = (
            #         "abs(y_LL) close to 1 detected; max(abs(y_LL))="
            #         + str(max_abs_yLL)
            #     )
            #     warnings.warn(warn_msg, RuntimeWarning)
            #     y_LL[np.isclose(np.abs(y_LL), 1.0)] = (
            #         np.sign(y_LL[np.isclose(np.abs(y_LL), 1.0)]) * 0.9999
            #     )

            nLL = -1.0 * np.sum(
                stats.norm.logpdf(
                    np.arctanh(y), loc=np.arctanh(y_LL), scale=sigma
                )
            )
        else:
            nLL = -1.0 * np.sum(stats.norm.logpdf(y, loc=y_LL, scale=sigma))
        return nLL

    def negative_log_likelihood_function(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = DEFAULT_N_JOBS,
        backend: str = DEFAULT_BACKEND,
    ) -> Callable[[tuple[float, ...]], float]:
        """Creates a function that can be fed into scipy.optimizer.minimize"""

        def f(params: tuple[float, ...]):
            return self.negative_log_likelihood(
                X, y, params, n_jobs=n_jobs, backend=backend
            )

        return f

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        guesses: list[float] | None = None,
        bounds: list[tuple[float, ...]] | None = None,
        opt_method: str = "Nelder-Mead",
        tol: float | None = None,
        estimate_SE: str | None = "bootstrap_parallel",
        n_sim: int = 500,
        n_jobs: int = DEFAULT_N_JOBS,
        backend: str = DEFAULT_BACKEND,
        random_seed: int = 1234,
    ) -> list:
        """
        Default solver in Nelder-Mead as used in the Karspeck paper
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
        default max-iter is 200 x (number_of_variables)
        for 3 variables (Lx, Ly, theta) --> 200x3 = 600
        note: unlike variogram fitting, no nugget, no sill, and no residue
        variance (normalised data but Fisher transform needed?)
        can be adjusted using "maxiter" within "options" kwargs

        Much of the variable names are defined the same way as earlier

        Parameters
        ----------
        X, y : np.ndarray
            distances and observed correlations
        guesses=None :
            Tuples/lists of initial values to scipy.optimize.minimize
        bounds=None :
            Tuples/lists of bounds for fitted parameters
        opt_method : str
            scipy.optimize.minimize optimisation method. Defaults to
            "Nelder-Mead".
        tol=None : float
            scipy.optimize.minimize convergence tolerance
        estimate_SE='bootstrap_parallel' : str
            how to estimate standard error if needed
        n_sim=500 : int
            number of bootstrap to estimate standard error
        n_jobs=_default_n_jobs : int
            number of threads
        backend=_default_backend : str
            joblib backend
        random_seed=1234 : int, random seed for bootstrap

        Returns
        -------
        list : [fitted parameters,
                standard error of the fitted parameters,
                bounds of fitted parameters]
        """
        ##
        guesses = guesses or self.default_guesses
        bounds = bounds or self.default_bounds

        if not self.unit_sigma:
            guesses.append(0.1)
            bounds.append((0.0001, 0.5))
        ##
        LL_observedXy_unknownparams = self.negative_log_likelihood_function(
            X, y, n_jobs=n_jobs, backend=backend
        )
        ##
        print("X range: ", np.min(X), np.max(X))
        print("y range: ", np.min(y), np.max(y))
        zipper = zip(guesses, bounds)
        for g, b in zipper:
            print("init: ", g, "bounds: ", b)
        ##
        results = minimize(
            LL_observedXy_unknownparams,
            guesses,
            bounds=bounds,
            method=opt_method,
            tol=tol,
        )
        ##
        # This does not account for standard errors in the
        # correlation/covariance matrix!
        if estimate_SE is None:
            print("Standard error estimates not required")
            return [results, None, bounds]

        match estimate_SE:
            case "bootstrap_serial":
                # Serial
                sim_params = []
                for looper in range(n_sim):
                    sim_params.append(
                        self._bootstrap_once(
                            X,
                            y,
                            guesses,
                            bounds,
                            opt_method,
                            tol=tol,
                            seed=random_seed + looper,
                        )
                    )
                sim_params = np.array(sim_params)
                SE = np.std(sim_params, axis=0)
            case "bootstrap_parallel":
                # Parallel
                # On JASMIN Jupyter: n_jobs = 5 leads to 1/3 wallclock time
                kwargs_0 = {"n_jobs": n_jobs, "backend": backend}
                workers = range(n_sim)
                sim_params = Parallel(**kwargs_0)(
                    delayed(self._bootstrap_once)(
                        X,
                        y,
                        guesses,
                        bounds,
                        opt_method,
                        tol=tol,
                        seed=random_seed + worker,
                    )
                    for worker in workers
                )
                sim_params = np.array(sim_params)
                SE = np.std(sim_params, axis=0)
            case "hessian":
                # note: autograd does not work with scipy's Bessel functions
                raise NotImplementedError(
                    "Second order deriviative (Hessian) of "
                    + "Fisher Information not implemented"
                )
            case _:
                raise ValueError("Unknown estimate_SE")

        return [results, SE, bounds]

    def _bootstrap_once(
        self,
        X: np.ndarray,
        y: np.ndarray,
        guesses: list[float],
        bounds: list[tuple[float, ...]],
        opt_method: str,
        tol: float | None = None,
        seed: int = 1234,
    ):
        """Bootstrap refit the Matern parameters"""
        rng = np.random.RandomState(seed)
        len_obs = len(y)
        i_obs = np.arange(len_obs)
        bootstrap_i = rng.choice(i_obs, size=len_obs, replace=True)
        X_boot = X[bootstrap_i, ...]
        y_boot = y[bootstrap_i]
        LL_bootXy_simulatedparams = self.negative_log_likelihood_function(
            X_boot, y_boot
        )
        ans = minimize(
            LL_bootXy_simulatedparams,
            guesses,
            bounds=bounds,
            method=opt_method,
            tol=tol,
        )
        return ans.x


class CovarianceCube:
    """
    Class to build spatial covariance and correlation matricies
    Interacts with MLE_c_ij_Builder_Karspeck to build covariance models

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
        data_cube,
    ) -> None:
        # Check input data_cube is actually usable
        self.tcoord_pos = -1
        self.xycoords_pos = []
        self.xycoords_name = []
        for i, coord in enumerate(data_cube.coords()):
            if coord.standard_name == "time":
                self.tcoord_pos = i
            if coord.standard_name in ["latitude", "longitude"]:
                self.xycoords_pos.append(i)
                self.xycoords_name.append(coord.standard_name)
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
        # if 'lat' in self.xycoords_name[0]:
        #     self.xycoords_name = self.xycoords_name[::-1]

        # Defining the input data
        self.data_cube = data_cube
        self.xy_shape = self.data_cube[0].shape
        if len(self.xy_shape) != 2:
            raise ValueError(
                "Time slice maps should be 2D; check extra dims (ensemble?)"
            )
        self.big_covar_size = np.prod(self.xy_shape)

        # Detect data mask and determine dimension of array without masked data
        # Almost certain True near the coast
        self.data_has_mask = ma.is_masked(self.data_cube.data)
        if self.data_has_mask:
            # Depending on dataset, the mask might not be invariant
            # (like sea ice)
            # xys with time varying mask are currently discarded.
            # If analysis is conducted seasonally
            # this should normally not a problem unless in high latitudes
            self.cube_mask = np.any(self.data_cube.data.mask, axis=0)
            self.cube_mask_1D = self.cube_mask.flatten()
            self._self_mask()
            self.small_covar_size = np.sum(np.logical_not(self.cube_mask))
        else:
            self.cube_mask = np.zeros_like(data_cube[0].data.data, dtype=bool)
            self.cube_mask_1D = self.cube_mask.flatten()
            self.small_covar_size = self.big_covar_size

        # Look-up table of the coordinates
        self.xx, self.yy = np.meshgrid(
            self.data_cube.coord("longitude").points,
            self.data_cube.coord("latitude").points,
        )
        self.xm = ma.masked_where(self.cube_mask, self.xx)
        self.ym = ma.masked_where(self.cube_mask, self.yy)
        self.xy = np.column_stack([self.xm.compressed(), self.ym.compressed()])
        self.xy_full = np.column_stack([self.xm.flatten(), self.ym.flatten()])

        # Length of time dimension
        self.time_n = len(self.data_cube.coord("time").points)

        # Calculate the actual covariance and correlation matrix:
        self._calc_cov(correlation=True)

        return None

    def _calc_cov(
        self,
        correlation: bool = False,
        rounding: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        correlation : bool
            to state if you want a correlation (normalised covariance or not)
        rounding : int
            round the values of the output

        Returns
        -------
        None
        """
        # Reshape data to (t, xy),
        # get rid of mask values -- cannot caculate cov for such data
        xyflatten_data = self.data_cube.data.reshape(
            (
                len(self.data_cube.coord("time").points),
                self.big_covar_size,
            )
        )
        xyflatten_data = ma.compress_rowcols(xyflatten_data, -1)
        # Remove mean --
        # even data that says "SST anomalies" don't have zero mean (?!)
        xy_mean = np.mean(xyflatten_data, axis=0, keepdims=True)
        xyflatten_data = xyflatten_data - xy_mean
        self.Cov = np.matmul(np.transpose(xyflatten_data), xyflatten_data)
        # xy_cov = _AT_A(xyflatten_data)
        if correlation:
            return self._cov2cor(rounding=rounding)
        if rounding is not None:
            self.Cov = np.round(self.Cov, rounding)

    def _cov2cor(
        self,
        rounding: int | None = None,
    ) -> None:
        """
        Normalises the covariance matrices within the class instance
        and return correlation matrices
        https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b

        Parameters
        ----------
        rounding : int
            round the values of the output

        Returns
        -------
        None
        """
        stdevs = np.sqrt(np.diag(self.Cov))
        normalisation = np.outer(stdevs, stdevs)
        self.Corr = self.Cov / normalisation
        self.Corr[self.Cov == 0] = 0
        # sigma_inverse = np.zeros_like(self.Cov)
        # np.fill_diagonal(sigma_inverse,
        #                  np.reciprocal(np.sqrt(np.diagonal(self.Cov))))
        # ans = np.matmul(np.matmul(sigma_inverse, self.Cov), sigma_inverse)
        # #ans = _Dinv_A_Dinv(self.Cov)
        if rounding is not None:
            self.Corr = np.round(self.Corr, rounding)
        return None

    def calc_distance_angle_selfcube(
        self,
        haversine: bool = False,
        compressed: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute distances between grid points of cube data stored
        within class instance

        Parameters
        ----------
        haversine : bool
            Use the haversine instead
        compressed : bool
            Do a np.ma.MaskedArray.comressed, this get rids of masked points...

        Returns
        -------
        D, A : np.ndarray [float]
            D (distance in km), A (distance in angle)
        """
        xx, yy = np.meshgrid(
            self.data_cube.coord("longitude").points,
            self.data_cube.coord("latitude").points,
        )
        xx = ma.masked_where(self.cube_mask, xx)
        yy = ma.masked_where(self.cube_mask, yy)
        if compressed:
            unmeshed_x = xx.compressed()
            unmeshed_y = yy.compressed()
        else:
            unmeshed_x = xx.flatten()
            unmeshed_y = yy.flatten()
        D, A = self._calc_distance_angle(
            unmeshed_x, unmeshed_y, haversine=haversine
        )
        return D, A

    def _calc_distance_angle(
        self,
        unmeshed_x: np.ndarray,
        unmeshed_y: np.ndarray,
        haversine: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute distance and angle between all points."""
        yx_og = np.column_stack([unmeshed_y, unmeshed_x])
        if haversine:
            # Great circle - Earth
            yx = np.column_stack(
                [
                    np.array([np.deg2rad(lat) for lat in unmeshed_y]),
                    np.array([np.deg2rad(lon) for lon in unmeshed_x]),
                ]
            )
            D = haversine_distances(yx) * RADIUS_OF_EARTH_KM
        else:
            # Delta degrees - locally flat Earth, treating like it an image
            yx = yx_og
            D = euclidean_distances(yx)

        # Angle difference vs x/longitude-axis - below two yields the same
        # result
        # _dy = euclidean_distances(
        #     np.column_stack([unmeshed_y, np.zeros_like(unmeshed_y)])
        # )
        # _dx = euclidean_distances(
        #     np.column_stack([np.zeros_like(unmeshed_x), unmeshed_x])
        # )
        _dy = euclidean_distances(
            np.column_stack([yx[:, 0], np.zeros_like(unmeshed_y)])
        )
        _dx = euclidean_distances(
            np.column_stack([np.zeros_like(unmeshed_x), yx[:, 1]])
        )
        dy = np.triu(_dy) - np.tril(_dy)
        dx = np.triu(_dx) - np.tril(_dx)
        # In radians; note, arctan2 can tell if dy and dx are negative
        A = np.arctan2(dy, dx)
        ##
        return D, A  # Both D and A are square matrices

    def _self_mask(self) -> None:
        """Broadcast cube_mask to all observations"""
        broadcasted_mask = np.broadcast_to(
            self.cube_mask, self.data_cube.data.shape
        )
        self.data_cube.data = ma.masked_where(
            broadcasted_mask, self.data_cube.data
        )

    def _reverse_mask_from_compress_1d(
        self,
        compressed_1D_vector: np.ndarray,
        fill_value: float = 0.0,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        Since there are lot of flatten and compressing going on for observations
        and fitted parameters, this reverses the 1D array to the original 2D map

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

        DANGER WARNING, use different fill_value depending on situation
        This affects how signal and image processing module interacts
        with missing and masked values
        They don't ignore them, so a fill_value like 0 may be sensible
        for covariance (-999.99 will
        do funny things if it finds its way to a convolution and filter)
        iris doesn't care, but should use something like -999.99 or something
        """
        compressed_counter = 0
        ans = np.zeros_like(self.cube_mask_1D, dtype=dtype)
        for i in range(len(self.cube_mask_1D)):
            if not self.cube_mask_1D[i]:
                ans[i] = compressed_1D_vector[compressed_counter]
                compressed_counter += 1
        ma.set_fill_value(ans, fill_value)
        ans = ma.masked_where(self.cube_mask_1D, ans)
        return ans

    def remap_one_point_2_map(
        self,
        compressed_vector: np.ndarray,
        cube_name: str = "stuff",
        cube_unit: str = "1",
    ) -> iris.cube.Cube:
        """
        Reverse one row/column of the covariance/correlation matrix
        to a plottable iris cube using mask defined in class
        """
        dummy_cube = self.data_cube[0, :, :].copy()
        masked_vector = self._reverse_mask_from_compress_1d(compressed_vector)
        dummy_cube.data = masked_vector.reshape(self.xy_shape)
        dummy_cube.rename(cube_name)
        dummy_cube.units = cube_unit
        return dummy_cube

    def ps2006_kks2011_model(
        self,
        xy_point: int,
        matern_ellipse: MaternEllipseModel,
        max_distance: float = 20.0,
        min_distance: float = 0.3,
        delta_x_method: str = "Modified_Met_Office",
        guesses: list[float] | None = None,
        bounds: list[tuple[float, ...]] | None = None,
        opt_method: str = "Nelder-Mead",
        tol: float = 0.001,
        estimate_SE: str | None = None,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> dict[str, Any]:
        """
        Fit ellipses/covariance models using adhoc local covariances

        the form of the covariance model depends on "fform"
        isotropic (radial distance only)
        anistropic (x and y are different, but not rotated)
        anistropic_rotated (rotated)

        add _pd to fform uses phyiscal distances instead of degrees
        without _pd, estimation uses degree lat lons

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
            "Spherical_COS_Law": uses COS(C) = COS(A)COS(B)
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
            region, where anomalies are smooth over very large
            (~10000km) scales.

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

        estimate_SE=None
            The code can estimate the standard error if the Matern parameters.
            This is not usually used or discussed for the purpose of kriging.
            Certain opt_method (gradient descent) can do this automatically
            using Fisher Info for certain covariance funciton,
            but is not possible for some nasty functions (aka Bessel
            func) gets involved nor it is possible for some optimisers
            (such as Nelder-Mead).
            The code does it using bootstrapping.

        n_jobs=_default_n_jobs: int
            If parallel processing, number of threads to use.

        Returns
        -------
        ans : dictionary
            Dictionary with results of the fit
            and the observed correlation matrix
        """
        ##
        lonlat = self.xy[xy_point]
        correlation_vector = self.Corr[xy_point, :]
        ##
        R2 = self.data_cube[0].copy()
        R2x = self._reverse_mask_from_compress_1d(self.Corr[xy_point, :])
        R2.data = R2x.reshape(self.xy_shape)
        R2.units = "1"
        ##
        # dx and dy are in degrees
        dx = np.array([a - lonlat[0] for a in self.xy[:, 0]])
        dy = np.array([a - lonlat[1] for a in self.xy[:, 1]])
        dx[dx > 180.0] -= 360.0
        dx[dx < -180.0] += 360.0
        # Delete the origin (can't have dx = dy = 0)
        dx = np.delete(dx, xy_point)
        dy = np.delete(dy, xy_point)
        correlation_vector2 = np.delete(correlation_vector, xy_point)
        ##
        # distance is in delta-degrees
        lonlat_vector = np.column_stack([dx, dy])
        distance = linalg.norm(lonlat_vector, axis=1)
        # distance_i = np.abs(dx)
        # distance_j = np.abs(dy)
        distance_i = dx
        distance_j = dy
        dx_sign = np.sign(dx)
        ##
        if matern_ellipse.physical_distance:
            # There are two ways to do that
            # (1) Use the Haversine formula and solve for distance_ii
            # (2) Use the law of cosines, taking inputs of distance_ii and
            # distance_jj and solve for radial distance <-- this retains the up
            # and down....

            # Haversine -- (delta_lat, delta_lon)
            # sklearn.metrics.pairwise.haversine_distances(X, Y=None)[source]
            # Compute the Haversine distance between samples in X and Y.
            # The Haversine (or great circle) distance is the angular distance
            # between two points on the surface of a sphere.
            # The first coordinate of each point is assumed to be the latitude,
            # the second is the longitude, given in radians.
            # The dimension of the data must be 2.
            latlon_vector2 = np.column_stack(
                [np.deg2rad(lonlat[1] + dy), np.deg2rad(lonlat[0] + dx)]
            )
            latlon2 = np.array([np.deg2rad(lonlat[1]), np.deg2rad(lonlat[0])])
            latlon2 = latlon2[np.newaxis, :]
            X_train_radial = haversine_distances(latlon_vector2, latlon2)[:, 0]
            distance_jj = np.deg2rad(distance_j)
            #
            # Law of cosines/Pyth Theroem on a sphere surface:
            # https://sites.math.washington.edu/~king/coursedir/m445w03/class/02-03-lawcos-answers.html
            # COS(Haversine Dist) = COS(Delta_Lat) COS(Delta_X)
            #
            # Note:
            # Delta_X != Delta_LON or Delta_Lon x COS LAT
            # Delta_X itself is a great circle distance
            # Here meridonal displacement is always defined relative
            # to the north and south pole!
            #
            if delta_x_method == "Spherical_COS_Law":
                # This doesn't appears to work... nrecheck needed
                inside_arccos = np.cos(X_train_radial) / np.cos(distance_jj)
                print("Check, num of inside_arccos vals = ", len(inside_arccos))
                print(
                    "Check, num of abs(inside_arccos) > 1.0 = ",
                    np.sum(np.abs(inside_arccos) > 1.0),
                )
                print(
                    "Check, max(inside_arccos): max(inside_arccos) = ",
                    inside_arccos.max(),
                )
                # Numerical issues may lead to numbers slightly greater than 1.0
                # or less than -1.0
                inside_arccos[inside_arccos > 1.0] = 1.0
                inside_arccos[inside_arccos < -1.0] = -1.0
                distance_ii = dx_sign * np.arccos(inside_arccos)
            elif delta_x_method == "Met_Office":
                # Cylindrical approximation
                distance_ii = np.deg2rad(dx)
            elif delta_x_method == "Modified_Met_Office":
                average_cos = 0.5 * (
                    np.cos(np.deg2rad(lonlat[1] + dy))
                    + np.cos(np.deg2rad(lonlat[1]))
                )
                distance_ii = np.deg2rad(dx) * average_cos
            else:
                raise ValueError("Unknown delta_x_method")
            ##
            # Converts dx and dy to physical distance (km)
            X_train_directional = np.column_stack(
                [
                    distance_ii * RADIUS_OF_EARTH_KM,
                    distance_jj * RADIUS_OF_EARTH_KM,
                ]
            )  # noqa: E501
            X_train_radial = X_train_radial * RADIUS_OF_EARTH_KM
        else:
            X_train_directional = np.column_stack([distance_i, distance_j])
            X_train_radial = distance.copy()
        y_train = correlation_vector2.copy()
        ##
        print("Calculation check for X_train_directional")
        print(X_train_directional.shape)
        print(
            "i-th component range, min, max: ",
            np.ptp(X_train_directional[:, 0]),
            np.min(X_train_directional[:, 0]),
            np.max(X_train_directional[:, 0]),
        )
        print(
            "j-th component range, min, max: ",
            np.ptp(X_train_directional[:, 1]),
            np.min(X_train_directional[:, 1]),
            np.max(X_train_directional[:, 1]),
        )
        distance_limit = np.where(distance > max_distance)[0].tolist()
        distance_threshold = np.where(distance < min_distance)[0].tolist()
        xys_2_drop = list(set(distance_limit + distance_threshold))
        X_train_directional = np.delete(X_train_directional, xys_2_drop, axis=0)
        X_train_radial = np.delete(X_train_radial, xys_2_drop, axis=0)
        y_train = np.delete(y_train, xys_2_drop)
        ##
        if matern_ellipse.anisotropic:
            X_train = X_train_directional
        else:
            X_train = X_train_radial.reshape(-1, 1)
        ##
        results, _, bbs = matern_ellipse.fit(
            X_train,
            y_train,
            guesses=guesses,
            bounds=bounds,
            opt_method=opt_method,
            tol=tol,
            estimate_SE=estimate_SE,
            n_jobs=n_jobs,
        )
        ##
        if matern_ellipse.unit_sigma:
            model_params = results.x.tolist()
            stdev = None
        else:
            model_params = results.x.tolist()[:-1]
            stdev = results.x.tolist()[-1]
        # Meaning of fit_success (int)
        # 0: success
        # 1: success but with one parameter reaching lower bounadries
        # 2: success but with one parameter reaching upper bounadries
        # 3: success with multiple parameters reaching the boundaries
        #    (aka both Lx and Ly), can be both at lower or upper boundaries
        # 9: fail, probably due to running out of maxiter
        #    (see scipy.optimize.minimize kwargs "options)"
        if results.success:
            fit_success: int = 0
            for model_param, bb in zip(model_params, bbs):
                left_check = math.isclose(model_param, bb[0], rel_tol=0.01)
                right_check = math.isclose(model_param, bb[1], rel_tol=0.01)
                left_advisory = (
                    "near_left_bnd" if left_check else "not_near_left_bnd"
                )
                right_advisory = (
                    "near_right_bnd" if right_check else "not_near_rgt_bnd"
                )
                print(
                    "Convergence success after ",
                    results.nit,
                    " iterations :) : ",
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
            print("RMSE of multivariate norm fit = ", stdev)
        else:
            print("Convergence fail after ", results.nit, " iterations.")
            print(model_params)
            fit_success = 9
        print("QC flag = ", fit_success)
        # append standard deviation
        model_params.append(np.sqrt(self.Cov[xy_point, xy_point] / self.time_n))
        model_params.append(fit_success)
        model_params.append(results.nit)
        ##
        v_coord = make_v_aux_coord(matern_ellipse.v)
        template_cube = self._make_template_cube(xy_point)
        model_as_cubelist = create_output_cubes(
            template_cube,
            model_type=matern_ellipse.model_type,
            additional_meta_aux_coords=[v_coord],
            default_values=model_params,
        )["param_cubelist"]
        ##
        return {
            "Correlation": R2,
            "MaternObj": matern_ellipse,
            "Model": results,
            "Model_Type": matern_ellipse.model_type,
            "Model_as_1D_cube": model_as_cubelist,
        }

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

        a = self.xy_full if use_full else self.xy
        idx = int(((a[:, 0] - lon) ** 2.0 + (a[:, 1] - lat) ** 2.0).argmin())
        return idx, a[idx, :]

    def _xy_2_xy_full_index(self, xy_point: int) -> int:
        """
        Given xy index in that corresponding to a latlon
        in the covariance (masked value ma.MaskedArray compressed),
        what is its index with masked values (i.e. ndarray flatten)
        """
        return int(
            np.argwhere(
                np.all((self.xy_full - self.xy[xy_point, :]) == 0, axis=1)
            )[0]
        )

    def _make_template_cube(self, xy_point: int) -> iris.cube.Cube:
        """Make a template cube for lat lon corresponding to xy_point index"""
        xy = self.xy[xy_point, :]
        return self._make_template_cube2(xy)
        # t_len = len(self.data_cube.coord('time').points)
        # template_cube = self.data_cube[t_len // 2].intersection(
        #     longitude=(xy[0] - 0.05, xy[0] + 0.05),
        #     latitude=(xy[1] - 0.05, xy[1] + 0.05),
        # )
        # return template_cube

    def _make_template_cube2(self, lonlat: np.ndarray) -> iris.cube.Cube:
        """Make a template cube for lat lon"""
        t_len = len(self.data_cube.coord("time").points)
        template_cube = self.data_cube[t_len // 2].intersection(
            longitude=(lonlat[0] - 0.05, lonlat[0] + 0.05),
            latitude=(lonlat[1] - 0.05, lonlat[1] + 0.05),
        )
        return template_cube

    def __str__(self):
        return str(self.__class__)
        # return str(self.__class__) + ": " + str(self.__dict__)


def sigma_rot_func(
    Lx: float,
    Ly: float,
    theta: float | None,
) -> np.ndarray:
    """
    Equation 15 in Karspeck el al 2011 and Equation 6
    in Paciorek and Schervish 2006,
    assuming Sigma(Lx, Ly, theta) locally/moving-window invariant or
    we have already taken the mean (Sigma overbar, PP06 3.1.1)
    """
    # sigma is a 2x2 matrix
    if theta is None:
        theta = 0.0
    r = R.from_rotvec([0, 0, theta])
    r_matrix_2d = r.as_matrix()[:2, :2]
    sigma = np.matmul(
        np.matmul(r_matrix_2d, np.diag([Lx**2, Ly**2])),
        np.transpose(r_matrix_2d),
    )
    return sigma


def mahal_dist_func_rot(
    delta_x: float,
    delta_y: float,
    Lx: float,
    Ly: float,
    theta: float | None = None,
) -> float:
    """
    Calculate tau if Lx, Ly, theta is known (aka, this takes the additional step
    to compute sigma)
    This is needed for MLE estimation of Lx, Ly, and theta

    for MLE:
    d is distance between two points
    Lx, Ly, theta are unknown parameters that need to be estimated
    and replaces d/rou in equation 14 in Karspect et al paper

    Parameters
    ----------
    x_i, x_j : float
        displacement to remote point as in: (x_i) i + (x_j) j in old school
        vector notation
    Lx, Ly : float
        Lx, Ly scale (km or degrees)
    theta : float
        rotation angle (RADIANS ONLY!!!!!)

    Returns
    -------
    tau : float
        Mahalanobis distance
    """
    # sigma is 4x4 matrix  QUESTION: 2x2?
    if theta is not None:
        sigma = sigma_rot_func(Lx, Ly, theta)
    else:
        sigma = np.diag(np.array([Lx**2.0, Ly**2.0]))
    xi_minus_xj = np.array([delta_x, delta_y])
    tau_inside_squareroot = np.matmul(
        np.matmul(np.transpose(xi_minus_xj), linalg.inv(sigma)), xi_minus_xj
    )
    tau = np.sqrt(tau_inside_squareroot)
    logging.debug(f"{tau = }")
    return tau  # tau is a scalar


def mahal_dist_func_sigma(
    delta_x: float,
    delta_y: float,
    sigma: float,
) -> float:
    """
    Calculate tau directly if sigma is already known
    Useful for sigma_bar computations
    """
    xi_minus_xj = np.array([delta_x, delta_y])
    tau_inside_squareroot = np.matmul(
        np.matmul(np.transpose(xi_minus_xj), linalg.inv(sigma)), xi_minus_xj
    )
    tau = np.sqrt(tau_inside_squareroot)
    logging.debug(f"{tau = }")
    return tau


def mahal_dist_func(
    delta_x: float,
    delta_y: float,
    Lx: float,
    Ly: float,
) -> float:
    """
    Compute tau for non-rotational case;
    unlike mahal_dist_func_sigma, sigma is not needed
    """
    return mahal_dist_func_rot(delta_x, delta_y, Lx, Ly, theta=None)


def make_v_aux_coord(v: float) -> icoords.AuxCoord:
    """Create an iris coord for the Matern (positive) shape parameter"""
    return icoords.AuxCoord(v, long_name="v_shape", units="no_unit")


def c_ij_anistropic_rotated(
    v: float,
    stdev: float,
    x_i: float,
    x_j: float,
    Lx: float,
    Ly: float,
    theta: float | None,
    stdev_j: float | None = None,
) -> float:
    """
    Covariance structure between base point i and j
    Assuming local stationarity or slowly varing
    so that some terms in PS06 drops off (like Sigma_i ~ Sigma_j instead of
    treating them as different) (aka second_term below)
    this makes formulation a lot simplier
    We let stdev_j opens to changes,
    but in pracitice, we normalise everything to correlation so
    stdev == stdev_j == 1

    Parameters
    ----------
    v : float
        Matern shape parameter
    stdev : float, standard deviation, local point
    x_i, x_j : float
        displacement to remote point as in: (x_i) i + (x_j) j in old school
        vector notation
    Lx, Ly : float
        Lx, Ly scale (km or degrees)
    theta : float
        rotation angle (RADIANS ONLY!!!!!)
    stdev_j : float
        standard deviation, remote point

    Returns
    -------
    ans : float
        Covariance/correlation between local and remote point given displacement
        and Matern covariance parameters
    """
    stdev_j = stdev_j or stdev

    # sigma = sigma_rot_func(Lx, Ly, theta)
    tau = mahal_dist_func_rot(x_i, x_j, Lx, Ly, theta=theta)

    first_term = (stdev * stdev_j) / (gamma(v) * (2.0 ** (v - 1)))
    # If data is assumed near stationary locally, sigma_i ~ sigma_j same
    # making (sigma_i)**1/4 (sigma_j)**1/4 / (mean_sigma**1/2) = 1.0
    # Treating it the otherwise is a major escalation to the computation
    # See discussion 2nd paragraph in 3.1.1 in Paciroke and Schervish 2006
    # second_term = 1.0
    third_term = (2.0 * tau * np.sqrt(v)) ** v
    forth_term = modified_bessel_2nd(v, 2.0 * tau * np.sqrt(v))
    ans = first_term * third_term * forth_term
    # ans = first_term * second_term * third_term * forth_term

    logging.debug(f"{first_term = }, {first_term.shape = }")
    logging.debug(f"{third_term = }, {third_term.shape = }")
    logging.debug(f"{forth_term = }, {forth_term.shape = }")
    logging.debug(f"{ans = }, {ans.shape = }")
    return ans


def c_ij_anistropic_unrotated(
    v: float,
    stdev: float,
    x_i: float,
    x_j: float,
    Lx: float,
    Ly: float,
    stdev_j: float | None = None,
) -> float:
    """Alias for non-rotated version of c_ij_anistropic_rotated"""
    return c_ij_anistropic_rotated(
        v, stdev, x_i, x_j, Lx, Ly, theta=None, stdev_j=stdev_j
    )


def c_ij_istropic(
    v: float,
    stdev: float,
    displacement: float,
    R: float,
    stdev_j: float | None = None,
) -> float:
    """
    Isotropic version of c_ij_anistropic_rotated

    Parameters
    ----------
    v : float
        Matern shape parameter
    stdev : float
        standard deviation, local point
    displacement : float
        displacement to remote point
    R : float
        range parameter (km or degrees)
    stdev_j : float
        standard deviation, remote point

    Returns
    -------
    ans : float
        Covariance/correlation between local and remote point given displacement
        and Matern covariance parameters
    """
    stdev_j = stdev_j or stdev

    tau = np.abs(displacement) / R

    first_term = (stdev * stdev_j) / (gamma(v) * (2.0 ** (v - 1)))
    third_term = (2.0 * tau * np.sqrt(v)) ** v
    forth_term = modified_bessel_2nd(v, 2.0 * tau * np.sqrt(v))
    ans = first_term * third_term * forth_term
    return ans


class MLE_c_ij_Builder_Karspeck:
    """
    The class that contains variogram/ellipse fitting methods and parameters

    This class assumes your input to be a standardised correlation matrix
    They are easier to handle because stdevs in the covariance function become 1

    Parameters
    ----------
    v : float
        Matern shape parameter
    fform : str
        See dict at the top of this code to see valid options
        This determines number of parameters to estimated
    standardised_cov_matrix : bool
        If you set this to False, it will raise an error
        See docstring for class, this expect correlation matrices only!
    unit_sigma : bool
        See ps2006_kks2011_model above
        Basically do you want the stdev related to variogram fit deivations
        to be estimated; this is usually True (the deviations are ignored),
        and does not affect the results
    """

    def __init__(
        self,
        ellipse_model: MaternEllipseModel,
        standardised_cov_matrix: bool = True,
        unit_sigma: bool = True,
    ) -> None:
        if not standardised_cov_matrix:
            raise NotImplementedError(
                "Standardised/normalise covariance matrix first to "
                + "correlation matrix"
            )

        self.fform = ellipse_model.fform
        self.unit_sigma = unit_sigma

        match self.fform:
            case "isotropic":
                self.n_params = 1
                self.default_guesses = [7.0]
                self.default_bounds = [
                    (0.5, 50.0),
                ]
                self.c_ij = lambda X, rou: c_ij_istropic(v, 1, X, rou)
            case "isotropic_pd":
                self.n_params = 1
                self.default_guesses = [deg_to_km(7.0)]
                self.default_bounds = [
                    (deg_to_km(0.5), deg_to_km(50)),
                ]
                self.c_ij = lambda X, rou: c_ij_istropic(v, 1, X, rou)
            case "anistropic":
                self.n_params = 2
                self.default_guesses = [7.0, 7.0]
                self.default_bounds = [(0.5, 50.0), (0.5, 30.0)]
                self.c_ij = lambda X, Lx, Ly: c_ij_anistropic_unrotated(
                    v, 1, X[0], X[1], Lx, Ly
                )
            case "anistropic_pd":
                self.n_params = 2
                self.default_guesses = [deg_to_km(7.0), deg_to_km(7.0)]
                self.default_bounds = [
                    (deg_to_km(0.5), deg_to_km(50)),
                    (deg_to_km(0.5), deg_to_km(30)),
                ]
                self.c_ij = lambda X, Lx, Ly: c_ij_anistropic_unrotated(
                    v, 1, X[0], X[1], Lx, Ly
                )
            case "anistropic_rotated":
                self.n_params = 3
                self.default_guesses = [7.0, 7.0, 0.0]
                self.default_bounds = [
                    (0.5, 50.0),
                    (0.5, 30.0),
                    (-2.0 * np.pi, 2.0 * np.pi),
                ]
                self.c_ij = lambda X, Lx, Ly, theta: c_ij_anistropic_rotated(
                    v, 1, X[0], X[1], Lx, Ly, theta
                )
            case "anistropic_rotated_pd":
                self.n_params = 3
                self.default_guesses = [deg_to_km(7.0), deg_to_km(7.0), 0.0]
                self.default_bounds = [
                    (deg_to_km(0.5), deg_to_km(50)),
                    (deg_to_km(0.5), deg_to_km(30)),
                    (-2.0 * math.pi, 2.0 * math.pi),
                ]
                self.c_ij = lambda X, Lx, Ly, theta: c_ij_anistropic_rotated(
                    v, 1, X[0], X[1], Lx, Ly, theta
                )
            case _:
                raise ValueError(f"Unknown fform value {fform}")

        return None

    def negativeloglikelihood(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: tuple[float, ...],
        arctanh_transform: bool = True,
        backend: str = DEFAULT_BACKEND,
        n_jobs: int = DEFAULT_N_JOBS,
    ):
        """
        Compute the negative log-likelihood given observed X independent
        observations (displacements) and y dependent variable (the observed
        correlation), and Matern parameters params. Namely does the Matern
        covariance function using params, how close it explains the observed
        displacements and correlations.

        log(LL) = SUM (f (y,x|params) )
        params = Maximise (log(LL))
        params = Minimise (-log(LL)) which is how usually the computer solves it
        assuming errors of params are normally distributed

        There is a hidden scale/standard deviation in
        stats.norm.logpdf(scale, which defaults to 1)
        but since we have scaled our values to covariance to correlation (and
        even used Fisher transform) as part of the function, it can be dropped

        Otherwise, you need to have stdev as the last value of params, and
        should be set to the scale parameter

        Parameters
        ----------
        X : np.ndarray
            Observed displacements
        y : np.ndarray
            Observed correlation
        params : tuple of Matern parameters
            (in the current optimize iteration) or if you want to
            compute the actual negative log-likelihood
        arctanh_transform : bool
            Should the Fisher (arctanh) transform be used
            This is usually option, but it does make the computation
            more stable if they are close to 1 (or -1; doesn't apply here)
        backend : str
            joblib backend
            See https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        n_jobs : int
            Number of threads/parallel computation

        Returns
        -------
        nLL : float
            The negative log likelihood
        """
        if self.n_params == 1:
            rou = params[0]
            if self.unit_sigma:
                sigma = 1
            else:
                sigma = params[1]
            y_LL = self.c_ij(X, rou)
        elif self.n_params == 2:
            Lx = params[0]
            Ly = params[1]
            if self.unit_sigma:
                sigma = 1
            else:
                sigma = params[2]
            y_LL = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(self.c_ij)(X[n_x_j, :], Lx, Ly)
                for n_x_j in range(X.shape[0])
            )
            # y_LL = []
            # for n_x_j in range(X.shape[0]):
            #    y_LL.append(self.c_ij(X[n_x_j, :], Lx, Ly))
        elif self.n_params == 3:
            Lx = params[0]
            Ly = params[1]
            theta = params[2]
            if self.unit_sigma:
                sigma = 1
            else:
                sigma = params[3]
            y_LL = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(self.c_ij)(X[n_x_j, :], Lx, Ly, theta)
                for n_x_j in range(X.shape[0])
            )
            # y_LL = []
            # for n_x_j in range(X.shape[0]):
            #    y_LL.append(self.c_ij(X[n_x_j,:], Lx, Ly, theta))
        else:
            raise ValueError("Unexpected length of self.n_params.")
        y_LL = np.array(y_LL)
        # if y is correlation,
        # it might be useful to Fisher transform them before plugging into
        # norm.logpdf this affects values close to 1 and -1
        # imposing better behavior to the differences at the tail
        if arctanh_transform:
            #
            # Warning against arctanh(abs(y) > 1); (TO DO: Add correction later)
            arctanh_threshold = 0.999999
            # arctanh_threshold = 1.0
            max_abs_y = np.max(np.abs(y))
            max_abs_yLL = np.max(np.abs(y_LL))
            if max_abs_y >= arctanh_threshold:
                warn_msg = "abs(y) >= " + str(arctanh_threshold) + " detected; "
                warn_msg += "fudged to threshold; max(abs(y))=" + str(max_abs_y)
                warnings.warn(warn_msg, RuntimeWarning)
                y[np.abs(y) > arctanh_threshold] = (
                    np.sign(y[np.abs(y) > arctanh_threshold])
                    * arctanh_threshold
                )
                # y[np.abs(y) > 1] = np.sign(y[np.abs(y) > 1]) * 0.9999
            # if np.any(np.isclose(np.abs(y), 1.0)):
            #     warn_msg = (
            #         "abs(y) is close to 1; max(abs(y))="
            #         + str(max_abs_y)
            #     )
            #     warnings.warn(warn_msg, RuntimeWarning)
            #     y[np.isclose(np.abs(y), 1.0)] = (
            #         np.sign(y[np.isclose(np.abs(y), 1.0)]) * 0.9999
            #     )
            #
            if max_abs_yLL >= 1:
                warn_msg = (
                    "abs(y_LL) >= " + str(arctanh_threshold) + " detected; "
                )
                warn_msg += "fudged to threshold; max(abs(y_LL))=" + str(
                    max_abs_yLL
                )
                warnings.warn(warn_msg, RuntimeWarning)
                y_LL[np.abs(y_LL) > arctanh_threshold] = (
                    np.sign(y_LL[np.abs(y_LL) > arctanh_threshold])
                    * arctanh_threshold
                )
                # y_LL[np.abs(y_LL) > 1] = (
                #     np.sign(y_LL[np.abs(y_LL) > 1]) * 0.9999
                # )
            # if np.any(np.isclose(np.abs(y_LL), 1.0)):
            #     warn_msg = (
            #         "abs(y_LL) close to 1 detected; max(abs(y_LL))="
            #         + str(max_abs_yLL)
            #     )
            #     warnings.warn(warn_msg, RuntimeWarning)
            #     y_LL[np.isclose(np.abs(y_LL), 1.0)] = (
            #         np.sign(y_LL[np.isclose(np.abs(y_LL), 1.0)]) * 0.9999
            #     )
            #
            nLL = -1.0 * np.sum(
                stats.norm.logpdf(
                    np.arctanh(y), loc=np.arctanh(y_LL), scale=sigma
                )
            )
        else:
            nLL = -1.0 * np.sum(stats.norm.logpdf(y, loc=y_LL, scale=sigma))
        return nLL

    def negativeloglikelihood_function(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = DEFAULT_N_JOBS,
        backend: str = DEFAULT_BACKEND,
    ):
        """Creates a function that can be fed into scipy.optimizer.minimize"""

        def f(params):
            return self.negativeloglikelihood(
                X, y, params, n_jobs=n_jobs, backend=backend
            )

        return f

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        guesses: list[float] | None = None,
        bounds: list[tuple[float, ...]] | None = None,
        opt_method: str = "Nelder-Mead",
        tol: float | None = None,
        estimate_SE: str | None = "bootstrap_parallel",
        n_sim: int = 500,
        n_jobs: int = DEFAULT_N_JOBS,
        backend: str = DEFAULT_BACKEND,
        random_seed: int = 1234,
    ) -> list:
        """
        Default solver in Nelder-Mead as used in the Karspeck paper
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
        default max-iter is 200 x (number_of_variables)
        for 3 variables (Lx, Ly, theta) --> 200x3 = 600
        note: unlike variogram fitting, no nugget, no sill, and no residue
        variance (normalised data but Fisher transform needed?)
        can be adjusted using "maxiter" within "options" kwargs

        Much of the variable names are defined the same way as earlier

        Parameters
        ----------
        X, y : np.ndarray
            distances and observed correlations
        guesses=None :
            Tuples/lists of initial values to scipy.optimize.minimize
        bounds=None :
            Tuples/lists of bounds for fitted parameters
        opt_method='Nelder-Mead' : str
            scipy.optimize.minimize optimisation method
        tol=None : float
            scipy.optimize.minimize convergence tolerance
        estimate_SE='bootstrap_parallel' : str
            how to estimate standard error if needed
        n_sim=500 : int
            number of bootstrap to estimate standard error
        n_jobs=_default_n_jobs : int
            number of threads
        backend=_default_backend : str
            joblib backend
        random_seed=1234 (NEW): int, random seed for bootstrap

        Returns
        -------
        list : [fitted parameters,
                standard error of the fitted parameters,
                bounds of fitted parameters]
        """
        ##
        guesses = guesses or self.default_guesses
        bounds = bounds or self.default_bounds

        if not self.unit_sigma:
            guesses.append(0.1)
            bounds.append((0.0001, 0.5))
        ##
        LL_observedXy_unknownparams = self.negativeloglikelihood_function(
            X, y, n_jobs=n_jobs, backend=backend
        )
        ##
        print("X range: ", np.min(X), np.max(X))
        print("y range: ", np.min(y), np.max(y))
        zipper = zip(guesses, bounds)
        for g, b in zipper:
            print("init: ", g, "bounds: ", b)
        ##
        results = minimize(
            LL_observedXy_unknownparams,
            guesses,
            bounds=bounds,
            method=opt_method,
            tol=tol,
        )
        ##
        # This does not account for standard errors in the
        # correlation/covariance matrix!
        if estimate_SE is None:
            print("Standard error estimates not required")
            return [results, None, bounds]

        if estimate_SE == "bootstrap_serial":
            # Serial
            sim_params = []
            for looper in range(n_sim):
                sim_params.append(
                    self._bootstrap_once(
                        X,
                        y,
                        guesses,
                        bounds,
                        opt_method,
                        tol=tol,
                        seed=random_seed + looper,
                    )
                )
            sim_params = np.array(sim_params)
            SE = np.std(sim_params, axis=0)
        elif estimate_SE == "bootstrap_parallel":
            # Parallel
            # On JASMIN Jupyter: n_jobs = 5 leads to 1/3 wallclock time
            kwargs_0 = {"n_jobs": n_jobs, "backend": backend}
            workers = range(n_sim)
            sim_params = Parallel(**kwargs_0)(
                delayed(self._bootstrap_once)(
                    X,
                    y,
                    guesses,
                    bounds,
                    opt_method,
                    tol=tol,
                    seed=random_seed + worker,
                )
                for worker in workers
            )
            sim_params = np.array(sim_params)
            SE = np.std(sim_params, axis=0)
        elif estimate_SE == "hessian":
            # note: autograd does not work with scipy's Bessel functions
            raise NotImplementedError(
                "Second order deriviative (Hessian) of "
                + "Fisher Information not implemented"
            )
        else:
            raise ValueError("Unknown estimate_SE")
        return [results, SE, bounds]

    def _bootstrap_once(
        self,
        X: np.ndarray,
        y: np.ndarray,
        guesses: list[float],
        bounds: list[tuple[float, ...]],
        opt_method: str,
        tol: float | None = None,
        seed: int = 1234,
    ):
        """Bootstrap refit the Matern parameters"""
        rng = np.random.RandomState(seed)
        len_obs = len(y)
        i_obs = np.arange(len_obs)
        bootstrap_i = rng.choice(i_obs, size=len_obs, replace=True)
        X_boot = X[bootstrap_i, ...]
        y_boot = y[bootstrap_i]
        LL_bootXy_simulatedparams = self.negativeloglikelihood_function(
            X_boot, y_boot
        )
        ans = minimize(
            LL_bootXy_simulatedparams,
            guesses,
            bounds=bounds,
            method=opt_method,
            tol=tol,
        )
        return ans.x


def create_output_cubes(
    template_cube,
    model_type: MODEL_TYPE = "ps2006_kks2011_iso",
    default_values: list[float] | None = None,
    additional_meta_aux_coords: list | None = None,
    dtype=np.float32,
) -> dict:
    """
    For data presentation, create template iris cubes to insert data

    Parameters
    ----------
    template_cube : iris.cube.Cube
        A cube to be copied as the template
    model_type : MODEL_TYPE
        See dict at top, string to add auxcoord for outputs
    default_values : list[float]
        Default values to be put in cube, not neccessary a masked/fill value
    additional_meta_aux_coords : list of icoords.AuxCoord
        Whatever additional auxcoord metadata you want to add
    dtype=np.float32 : numpy number type
        The number type to be used in the cube usually np.float32 or
        np.float(64)

    Returns
    -------
    dictionary:
        "param_cubelist"
            Instance of iris.cube.CubeList() that contains cubes to be filled in
        "param_names"
            the names of the variable that get added to param_cubelist
    """
    default_values = default_values or [-999.9, -999.9, -999.9]

    supercategory = MODEL_TYPE_TO_SUPERCATEGORY[model_type]
    params_dict = SUPERCATEGORY_PARAMS[supercategory]

    model_type_coord = icoords.AuxCoord(
        model_type, long_name="fitting_model", units="no_unit"
    )
    supercategory_coord = icoords.AuxCoord(
        supercategory,
        long_name="supercategory_of_fitting_model",
        units="no_unit",
    )

    ans_cubelist = iris.cube.CubeList()
    ans_paramlist = []

    for param, default_value in zip(params_dict, default_values):
        ans_paramlist.append(param)
        param_cube = template_cube.copy()
        param_cube.rename(param)
        param_cube.long_name = param
        param_cube.add_aux_coord(model_type_coord)
        param_cube.add_aux_coord(supercategory_coord)
        param_cube.units = params_dict[param]
        if additional_meta_aux_coords is not None:
            for add_coord in additional_meta_aux_coords:
                param_cube.add_aux_coord(add_coord)
        if param_cube.ndim == 0:
            param_cube = iutil.new_axis(
                iutil.new_axis(param_cube, "longitude"), "latitude"
            )
        param_cube.data[:] = default_value
        param_cube.data = param_cube.data.astype(dtype)
        if ma.isMaskedArray(template_cube.data):
            param_cube.data.mask = template_cube.data.mask
        ans_cubelist.append(param_cube)
    return {"param_cubelist": ans_cubelist, "param_names": ans_paramlist}


def main():
    """Main - keep calm and does nothing"""
    print("=== Main ===")
    # _test_load()


if __name__ == "__main__":
    main()
