"""Functions for helping with perturbations/random drawing"""

import numpy as np
from scipy import stats
import logging

from glomar_gridding.kriging import (
    Kriging,
    adjust_small_negative,
    _extended_inverse,
)


class StochasticKriging(Kriging):
    """DOCS"""

    def set_simple_kriging_weights(
        self,
        simple_kriging_weights: np.ndarray,
    ) -> None:
        """DOCS"""
        self.simple_kriging_weights = simple_kriging_weights

    def get_kriging_weights(
        self, idx: np.ndarray, error_cov: np.ndarray | None = None
    ) -> None:
        """DOCS"""
        obs_obs_cov = self.covariance[idx[:, None], idx[None, :]]

        # Add error covariance
        if error_cov is not None:
            if error_cov.shape[0] != len(idx):
                error_cov = error_cov[idx[:, None], idx[None, :]]
            obs_obs_cov += error_cov

        obs_obs_cov_inv = np.linalg.inv(obs_obs_cov)
        self.kriging_weights_from_inverse(obs_obs_cov_inv, idx)

        return None

    def kriging_weights_from_inverse(
        self,
        inv: np.ndarray,
        idx: np.ndarray,
    ) -> None:
        """DOCS"""
        if len(idx) != inv.shape[0] - 1:
            raise ValueError("inv must be square with side length == len(idx)")
        obs_grid_cov = self.covariance[idx, :]
        M = self.covariance.shape[0]

        self.simple_kriging_weights = (inv @ obs_grid_cov).T

        # Add Lagrange multiplier
        obs_obs_cov_inv = _extended_inverse(inv)
        obs_grid_cov = np.concatenate((obs_grid_cov, np.ones((1, M))), axis=0)
        self.kriging_weights = (obs_obs_cov_inv @ obs_grid_cov).T

        return None

    def get_uncertainty(self) -> np.ndarray:
        """
        Compute the kriging uncertainty. This requires the attribute
        `kriging_weights` to be computed.

        Returns
        -------
        uncert : numpy.ndarray
            The Kriging uncertainty.
        """
        if not hasattr(self, "kriging_weights"):
            raise KeyError("Please compute Kriging Weights first")

        dz_squared = np.diag(self.covariance - self.kriging_weights)
        dz_squared = adjust_small_negative(dz_squared)
        uncert = np.sqrt(dz_squared)
        uncert[np.isnan(uncert)] = 0.0
        return uncert

    def constraint_mask(
        self,
        idx: np.ndarray,
    ) -> np.ndarray:
        r"""
        Compute the observational constraint mask (A14 in Morice et al. (2021) -
        10.1029/2019JD032361) to determine if a grid point should be
        masked/weights modified by how far it is to its near observed point

        Note: typo in Section A4 in Morice et al 2021 (confired by authors).

        Equation to use is A14 is incorrect. Easily noticeable because
        dimensionally incorrect is wrong, but the correct answer is easy to
        figure out.

        Correct Equation (extra matrix inverse for :math:`K_{obs} + E`):
        .. math::
            1 - diag\\(K - K_{cross}^T @ (K + E)^{-1} @ K_{cross}\\)  / diag(K)
            < alpha

        This can be re-written as:
        .. math::
            diag\\(K_{cross}^T @ (K_{obs} + E)^{-1} @ K_{cross}\\) / diag(K)
            < alpha

        alpha is chosen to be 0.25 in the UKMO paper

        Written by S. Chan, modified by J. Siddons.

        This requires the Kriging weights from simple Kriging. If these are not
        set then please set them or compute them.

        Parameters
        ----------
        idx : numpy.ndarray
            The 1d indices of observation grid points. These values should be
            between 0 and (N * M) - 1 where N, M are the number of longitudes
            and latitudes respectively. Note that these values should also be
            computed using "C" ordering in numpy reshaping. They can be
            computed from a grid using glomar_gridding.grid.map_to_grid. Each
            value should only appear once. Points that contain more than 1
            observation should be averaged. Used to compute the Kriging weights.

        Returns
        -------
        constraint_mask : numpy.ndarray
            Constraint mask values, the left-hand-side of equation A14 from
            Morice et al. (2021). This is a vector of length `k_obs.size[0]`.

        Reference
        ---------
        Morice et al. (2021) : https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019JD032361
        """
        if not hasattr(self, "simple_kriging_weights"):
            raise KeyError("Please set kriging weights")

        numerator = np.diag(
            self.covariance[:, idx] @ self.simple_kriging_weights.T
        )
        denominator = np.diag(self.covariance)
        return np.divide(numerator, denominator)

    def solve(
        self,
        grid_obs: np.ndarray,
        idx: np.ndarray,
        error_cov: np.ndarray | None = None,
        simulated_state: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""
        Solves the combined Stochastic Kriging problem. Computes the Kriging
        weights if the `kriging_weights` attribute is not already set.

        Stochastic Kriging is a combined Ordinary and Simple Kriging approach.
        First a gridded field is generated for the observations using Ordinary
        Kriging. Secondly, a simulated gridded field is generated using Simple
        Kriging from a simulated state and simulated observations. The simulated
        state can be pre-computed or computed by this method. The simulated
        observations are drawn from the error covariance and located at the
        simulated state. The difference between the simulated gridded field and
        the simulated state is added to the gridded field.

        The solution to Kriging is:

        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross} \\times y

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points), and :math:`y` are the observation values.

        In this case, the :math:`K_{obs}`, :math:`K_{cross}` and are extended
        with a Lagrange multiplier term, ensuring that the Kriging weights are
        constrained to sum to 1. For the Ordinary Kriging component.

        The matrix :math:`K_{obs}` is extended by one row and one column, each
        containing the value 1, except at the diagonal point, which is 0. The
        :math:`K_{cross}` matrix is extended by an extra row containing values
        of 1. For the Ordinary Kriging component.

        Parameters
        ----------
        grid_obs : numpy.ndarray
            The observation values. If there are multiple observations in any
            grid box then these values need to be averaged into one value per
            grid box.
        idx : numpy.ndarray
            The 1d indices of observation grid points. These values should be
            between 0 and (N * M) - 1 where N, M are the number of longitudes
            and latitudes respectively. Note that these values should also be
            computed using "C" ordering in numpy reshaping. They can be
            computed from a grid using glomar_gridding.grid.map_to_grid. Each
            value should only appear once. Points that contain more than 1
            observation should be averaged. Used to compute the Kriging weights.
        error_cov : numpy.ndarray | None
            Optionally add error covariance values to the covariance between
            observation grid points. Used to compute Kriging weights.

        Returns
        -------
        numpy.ndarray
            The solution to the ordinary Kriging problem (as a Vector, this may
            need to be re-shaped appropriately as a post-processing step).
        """
        if not hasattr(self, "kriging_weights"):
            self.get_kriging_weights(idx, error_cov)

        if error_cov is None:
            raise ValueError(
                "Error Covariance must be set to draw simulated observations"
            )

        # Simulate a state from the covariance matrix
        if simulated_state is None:
            simulated_state = scipy_mv_normal_draw(
                loc=np.zeros(self.covariance.shape[0]),
                cov=self.covariance,
                ndraws=1,
            )

        # Simulate observations
        error_cov = error_cov[idx[:, None], idx[None, :]]
        simulated_obs = simulated_state + scipy_mv_normal_draw(
            loc=np.zeros(error_cov.shape[0]),
            cov=error_cov,
            ndraws=1,
        )

        # Simulate a gridded field
        self.simulated_grid = self.simple_kriging_weights @ simulated_obs
        self.epsilon = self.simulated_grid - simulated_state

        # Add Lagrange multiplier
        grid_obs = np.append(grid_obs, 0)
        self.gridded_field = self.kriging_weights @ grid_obs
        return self.gridded_field + self.epsilon


def scipy_mv_normal_draw(  # noqa: C901
    loc: np.ndarray,
    cov: np.ndarray,
    ndraws: int = 1,
    eigen_rtol: float = 1e-6,
    eigen_fudge: float = 1e-8,
) -> np.ndarray:
    """
    Do a random multivariate normal draw using
    scipy.stats.multivariate_normal.rvs

    numpy.random.multivariate_normal can also,
    but fixing seeds are more difficult using numpy

    This function has similar API as GP_draw with less kwargs.

    Warning/possible future scipy version may change this:
    It seems if one uses stats.Covariance, you have to have add [0] from rvs
    function. The above behavior applies to scipy v1.14.0

    Parameters
    ----------
    loc : float
        the location for the normal dry
    cov : numpy.ndarray
        not a xarray/iris cube! Some of our covariances are saved in numpy
        format and not netCDF files
    n_draws : int
        number of simulations, this is usually set to 1 except during
    unit testing
    eigen_rtol : float
        relative tolerance to negative eigenvalues
    eigen_fudge : float
        forced minimum value of eigenvalues if negative values are detected

    Returns
    -------
    draw : np.ndarray
        The draw(s) from the multivariate random normal distribution defined
        by the loc and cov parameters. If the cov parameter is not
        positive-definite then a new covariance will be determined by adjusting
        the eigen decomposition such that the modified covariance should be
        positive-definite.
    """

    def any_complex(arr: np.ndarray) -> bool:
        return bool(np.any(np.iscomplex(arr)))

    cov_shape = cov.shape
    if len(cov_shape) != 2:
        raise ValueError("cov should be 2D.")
    if cov_shape[0] != cov_shape[1]:
        raise ValueError("cov is not a square matrix")
    try:
        draw = np.random.multivariate_normal(loc, cov, size=ndraws)
        return draw[0] if ndraws == 1 else draw
    except np.linalg.LinAlgError:
        pass
    except Exception as e:
        raise e

    # Try to use eigen decomposition to generate a new covariance matrix that
    # would be positive-definite
    w, v = np.linalg.eigh(cov)
    w = np.real_if_close(w)
    v = np.real_if_close(v)
    if any_complex(w):
        raise ValueError("w is complex")
    if any_complex(v):
        raise ValueError("v is complex")
    if np.any(w < 0):
        most_neg_eigval = np.min(w)
        largest_eig_val = np.max(w)
        rtol_check = np.abs(most_neg_eigval) / largest_eig_val
        logging.warning(
            "Negative eigenvalues detected: largest = "
            + f"{largest_eig_val}; smallest = {most_neg_eigval}; "
            + f"ratio = {rtol_check}"
        )
        if rtol_check >= eigen_rtol:
            raise ValueError("Negative eigenvalues are unexpectedly large.")
        w[w < eigen_fudge] = eigen_fudge

    cov2 = stats.Covariance.from_eigendecomposition((w, v))

    # WARN: Weird/inconsistent behavior warning
    # if size==1 and cov is an instance of stats.Covariance
    # return value has shape of (1, len(loc2),)
    # this behavior is consistent with size > 1 which yields (size, len(loc2))
    # but is INCONSISTENT with behavior when cov is a
    # valid numpy array ---> shape is (len(loc2),)

    draw = stats.multivariate_normal.rvs(
        mean=loc, cov=cov2.covariance, size=ndraws
    )
    # draw = np.random.multivariate_normal(loc, cov2.covariance, size=ndraws)
    return draw[0] if ndraws == 1 else draw
