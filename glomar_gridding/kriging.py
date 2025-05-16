"""
Functions for performing Kriging.

Interpolation using a Gaussian Process. Available methods are Simple and
Ordinary Kriging.
"""
################
# by A. Faulkner
# for python version >= 3.11
################

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from warnings import warn

from .utils import adjust_small_negative, intersect_mtlb

KrigMethod = Literal["simple", "ordinary"]


class Kriging(ABC):
    """
    Class for Kriging.

    Do not use this class, use SimpleKriging or OrdinaryKriging classes.
    """

    def __init__(self, covariance: np.ndarray) -> None:
        if not hasattr(self, "method"):
            raise TypeError(
                "Do not use the generic class directly, "
                + "use SimpleKriging or OrdinaryKriging"
            )
        self.covariance = covariance
        return None

    def set_kriging_weights(self, kriging_weights: np.ndarray) -> None:
        """
        Set Kriging Weights.

        Sets the `kriging_weights` attribute.

        Parameters
        ----------
        kriging_weights : numpy.ndarray
            The pre-computed kriging_weights to use.
        """
        self.kriging_weights = kriging_weights
        return None

    @abstractmethod
    def get_kriging_weights(
        self,
        idx: np.ndarray,
        error_cov: np.ndarray | None = None,
    ) -> None:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation. Optionally add an error covariance to the
        covariance between observation grid points.

        The Kriging weights are calculated as:

        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross}

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        Sets the `kriging_weights` attribute.

        Parameters
        ----------
        idx : numpy.ndarray[int] | list[int]
            The 1d indices of observation grid points. These values should be
            between 0 and (N * M) - 1 where N, M are the number of longitudes
            and latitudes respectively. Note that these values should also be
            computed using "C" ordering in numpy reshaping. They can be
            computed from a grid using glomar_gridding.grid.map_to_grid. Each
            value should only appear once. Points that contain more than 1
            observation should be averaged
        error_cov : numpy.ndarray | None
            Optionally add error covariance values to the covariance between
            observation grid points.
        """
        raise NotImplementedError(
            "`get_kriging_weights` not implemented for default class"
        )

    @abstractmethod
    def kriging_weights_from_inverse(
        self,
        inv: np.ndarray,
        idx,
    ) -> None:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation, using a pre-computed inverse of the covariance
        between grid-points with observations.

        The Kriging weights are calculated as:

        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross}

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        Sets the `kriging_weights` attribute.


        Parameters
        ----------
        inv : numpy.ndarray
            The pre-computed inverse of the covariance between grid-points with
            observations. :math:`(K_{obs} + E)^{-1}`
        idx : numpy.ndarray[int] | list[int]
            The 1d indices of observation grid points. These values should be
            between 0 and (N * M) - 1 where N, M are the number of longitudes
            and latitudes respectively. Note that these values should also be
            computed using "C" ordering in numpy reshaping. They can be
            computed from a grid using glomar_gridding.grid.map_to_grid. Each
            value should only appear once. Points that contain more than 1
            observation should be averaged
        """
        raise NotImplementedError(
            "`kriging_weights_from_inverse` not implemented for default class"
        )

    @abstractmethod
    def solve(
        self,
        grid_obs: np.ndarray,
        idx: np.ndarray,
        error_cov: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""
        Solves the Kriging problem. Computes the Kriging weights if the
        `kriging_weights` attribute is not already set. The solution to Kriging
        is:
        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross} \\times y

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points), and :math:`y` are the observation values.

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
            The solution to the Kriging problem (as a Vector, this may need to
            be re-shaped appropriately as a post-processing step).
        """
        raise NotImplementedError("`solve` not implemented for default class")

    @abstractmethod
    def get_uncertainty(self) -> np.ndarray:
        """
        Compute the kriging uncertainty. This requires the attribute
        `kriging_weights` to be computed.

        Returns
        -------
        uncert : numpy.ndarray
            The Kriging uncertainty.
        """
        raise NotImplementedError(
            "`get_uncertainty` not implemented for default class"
        )

    @abstractmethod
    def constraint_mask(self, idx: np.ndarray) -> np.ndarray:
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

        This requires that the `kriging_weights` attribute is set.

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
        raise NotImplementedError(
            "`constraint_mask` not implemented for default class"
        )


class SimpleKriging(Kriging):
    r"""
    Class for SimpleKriging.

    The equation for simple Kriging is:
    .. math::
        (K_{obs} + E)^{-1} \\times K_{cross} \\times y + \\mu

    Where :math:`\\mu` is a constant known mean, typically this is 0.

    Parameters
    ----------
    covariance : numpy.ndarray
        The spatial covariance matrix. This can be a pre-computed matrix loaded
        into the environment, or computed from a Variogram class or using
        Ellipse methods.
    """

    method: str = "simple"

    def get_kriging_weights(
        self,
        idx: np.ndarray,
        error_cov: np.ndarray | None = None,
    ) -> None:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation. Optionally add an error covariance to the
        covariance between observation grid points.

        The Kriging weights are calculated as:

        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross}

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        Sets the `kriging_weights` attribute.

        Parameters
        ----------
        idx : numpy.ndarray[int] | list[int]
            The 1d indices of observation grid points. These values should be
            between 0 and (N * M) - 1 where N, M are the number of longitudes
            and latitudes respectively. Note that these values should also be
            computed using "C" ordering in numpy reshaping. They can be
            computed from a grid using glomar_gridding.grid.map_to_grid. Each
            value should only appear once. Points that contain more than 1
            observation should be averaged
        error_cov : numpy.ndarray | None
            Optionally add error covariance values to the covariance between
            observation grid points.
        """
        obs_obs_cov = self.covariance[idx[:, None], idx[None, :]]
        obs_grid_cov = self.covariance[idx, :]

        # Add error covariance
        if error_cov is not None:
            if error_cov.shape[0] != len(idx):
                error_cov = error_cov[idx[:, None], idx[None, :]]
            obs_obs_cov += error_cov
        self.kriging_weights = np.linalg.solve(obs_obs_cov, obs_grid_cov).T

        return None

    def kriging_weights_from_inverse(
        self,
        inv: np.ndarray,
        idx,
    ) -> None:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation, using a pre-computed inverse of the covariance
        between grid-points with observations.

        The Kriging weights are calculated as:

        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross}

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        Sets the `kriging_weights` attribute.


        Parameters
        ----------
        inv : numpy.ndarray
            The pre-computed inverse of the covariance between grid-points with
            observations. :math:`(K_{obs} + E)^{-1}`
        idx : numpy.ndarray[int] | list[int]
            The 1d indices of observation grid points. These values should be
            between 0 and (N * M) - 1 where N, M are the number of longitudes
            and latitudes respectively. Note that these values should also be
            computed using "C" ordering in numpy reshaping. They can be
            computed from a grid using glomar_gridding.grid.map_to_grid. Each
            value should only appear once. Points that contain more than 1
            observation should be averaged
        """
        if len(idx) != inv.shape[0]:
            raise ValueError("inv must be square with side length == len(idx)")
        obs_grid_cov = self.covariance[idx, :]
        self.kriging_weights = (inv @ obs_grid_cov).T

    def solve(
        self,
        grid_obs: np.ndarray,
        idx: np.ndarray,
        error_cov: np.ndarray | None = None,
        mean: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        r"""
        Solves the simple Kriging problem. Computes the Kriging weights if the
        `kriging_weights` attribute is not already set. The solution to Kriging
        is:
        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross} \\times y

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points), and :math:`y` are the observation values.

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
        mean : numpy.ndarray | float
            Constant, known, mean value of the system. Defaults to 0.0.

        Returns
        -------
        numpy.ndarray
            The solution to the simple Kriging problem (as a Vector, this may
            need to be re-shaped appropriately as a post-processing step).
        """
        if not hasattr(self, "kriging_weights"):
            self.get_kriging_weights(idx, error_cov)

        return self.kriging_weights @ grid_obs + mean

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

        alpha = self.kriging_weights[:, -1]
        dz_squared = np.diag(self.covariance - self.kriging_weights)
        dz_squared -= alpha
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

        This requires that the `kriging_weights` attribute is set.

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
        if not hasattr(self, "kriging_weights"):
            raise KeyError("Please compute Kriging Weights first")

        numerator = np.diag(self.covariance[:, idx] @ self.kriging_weights.T)
        denominator = np.diag(self.covariance)
        return np.divide(numerator, denominator)


class OrdinaryKriging(Kriging):
    r"""
    Class for OrdinaryKriging.

    The equation for ordinary Kriging is:
    .. math::
        (K_{obs} + E)^{-1} \\times K_{cross} \\times y

    Where :math:`\\mu` is a constant known mean, typically this is 0.

    In this case, the :math:`K_{obs}`, :math:`K_{cross}` and :math:`y` values
    are extended with a Lagrange multiplier term, ensuring that the Kriging
    weights are constrained to sum to 1.

    The matrix :math:`K_{obs}` is extended by one row and one column, each
    containing the value 1, except at the diagonal point, which is 0. The
    :math:`K_{cross}` matrix is extended by an extra row containing values of 1.
    Finally, the grid observations :math:`y` is extended by a single value of 0
    at the end of the vector.

    Parameters
    ----------
    covariance : numpy.ndarray
        The spatial covariance matrix. This can be a pre-computed matrix loaded
        into the environment, or computed from a Variogram class or using
        Ellipse methods.
    """

    method: str = "ordinary"

    def get_kriging_weights(
        self,
        idx: np.ndarray,
        error_cov: np.ndarray | None = None,
    ) -> None:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation. Optionally add an error covariance to the
        covariance between observation grid points.

        The Kriging weights are calculated as:

        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross}

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        In this case, the :math:`K_{obs}`, :math:`K_{cross}` and are extended
        with a Lagrange multiplier term, ensuring that the Kriging weights are
        constrained to sum to 1.

        The matrix :math:`K_{obs}` is extended by one row and one column, each
        containing the value 1, except at the diagonal point, which is 0. The
        :math:`K_{cross}` matrix is extended by an extra row containing values
        of 1.

        Sets the `kriging_weights` attribute.

        Parameters
        ----------
        idx : numpy.ndarray[int] | list[int]
            The 1d indices of observation grid points. These values should be
            between 0 and (N * M) - 1 where N, M are the number of longitudes
            and latitudes respectively. Note that these values should also be
            computed using "C" ordering in numpy reshaping. They can be
            computed from a grid using glomar_gridding.grid.map_to_grid. Each
            value should only appear once. Points that contain more than 1
            observation should be averaged
        error_cov : numpy.ndarray | None
            Optionally add error covariance values to the covariance between
            observation grid points.
        """
        N = len(idx)
        M = self.covariance.shape[0]

        obs_obs_cov = self.covariance[idx[:, None], idx[None, :]]
        obs_grid_cov = self.covariance[idx, :]

        # Add error covariance
        if error_cov is not None:
            if error_cov.shape[0] != len(idx):
                error_cov = error_cov[idx[:, None], idx[None, :]]
            obs_obs_cov += error_cov

        # Add Lagrange multiplier
        obs_obs_cov = np.block(
            [[obs_obs_cov, np.ones((N, 1))], [np.ones((1, N)), 0]]
        )
        obs_grid_cov = np.concatenate((obs_grid_cov, np.ones((1, M))), axis=0)
        self.kriging_weights = np.linalg.solve(obs_obs_cov, obs_grid_cov).T

        return None

    def kriging_weights_from_inverse(
        self,
        inv: np.ndarray,
        idx,
    ) -> None:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation, using a pre-computed inverse of the covariance
        between grid-points with observations.

        The Kriging weights are calculated as:

        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross}

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        In this case, the inverse matrix must be computed from the covariance
        between observation grid-points with the Lagrange multiplier applied.

        This method is appropriate if one wants to compute the constraint mask
        which requires simple Kriging weights, which can be computed from the
        unextended covariance inverse. The extended inverse can then be
        calculated from that inverse.

        Sets the `kriging_weights` attribute.

        Parameters
        ----------
        inv : numpy.ndarray
            The pre-computed inverse of the covariance between grid-points with
            observations. :math:`(K_{obs} + E)^{-1}`
        idx : numpy.ndarray[int] | list[int]
            The 1d indices of observation grid points. These values should be
            between 0 and (N * M) - 1 where N, M are the number of longitudes
            and latitudes respectively. Note that these values should also be
            computed using "C" ordering in numpy reshaping. They can be
            computed from a grid using glomar_gridding.grid.map_to_grid. Each
            value should only appear once. Points that contain more than 1
            observation should be averaged
        """
        if len(idx) != inv.shape[0] - 1:
            raise ValueError("inv must be square with side length == len(idx)")
        obs_grid_cov = self.covariance[idx, :]

        # Add Lagrange multiplier
        M = self.covariance.shape[0]
        obs_grid_cov = np.concatenate((obs_grid_cov, np.ones((1, M))), axis=0)
        self.kriging_weights = (inv @ obs_grid_cov).T

    def solve(
        self,
        grid_obs: np.ndarray,
        idx: np.ndarray,
        error_cov: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""
        Solves the simple Kriging problem. Computes the Kriging weights if the
        `kriging_weights` attribute is not already set. The solution to Kriging
        is:
        .. math::
            (K_{obs} + E)^{-1} \\times K_{cross} \\times y

        Where :math:`K_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, :math:`K_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points), and :math:`y` are the observation values.

        In this case, the :math:`K_{obs}`, :math:`K_{cross}` and are extended
        with a Lagrange multiplier term, ensuring that the Kriging weights are
        constrained to sum to 1.

        The matrix :math:`K_{obs}` is extended by one row and one column, each
        containing the value 1, except at the diagonal point, which is 0. The
        :math:`K_{cross}` matrix is extended by an extra row containing values
        of 1.

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

        # Add Lagrange multiplier
        grid_obs = np.append(grid_obs, 0)

        return self.kriging_weights @ grid_obs

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
        simple_kriging_weights: np.ndarray | None = None,
        error_cov: np.ndarray | None = None,
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

        This requires the Kriging weights from simple Kriging. If these are
        not provided as an input, then they are calculated.

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
        simple_kriging_weights : numpy.ndarray | None,
            The Kriging weights for the equivalent simple Kriging system.
        error_cov : numpy.ndarray | None,
            The error covariance matrix. Used to compute the simple Kriging
            weights if not provided. Can be excluded if not Kriging with an
            error covariance.

        Returns
        -------
        constraint_mask : numpy.ndarray
            Constraint mask values, the left-hand-side of equation A14 from
            Morice et al. (2021). This is a vector of length `k_obs.size[0]`.

        Reference
        ---------
        Morice et al. (2021) : https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019JD032361
        """
        if simple_kriging_weights is None:
            obs_obs_cov = self.covariance[idx[:, None], idx[None, :]]
            obs_grid_cov = self.covariance[idx, :]

            # Add error covariance
            if error_cov is not None:
                if error_cov.shape[0] != len(idx):
                    error_cov = error_cov[idx[:, None], idx[None, :]]
                obs_obs_cov += error_cov
            simple_kriging_weights = np.linalg.solve(
                obs_obs_cov, obs_grid_cov
            ).T

        numerator = np.diag(self.covariance[:, idx] @ simple_kriging_weights.T)
        denominator = np.diag(self.covariance)
        return np.divide(numerator, denominator)

    def extended_inverse(self, simple_inv: np.ndarray) -> np.ndarray:
        r"""
        Compute the inverse of a covariance matrix :math:`S = K_{obs} + E`, and
        use that to compute the inverse of the extended version of the
        covariance matrix with Lagrange multipliers, used by Ordinary Kriging.

        This is useful when one needs to perform BOTH simple and ordinary
        Kriging, or when one wishes to compute the constraint mask for
        ordinary Kriging which requires the Kriging weights for the equivalent
        simple Kriging problem.

        The extended form of S is given by

        |       1 |
        |   S   1 |
        |       1 |
        | 1 1 1 0 |

        This approach follows Guttman 1946 10.1214/aoms/1177730946

        Parameters
        ----------
        simple_inv : numpy.matrix
            Inverse of the covariance between observation grid-points

        Returns
        -------
        numpy.matrix
            Inverse of the extended covariance matrix between observation
            grid-points including the Lagrange multiplier factors.
        """
        if len(simple_inv.shape) != 2:
            raise ValueError("S must be a matrix")

        d = 0
        B = np.ones((simple_inv.shape[0], 1))

        E = np.matmul(simple_inv, B)
        f = d - np.matmul(B.T, E)
        finv = 1 / f
        G = finv * E.T
        # H = finv * np.matmul(B.T, Ainv)
        K = simple_inv + np.matmul(E, G)

        return np.block([[K, -G.T], [-G, finv]])


def kriging(  # noqa: C901
    obs_idx: np.ndarray,
    weights: np.ndarray,
    obs: np.ndarray | None,
    interp_cov: np.ndarray | None,
    error_cov: np.ndarray,
    remove_obs_mean: int = 0,
    obs_bias: np.ndarray | None = None,
    method: KrigMethod = "simple",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Kriging using a chosen method.

    Get array of krigged observations and anomalies for all grid points in the
    domain.

    This function is deprecated in favour of SimpleKriging and OrdinaryKriging
    classes. It will be removed in version 1.0.0.

    Parameters
    ----------
    obs_idx : np.ndarray[int]
        Grid indices with observations. It is expected that this should be an
        ordering that lines up with the 1st dimension of weights. If
        `observations.dist_weights` or `observations.get_weights` was used to
        get the weights then this is the ordering of
        `sorted(df["gridbox"].unique())`, which is a sorting on lat and lon
    weights : np.ndarray[float]
        Weight matrix (inverse of counts of observations).
    obs : np.ndarray[float]
        All point observations/measurements for the chosen date.
    interp_cov : np.ndarray[float]
        interpolation covariance of all output grid points (each point in time
        and all points against each other).
    error_cov : np.ndarray[float]
        Measurement/Error covariance matrix.
    remove_obs_mean: int
        Should the mean or median from grib_obs be removed and added back onto
        grib_obs?
        0 = No (default action)
        1 = the mean is removed
        2 = the median is removed
        3 = the spatial meam os removed
    obs_bias : np.ndarray[float] | None
        Bias of all measurement points for a chosen date (corresponds to x_obs).
    method : KrigMethod
        The kriging method to use to fill in the output grid. One of "simple"
        or "ordinary".

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using the chosen kriging method.
    dz : np.ndarray[float]
        Uncertainty associated with the chosen kriging method.
    """
    warn(
        "kriging is deprecated and will be removed in version v1.0.0, "
        + "use SimpleKriging or OrdinaryKriging classes",
        DeprecationWarning,
    )
    if obs is None or interp_cov is None:
        raise ValueError(
            "Observations and interpolation covariance must be supplied"
        )
    if obs_bias is not None:
        print("With bias")
        grid_obs = weights @ (obs - obs_bias)
    else:
        grid_obs = weights @ obs

    grid_obs = np.squeeze(grid_obs) if len(grid_obs) > 1 else grid_obs

    match remove_obs_mean:
        case 0:
            grid_obs_av = None
        case 1:
            grid_obs_av = np.ma.average(grid_obs)
            grid_obs = grid_obs - grid_obs_av
        case 2:
            grid_obs_av = np.ma.median(grid_obs)
            grid_obs = grid_obs - grid_obs_av
        case 3:
            grid_obs_av = get_spatial_mean(grid_obs, error_cov)
            grid_obs = grid_obs - grid_obs_av
        case _:
            raise ValueError("Unknown 'remove_obs_mean' value")

    print(f"{grid_obs.shape = }")

    if error_cov.shape == interp_cov.shape:
        print(
            "Error covariance supplied is of the same size as interpolation "
            + "covariance, subsetting to indices of observation grids"
        )
        error_cov = error_cov[obs_idx[:, None], obs_idx[None, :]]

    print(f"{error_cov =}, {error_cov.shape =}")

    # S is the spatial covariance between all "measured" grid points
    # Plus the covariance due to the measurements, i.e. measurement noise, bias
    # noise, and sampling noise (R)
    obs_obs_cov = np.asarray(interp_cov[obs_idx[:, None], obs_idx[None, :]])
    obs_obs_cov += weights @ error_cov @ weights.T
    print(f"{obs_obs_cov =}, {obs_obs_cov.shape =}")
    # Ss is the covariance between to be "predicted" grid points (i.e. all) and
    # "measured" points
    obs_grid_cov = np.asarray(interp_cov[obs_idx, :])
    print(f"{obs_grid_cov =}, {obs_grid_cov.shape =}")

    match method.lower():
        case "simple":
            print("Performing Simple Kriging")
            z_obs, dz = kriging_simple(
                obs_obs_cov, obs_grid_cov, grid_obs, interp_cov
            )
        case "ordinary":
            print("Performing Ordinary Kriging")
            z_obs, dz = kriging_ordinary(
                obs_obs_cov, obs_grid_cov, grid_obs, interp_cov
            )
        case _:
            raise NotImplementedError(
                f"Kriging method {method} is not implemented. "
                'Expected one of "simple" or "ordinary"'
            )

    if grid_obs_av is not None:
        z_obs = z_obs + grid_obs_av

    return z_obs, dz


def unmasked_kriging(
    unmask_idx: np.ndarray,
    unique_obs_idx: np.ndarray,
    weights: np.ndarray,
    obs: np.ndarray,
    interp_cov: np.ndarray,
    error_cov: np.ndarray,
    remove_obs_mean: int = 0,
    obs_bias: np.ndarray | None = None,
    method: KrigMethod = "simple",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Kriging on a masked grid using a chosen method.

    Get array of krigged observations and anomalies for all grid points in the
    domain.

    This function is deprecated in favour of SimpleKriging and OrdinaryKriging
    classes. It will be removed in version 1.0.0.

    Parameters
    ----------
    unmask_idx : np.ndarray[int]
        Indices of all un-masked points for chosen date.
    unique_obs_idx : np.ndarray[int]
        Unique indices of all measurement points for a chosen date,
        representative of the indices of gridboxes, which have => 1 measurement.
    weights : np.ndarray[float]
        Weight matrix (inverse of counts of observations).
    obs : np.ndarray[float]
        All point observations/measurements for the chosen date.
    interp_cov : np.ndarray[float]
        Interpolation covariance of all output grid points (each point in time
        and all points
        against each other).
    error_cov : np.ndarray[float]
        Measurement/Error covariance matrix.
    remove_obs_mean: int
        Should the mean or median from obs be removed and added back onto obs?
        0 = No (default action)
        1 = the mean is removed
        2 = the median is removed
        3 = the spatial meam os removed
    obs_bias : np.ndarray[float] | None
        Bias of all measurement points for a chosen date (corresponds to x_obs).
    method : KrigMethod
        The kriging method to use to fill in the output grid. One of "simple"
        or "ordinary".

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using the chosen kriging method.
    dz : np.ndarray[float]
        Uncertainty associated with the chosen kriging method.
    """
    warn(
        "unmasked_kriging is deprecated and will be removed in version v1.0.0, "
        + "use SimpleKriging or OrdinaryKriging classes",
        DeprecationWarning,
    )
    obs_idx = get_unmasked_obs_indices(unmask_idx, unique_obs_idx)

    return kriging(
        obs_idx,
        weights,
        obs,
        interp_cov,
        error_cov,
        remove_obs_mean,
        obs_bias,
        method,
    )


def get_unmasked_obs_indices(
    unmask_idx: np.ndarray,
    unique_obs_idx: np.ndarray,
) -> np.ndarray:
    """
    Get grid indices with observations from un-masked grid-box indices and
    unique grid-box indices with observations.

    Parameters
    ----------
    unmask_idx : np.ndarray[int]
        List of all unmasked grid-box indices.
    unique_obs_idx : np.ndarray[int]
        Indices of grid-boxes with observations.

    Returns
    -------
    obs_idx : np.ndarray[int]
        Subset of grid-box indices containing observations that are unmasked.
    """
    unmask_idx = np.squeeze(unmask_idx) if unmask_idx.ndim > 1 else unmask_idx
    _, obs_idx, _ = intersect_mtlb(unmask_idx, unique_obs_idx)
    # index of the sorted unique (iid) in the full iid array
    obs_idx = obs_idx.astype(int)

    return obs_idx


def kriging_simple(
    obs_obs_cov: np.ndarray,
    obs_grid_cov: np.ndarray,
    grid_obs: np.ndarray,
    interp_cov: np.ndarray,
    mean: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Simple Kriging assuming a constant known mean.

    This function is deprecated in favour of SimpleKriging class. It will be
    removed in version 1.0.0.

    Parameters
    ----------
    obs_obs_cov : np.ndarray[float]
        Covariance between all measured grid points plus the
        covariance due to measurements (i.e. measurement noise, bias noise, and
        sampling noise). Can include error covariance terms.
    obs_grid_cov : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
        Does not contain error covarance.
    grid_obs : np.ndarray[float]
        Gridded measurements (all measurement points averaged onto the output
        gridboxes).
    interp_cov : np.ndarray[float]
        interpolation covariance of all output grid points (each point in time
        and all points against each other).
    mean : float
        The constant mean of the output field.

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using simple kriging.
    dz : np.ndarray[float]
        Uncertainty associated with the simple kriging.
    """
    warn(
        "kriging_simple is deprecated and will be removed in version v1.0.0, "
        + "use SimpleKriging",
        DeprecationWarning,
    )
    kriging_weights = np.linalg.solve(obs_obs_cov, obs_grid_cov).T
    kriged_result = kriging_weights @ grid_obs

    kriging_weights = kriging_weights @ obs_grid_cov
    dz_squared = np.diag(interp_cov - kriging_weights)
    dz_squared = adjust_small_negative(dz_squared)
    uncert = np.sqrt(dz_squared)
    uncert[np.isnan(uncert)] = 0.0

    print("Simple Kriging Complete")
    return kriged_result + mean, uncert


def kriging_ordinary(
    obs_obs_cov: np.ndarray,
    obs_grid_cov: np.ndarray,
    grid_obs: np.ndarray,
    interp_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Ordinary Kriging with unknown but constant mean.

    This function is deprecated in favour of OrdinaryKriging class. It will be
    removed in version 1.0.0.

    Parameters
    ----------
    obs_obs_cov : np.ndarray[float]
        Covariance between all measured grid points plus the covariance due to
        measurements (i.e. measurement noise, bias noise, and sampling noise).
        Can include error covariance terms, if these are being used.
    obs_grid_cov : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
        Does not contain error covarance.
    grid_obs : np.ndarray[float]
        Gridded measurements (all measurement points averaged onto the output
        gridboxes).
    interp_cov : np.ndarray[float]
        Interpolation covariance of all output grid points (each point in time
        and all points against each other).

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using ordinary kriging.
    dz : np.ndarray[float]
        Uncertainty associated with the ordinary kriging.
    """
    warn(
        "kriging_ordinary is deprecated and will be removed in version v1.0.0, "
        + "use OrdinaryKriging",
        DeprecationWarning,
    )
    # Convert to ordinary kriging, add Lagrangian multiplier
    N, M = obs_grid_cov.shape
    obs_obs_cov = np.block(
        [[obs_obs_cov, np.ones((N, 1))], [np.ones((1, N)), 0]]
    )
    obs_grid_cov = np.concatenate((obs_grid_cov, np.ones((1, M))), axis=0)
    grid_obs = np.append(grid_obs, 0)

    kriging_weights = np.linalg.solve(obs_obs_cov, obs_grid_cov).T
    kriged_result = kriging_weights @ grid_obs

    alpha = kriging_weights[:, -1]
    kriging_weights = kriging_weights @ obs_grid_cov
    uncert_squared = np.diag(interp_cov - kriging_weights) - alpha
    uncert_squared = adjust_small_negative(uncert_squared)
    uncert = np.sqrt(uncert_squared)
    # dz[np.isnan(dz)] = 0.0

    print("Ordinary Kriging Complete")
    return kriged_result, uncert


def get_spatial_mean(
    grid_obs: np.ndarray,
    covx: np.ndarray,
) -> float:
    """
    Compute the spatial mean accounting for auto-correlation.

    Parameters
    ----------
    grid_obs : np.ndarray
        Vector containing observations
    covx : np.ndarray
        Observation covariance matrix

    Returns
    -------
    spatial_mean : float
        The spatial mean defined as (1^T x C^{-1} x 1)^{-1} * (1^T x C^{-1} x z)

    Reference
    ---------
    https://www.css.cornell.edu/faculty/dgr2/_static/files/distance_ed_geostats/ov5.pdf
    """
    n = len(grid_obs)
    ones = np.ones(n)
    invcov = ones.T @ np.linalg.inv(covx)

    return float(1 / (invcov @ ones) * (invcov @ grid_obs))


def constraint_mask(
    obs_obs_cov: np.ndarray,
    obs_grid_cov: np.ndarray,
    interp_cov: np.ndarray,
) -> np.ndarray:
    """
    Compute the observational constraint mask (A14 in Morice et al. (2021) -
    10.1029/2019JD032361) to determine if a grid point should be masked/weights
    modified by how far it is to its near observed point

    Note: typo in Section A4 in Morice et al 2021 (confired by authors).

    Equation to use is A14 is incorrect. Easily noticeable because dimensionally
    incorrect is wrong, but the correct answer is easy to figure out.

    Correct Equation (extra matrix inverse for K+R):
    1 - diag( K(X*,X*) - k*^T @ (K+R)^{-1} @ k* )  / diag( K(X*,X*) )  < alpha

    This can be re-written as:
    diag(k*^T @ (K+R)^{-1} @ k*) / diag(K(X*, X*)) < alpha

    alpha is chosen to be 0.25 in the UKMO paper

    Written by S. Chan, modified by J. Siddons.

    Parameters
    ----------
    obs_obs_cov : np.ndarray[float]
        Covariance between all measured grid points plus the covariance due to
        measurements (i.e. measurement noise, bias noise, and sampling noise).
        Can include error covariance terms, if these are being used. This is
        `K + R` in the above equation.
    obs_grid_cov : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
        Does not contain error covarance. This is `k*` in the above equation.
    interp_cov : np.ndarray[float]
        Interpolation covariance of all output grid points (each point in time
        and all points against each other). This is `K(X*, X*)` in the above
        equation.

    Returns
    -------
    constraint_mask : numpy.ndarray
        Constraint mask values, the left-hand-side of equation A14 from Morice
        et al. (2021). This is a vector of length `k_obs.size[0]`.

    Reference
    ---------
    Morice et al. (2021) : https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019JD032361
    """
    # ky_inv = np.linalg.inv(k_obs + err_cov)
    # NOTE: Ax = b => x = A^{-1}b (x = solve(A, b))
    Kinv_kstar = np.linalg.solve(obs_obs_cov, obs_grid_cov)
    numerator = np.diag(obs_grid_cov.T @ Kinv_kstar)
    denominator = np.diag(interp_cov)
    constraint_mask = numerator / denominator
    # constraint_mask has the length of number of grid points
    # (obs-covered and interpolated.)
    return constraint_mask
