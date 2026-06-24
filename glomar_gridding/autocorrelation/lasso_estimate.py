"""
Direct estimation of weights used in vectorized autoregressive model
using grid point-by-grid point fitting of Lasso regression Tibshirani
[Tibshirani_Lasso] (aka seemingly (un)related regression)

Outputs can be organised to be used for class `Autoregressive1ForecastVector`
in `forecast_linear_ar1`

Uses sklearn

Default `lam` is estimated by tuning 1 deg x 1 deg ERA5 MAT
and ESA SST on grid points in the tropical Indian Ocean, Caribbean, and NINO3.4.
Your mileage may vary.

For the purpose of docstring
N - number of grid points
T - length of time series at each grid point
"""

import logging
import numpy as np
import scipy as sp
import warnings

import xarray as xr
from sklearn.linear_model import Lasso
from typing import Literal

from glomar_gridding.utils import check_shape
from glomar_gridding.transformation import standardise_data

LassoSelection = Literal["cyclic", "random"]


class LassoEstimate_AR1:
    """
    A method to estimate of weights for Autoregressive1ForecastVector
    using regularized line-by-line / grid point-by-grid point (aka seemingly
    (un)related regression SUR) Lasso regression.

    This method is significantly slower than using compute_autocorrelation,
    but handle spatial autocorrelation more properly.

    This method is not directly compatible with covariance estimated
    using the ellipse method. No correlation-must-decrease-with-distance
    constrains when estimating the Lasso weights.

    See:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    as well. This is just a wrapper for that but will loop through
    data_sample over many rows.

    Parameters
    ----------
    data_sample: numpy.ndarray
        A 2D data sample array;
        different columns for different time;
        different rows for different grid points/locations;
        dtype `float`;
        shape `(T, N)` following netCDF/GRIB convention
    standardise: bool
        Should data_sample be standardised automatically
        (i.e. `A* = A - E(A) / sigma(A)` )? Standardise if `True`
    lam: float
        The lambda hyperparameter for Lasso regression;
        it is called `alpha` in `scikit-learn`, but it is more common
        to be called it `lambda` in the literature.
        Tuning of it is usually done by orders of magnitude
        0.0001, 0.001, 0.01, 0.1, 1... etc (default of sklearn is 1)
        If lam=0, it becomes OLS; scikit-learn will probably raise
        warnings if one does that.
    lasso_selection: str
        `selection` in `scikit-learn` `lasso` class,
        see `scikit-learn` documentation
    out_of_sample_residues: bool
        Should the residues of the fit be estimated out of sample
        When set to `True`, data_sample is split by hold_out_ratio
        and data not used for training will be used to estimate
        the residues. Otherwise, residues are estimated in-sample.
    hold_out_ratio: float
        The ratio of data be used that is withheld for
        cross-validation. It is only used if `out_of_sample_residues`
        is `True`
    dtype: type
        `numpy` or python native float types that are used
        to store the outputs
    lasso_kws: dict | None
        Dict with (other) keyword arguments for `sklearn.linear_model.Lasso`;
        Do not include `alpha` or `selection` in lasso_kws; use
        `lasso_selection` and `lam` instead. If `alpha` or `selection` are
        in lasso_kws, they will be removed.
    """

    def __init__(
        self,
        data_sample: np.ndarray,
        standardise: bool = True,
        lam: float = 0.01,
        lasso_selection: LassoSelection = "random",
        out_of_sample_residues: bool = True,
        hold_out_ratio: float = 0.5,
        dtype: type = np.float32,
        lasso_kws: dict | None = None,
    ):
        """__init__ for LassoEstimate_AR1 class"""
        check_shape(data_sample)
        self.standardise = standardise
        if standardise:
            advisory = "Remember to adjust weights and predictions"
            advisory += " when deploying model for VAR(1)!"
            logging.debug(advisory)
            self.sample, self.sample_mean, self.sample_sigma = standardise_data(
                data_sample
            )
        else:
            advisory = "If some values along XY are systematically small, "
            advisory += "this may distort Lasso estimates, "
            advisory += "standardisation is recommended."
            warnings.warn(advisory, UserWarning)
            self.sample = data_sample
            self.sample_mean = np.zeros_like(data_sample[0, :])
            self.sample_sigma = np.ones_like(data_sample[0, :])
        self.out_of_sample_residues = out_of_sample_residues
        self.hold_out_ratio = hold_out_ratio
        self.n_t = self.sample.shape[0]  # Length of vector time series (axis=0)
        self.n_xy = self.sample.shape[1]  # Vector length (axis=1)
        self.split_holdout()
        self.dtype = dtype
        self.lam = lam
        self.selection = lasso_selection
        self.lasso_kws = {} if lasso_kws is None else lasso_kws

    def split_holdout(self):
        """
        Split data into training part and holdout cross-validation bits
        if `self.out_of_sample_residues` is `False`, no split will be used,
        and the full obs dataset will be used for training and residue
        estimation.
        """
        if self.out_of_sample_residues:
            logging.debug("Out-of-sample will be used to compute residues.")
            logging.debug("Holdout cross-validation will be performed.")
            self.split_point = int(self.n_t * self.hold_out_ratio)
            #
            # Training data
            self.X = self.sample[: (self.split_point - 1), :]
            self.y = self.sample[1 : (self.split_point), :]
            #
            # Holdout data
            self.X_test = self.sample[(self.split_point) : -1, :]
            self.y_test = self.sample[(self.split_point + 1) :, :]
        else:
            logging.debug("Training data will be used to compute residues.")
            logging.debug("Holdout cross-validation will not be performed.")
            logging.debug("Full input dataset will be used to fit the model.")
            self.split_point = None
            self.X = self.X_test = self.sample[:-1, :]
            self.y = self.y_test = self.sample[1:, :]

    def fit(self, convert_to_sparse: bool = False):
        """
        Fit the Lasso regression

        The complete fitting procedure following line-by-line estimate
        of the regression coefficients based on seemingly (un)related
        regression approach

        Parameters
        ----------
        convert_to_sparse: bool
            If True, runs make_coeff_sparse automatically,
            which converts coefficients to a scipy sparse
            matrix. Default False.

        Attributes
        ----------
        coefficients: numpy.ndarray
            The weights of the fits;
            shape `(N, N)`
        residues: numpy.ndarray
            The residues of the fit;
            length of residue time series depends on
            `out_of_sample_residues` and `hold_out_ratio`;
            shape `(N, T*)` in which `T*` <= `T`
        expanded: bool
            A bool flag that indicates if coefficients
            have been expanded by the expand_coefficients method
        """
        self.coefficients = np.zeros((self.n_xy, self.n_xy), dtype=self.dtype)
        # Note numpy default is row-major
        if self.out_of_sample_residues:
            self.residues = np.zeros(
                (self.n_xy, self.split_point - 1),
                dtype=self.dtype,
            )
        else:
            self.residues = np.zeros(
                (self.n_xy, self.n_t - 1),
                dtype=self.dtype,
            )
        for xy in range(self.n_xy):
            logging.debug(f"Now working on grid point {xy = }/{self.n_xy - 1}")
            y1 = self.y[:, xy]
            #
            self.lasso_kws.pop("alpha", None)
            self.lasso_kws.pop("selection", None)
            lasso = Lasso(
                alpha=self.lam,
                selection=self.selection,
                **self.lasso_kws,
            )
            lasso.fit(self.X, y1)
            self.coefficients[xy, :] = lasso.coef_
            #
            lasso_score = lasso.score(self.X, y1)
            n_near0_coef_ = np.sum(np.isclose(lasso.coef_, 0.0))
            n_meaningful_coef_ = lasso.coef_.shape[0] - n_near0_coef_
            #
            logging.debug(f"{lasso.coef_ = }")
            logging.debug(f"{n_near0_coef_ = }; {n_meaningful_coef_ = }")
            logging.debug(f"{np.min(lasso.coef_) = }; {np.max(lasso.coef_) = }")
            logging.debug(f"{lasso_score = }")
            #
            insample_predictions = lasso.predict(self.X)
            insample_y = y1
            insample_residuals = insample_y - insample_predictions
            insample_MAE = np.mean(np.abs(insample_residuals))
            insample_RMSE = np.sqrt(np.mean(np.square(insample_residuals)))
            logging.debug("Insample diagonstics:")
            logging.debug(f"MAE(Lasso(l={self.lam}))={insample_MAE}")
            logging.debug(f"RMSE(Lasso(l={self.lam}))={insample_RMSE}")
            #
            if self.out_of_sample_residues:
                outsample_predictions = lasso.predict(self.X_test)
                outsample_y1 = self.y_test[:, xy]
                outsample_residues = outsample_y1 - outsample_predictions
                outsample_MAE = np.mean(np.abs(outsample_residues))
                outsample_RMSE = np.sqrt(np.mean(np.square(outsample_residues)))
                logging.debug("Outsample diagonstics:")
                logging.debug(f"MAE(Lasso(l={self.lam}))={outsample_MAE}")
                logging.debug(f"RMSE(Lasso(l={self.lam}))={outsample_RMSE}")
                self.residues[xy, :] = outsample_residues
            else:
                self.residues[xy, :] = insample_residuals
        #
        if convert_to_sparse:
            self.make_coeff_sparse()
        self.expanded = False
        #
        logging.debug("Task complete")

    def make_coeff_sparse(self):
        """Convert the weights to scipy sparse format"""
        if hasattr(self, "coefficients"):
            logging.debug("coefficients matrix converted to sparse.")
            self.coefficients = sp.sparse.csr_array(self.coefficients)
        else:
            raise AttributeError("Coefficients have not been computed yet.")

    def expand_coefficients(
        self,
        D: np.ndarray | sp.sparse.sparray,
        dtype: type = np.float32,
        fill_value: float = 0.6,
    ):
        """
        Expands coefficients according to the subsampling
        matrix D. D should be array of integers of 0s and 1s,
        and will be converted to sparse if not sparse already.

        self.coefficient matrix will be converted to sparse
        if has not been converted.

        Parameters
        ----------
        D: numpy.ndarray | scipy.sparse.sparray
            A 2D array indicating unmasked location, should normally be sparse;
            If not an instance `sp.sparse.sparray`, it will converted to one;
            dtype `uint8` or `bool`;
            shape `(N, M)`
        dtype: type
            `numpy` or python native `float` types that are used for
            expanded array. Given the nature of this type of analysis
            double precision floats (float64) are overkill. Defaults
            to single precision floats (float32).
        fill_value: float
            The diagonal fill value of the weights; default 0.6
        """
        if not sp.sparse.issparse(self.coefficients):
            self.make_coeff_sparse()
        D = _d_check(D)
        #
        W = D.T @ self.coefficients @ D
        # See:
        # https://stackoverflow.com/questions/32743584/python-lil-matrix-vs-csr-matrix-in-extremely-large-sparse-matrices
        logging.debug(f"Filling diagonals of empty rows with {fill_value}")
        empty_rows = W.getnnz(1) == 0
        where_empty = np.where(empty_rows)[0]
        n_empty_rows = np.sum(empty_rows)
        W_coo = W.tocoo()
        W_dat = W_coo.data.copy()
        W_row = W_coo.coords[0].copy()
        W_col = W_coo.coords[1].copy()
        del W_coo
        W_dat = np.append(W_dat, np.array([fill_value] * n_empty_rows))
        W_row = np.append(W_row, where_empty)
        W_col = np.append(W_col, where_empty)
        self.coefficients = sp.sparse.csr_matrix(
            (W_dat, (W_row, W_col)),
            shape=W.shape,
            dtype=dtype,
        )
        self.expanded = True

    def shrink_coefficients(
        self,
        D: np.ndarray | sp.sparse.sparray,
    ):
        """
        Reverses expand_coefficients, again by
        the subsampling matrix D.

        Parameters
        ----------
        D: numpy.ndarray | scipy.sparse.sparray
            A 2D array indicating unmasked location, should normally be sparse;
            If not an instance `sp.sparse.sparray`, it will converted to one;
            dtype `uint8` or `bool`;
            shape `(N, M)`
        """
        D = _d_check(D)
        self.coefficients = D @ self.coefficients @ D.T
        self.expanded = False

    def to_xarray_da_coefficients(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        name: str = "weights",
        attrs: dict | None = {"description": "weights", "units": "1"},
    ) -> xr.DataArray:
        """
        Convert expanded coefficients into a xarray DataArray instance

        Example attrs for netCDF:

        .. code-block::python
           attrs = {
               "lasso_hyperparm_lam": 0.01,
               "description": "standardised VAR(1) weights for SST",
               "units": "1",
           }

        Parameters
        ----------
        lats: numpy.ndarray
            Array of latitude values, such as `np.linspace(-19.5, 19.5, 40)`
        lons: numpy.ndarray
            Array of longitude values, such as `np.linspace(-179.5, 179.5, 360)`
        name: str
            Name of the variable
        attrs: dict
            Attributes to be added the `xarray.DataArray`

        Returns
        -------
        da: xarray.DataArray
            self.coefficients as a DataArray that has metadata and
            written easily to netCDF file
        """
        if not self.expanded:
            warnings.warn(
                "self.coefficients have not been expanded yet",
                UserWarning,
            )
        if attrs is None:
            attrs = {}
        xx, yy = np.meshgrid(lons, lats)
        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        w_dim = xr.Coordinates(
            {
                "row": np.arange(self.coefficients.shape[0], dtype=np.uint32),
                "col": np.arange(self.coefficients.shape[0], dtype=np.uint32),
            }
        )
        w_dim.update(
            {
                "row_lat": ("row", yy),
                "col_lat": ("col", yy),
                "row_lon": ("row", xx),
                "col_lon": ("col", xx),
            }
        )
        da = xr.DataArray(
            data=self.coefficients.toarray(),
            coords=w_dim,
            name=name,
            attrs=attrs,
        )
        return da


def _d_check(D: np.ndarray | sp.sparse.sparray):
    """
    Basic check if D is a (sparse) array.
    Error raised if D is not an array.
    Convert to sparse if needed to.

    Parameters
    ----------
    D: numpy.ndarray | scipy.sparse.sparray
        An array to be checked (usually a subsampling
        matrix)

    Returns
    -------
    D: scipy.sparse.sparray
        Sparse version of D
    """
    if sp.sparse.issparse(D):
        logging.debug("D is sparse.")
    elif isinstance(D, np.ndarray):
        logging.debug("Converting D to sparse.")
        D = sp.sparse.csr_array(D.copy(), dtype=np.uint8)
    else:
        raise ValueError(f"Unknown object {type(D)}.")
    return D
