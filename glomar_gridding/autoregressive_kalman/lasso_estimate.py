"""
Direct estimation of weights used in vectorized autoregressive model
using gridpoint-by-gridpoint fitting of Lasso regression
(aka seemingly (un)related regression)

Outputs can be organised to be used for class `Autoregressive1ForecastVector`
in `forecast_linear_ar1`

Uses sklearn

Default lasso_lambda_hyperparm is estimated by tuning 1 deg x 1 deg ERA5 MAT
and ESA SST on gridpoints in the tropical Indian Ocean, Carribean, and NINO3.4.

For the purpose of docstring
N - number of gridpoints
T - length of time series at each grid point
"""
import numpy as np
import scipy as sp

from sklearn import linear_model
from typing import Literal

Lasso = linear_model.Lasso
LassoSelection = Literal['cyclic', 'random']

OUT_DTYPE = np.float32


class LassoEstimate_AR1():
    """
    A method to estimate of weights for Autoregressive1ForecastVector
    using regularized line-by-line / gridpoint-by-gridpoint (aka seemingly
    (un)related regression SUR) Lasso regression.

    This method is signficantly slower than using compute_autocorrelation
    but handle spatial autocorrelation more properly.

    This method is not directly compatible with covariance estimated
    using the ellipse method. No correlation-must-decrease-with-distance
    constrains when estimating the Lasso weights.
    """

    def __init__(
        self,
        data_sample: np.ndarray,
        standardise: bool = True,
        lasso_lambda_hyperparm: float = 0.01,
        lasso_selection: LassoSelection = "random",
        out_of_sample_residues: bool = True,
        hold_out_ratio: float = 0.5,
        dtype: type = OUT_DTYPE,
    ):
        """
        __init__ for LassoEstimate_AR1 class

        See:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        as well. This is just a wrapper for that but will loop through
        data_sample over many rows.

        Parameters
        ----------
        data_sample: numpy.ndarray
            A 2D data sample array
            Different columns for different time
            Different rows for different grid points/locations
            dtype `float`
            shape `(T, N)` following netCDF/Grib convention
        standardise: bool
            Should data_sample be standardised automatically
            (i.e. A* = A - E(A) / sigma(A) )
        lasso_lambda_hyperparm: float
            The lambda hyperparameter for Lasso regression
            It is called `alpha` in sklearn, but it is more common
            to call in lambda in the literature.
            Tuning of alpha/lambda is usually done by orders of magitude
            0.0001, 0.001, 0.01, 0.1, 1... etc (default of sklearn is 1)
        lasso_selection: str
            `selection` in sklearn lasso class, see sklearn documentation
        out_of_sample_residues: bool
            Should the residues of the fit be estimated out of sample
            When set to True, data_sample is split by hold_out_ratio
            and data not used for training will be used to estimate
            the residues. Otherwise, residues are estimated in-sample.
        hold_out_ratio: float
            The ratio of data be used that is withheld for
            cross-validation. It is only used if out_of_sample_residues
            is True
        dtype: type
            numpy or python native float types that are used
            to store the outputs
        """
        check_shape(data_sample)
        self.standardise = standardise
        if standardise:
            advisory = "Remember adjust weights and predictions"
            advisory += " when deploying model for VAR(1)!"
            print(advisory)
            self.sample, self.sample_mean, self.sample_sigma = standardise_data(
                data_sample
            )
        else:
            advisory = "If some values along XY are systemtically small, "
            advisory += "this may distort Lasso estimates, "
            advisory += "standardisation is recommended."
            print(advisory)
            self.sample = data_sample
            self.sample_mean = np.zeros_like(data_sample[0, :])
            self.sample_sigma = np.ones_like(data_sample[0, :])
        self.out_of_sample_residues = out_of_sample_residues
        self.hold_out_ratio = hold_out_ratio
        self.n_t = self.sample.shape[0]  # Length of vector time series (axis=0)
        self.n_xy = self.sample.shape[1]  # Vector length (axis=1)
        self.split_holdout()
        self.dtype = dtype
        self.lambda_hyperparm = lasso_lambda_hyperparm
        self.selection = lasso_selection

    def split_holdout(self):
        """
        Split data into training part and holdout cross-validation bits
        if self.out_of_sample_residues is False, no split will be used,
        and the full obs dataset will be used for training and residue
        estimation.
        """
        if self.out_of_sample_residues:
            print('Out-of-sample will be used to compute residues.')
            print('Holdout cross-validation will be performed.')
            self.half_of_the_data = int(self.n_t * self.hold_out_ratio)
            #
            # Training data
            self.X = self.sample[:(self.half_of_the_data - 1), :]
            self.all_y = self.sample[1:(self.half_of_the_data), :]
            #
            # Holdout data
            self.X_withheld = self.sample[(self.half_of_the_data):-1, :]
            self.all_y_withheld = self.sample[(self.half_of_the_data + 1):, :]
        else:
            print('Training data will be used to compute residues.')
            print('Holdout cross-validation will not be performed.')
            print('The full input dataset will be used to fit the model.')
            self.X = self.X_withheld = self.sample[:-1, :]
            self.all_y = self.all_y_withheld = self.sample[1:, :]

    def fit(self):
        """Fit the Lasso regression"""
        self.line_by_line_sur_lasso()

    def line_by_line_sur_lasso(self):
        """
        The complete fitting procedure following line-by-line estimate
        of the regression coefficients
        based on seemingly (un)related regression approach

        Adds the following attributes to class instance:

        Returns
        -------
        self.coefficients: numpy.ndarray
            The weights of the fits
            shape `(N, N)`
        self.residues: numpy.ndarray
            The residues of the fit
            The length of residue time series
            depends on `out_of_sample_residues`
            and `hold_out_ratio`
            shape `(N, T*)`
            in which `T*` <= `T`
        """
        self.coefficients = np.zeros((self.n_xy, self.n_xy), dtype=self.dtype)
        # Note numpy default is row-major
        if self.out_of_sample_residues:
            self.residues = np.zeros(
                (self.n_xy, self.half_of_the_data - 1),
                dtype=self.dtype,
            )
        else:
            self.residues = np.zeros(
                (self.n_xy, self.n_t - 1),
                dtype=self.dtype,
            )
        the_lr_looper = range(self.n_xy)
        for xy in the_lr_looper:
            print(f"Now working on grid point/line {xy = }/{self.n_xy - 1}")
            y = self.all_y[:, xy]
            #
            ossaL = Lasso(
                alpha=self.lambda_hyperparm,
                selection=self.selection,
            )
            ossaL.fit(self.X, y)
            self.coefficients[xy, :] = ossaL.coef_
            #
            lasso_score = ossaL.score(self.X, y)
            n_near0_coef_ = np.sum(np.isclose(ossaL.coef_, 0.0))
            n_meaningful_coef_ = ossaL.coef_.shape[0] - n_near0_coef_
            #
            print(f"{ossaL.coef_ = }")
            print(f"{n_near0_coef_ = }; {n_meaningful_coef_ = }")
            print(f"{np.min(ossaL.coef_) = }; {np.max(ossaL.coef_) = }")
            print(f"{lasso_score = }")
            #
            insample_predictions = ossaL.predict(self.X)
            insample_y = y
            insample_residuals = insample_y - insample_predictions
            insample_MAE = np.mean(np.abs(insample_residuals))
            insample_RMSE = np.sqrt(np.mean(np.square(insample_residuals)))
            print("Insample diagonstics:")
            print(f"MAE(Lasso(l={self.lambda_hyperparm})) = {insample_MAE}")
            print(f"RMSE(Lasso(l={self.lambda_hyperparm})) = {insample_RMSE}")
            #
            if self.out_of_sample_residues:
                outsample_predictions = ossaL.predict(self.X_withheld)
                outsample_y = self.all_y_withheld[:, xy]
                outsample_residues = outsample_y - outsample_predictions
                outsample_MAE = np.mean(np.abs(outsample_residues))
                outsample_RMSE = np.sqrt(np.mean(np.square(outsample_residues)))
                print("Outsample diagonstics:")
                print(f"MAE(Lasso(l={self.lambda_hyperparm}))={outsample_MAE}")
                print(f"RMSE(Lasso(l={self.lambda_hyperparm}))={outsample_RMSE}")
                self.residues[xy, :] = outsample_residues
            else:
                self.residues[xy, :] = insample_residuals
        print("Task complete")

    def make_coeff_sparse(self):
        """Convert the weights to scipy sparse format"""
        if hasattr(self, 'coefficients'):
            self.coefficients = sp.sparse.csr_array(self.coefficients)
        else:
            print('Coefficients have not been computed yet.')


def check_shape(X):
    """
    Check X to be 2D
    This should follow the convention that
    Each column represent different xy gridpoints
    Each row represents another time
    Proper netCDF files should have the correct order of the dimensions:
    (E), T, (Z), Y, X
    So reshaping (T, Y, X) >> T, YX will be sufficient
    """
    if len(X.shape) != 2:
        raise ValueError(f'Unexpected shape: {X.shape = }')


def standardise_data(
        X,
        normalise=True,
        return_mean_and_std=True,
        axis=0,
    ):
    """
    Standardise the data samples by
    1) subtracting the mean
    2) divide by standard deviation

    Parameters
    ----------
    X: np.ndarray
        As in data_sample in LassoEstimate_AR1 class
    normalise: bool
        If set to True, divide by sample standard deviation
    return_mean_and_std: bool
        Return the sample mean and standard deviation prior to
        standardisation
    axis: int
        The axis that the standardisation is applied
        axis=0 follows the convention that each column
        is a seperate time series, which is the
        standard for netCDF and Grib files (i.e. time as 1st dimension;
        (T, YX) convention).
    """
    X_bar = np.mean(X, axis=axis)
    if normalise:
        X_std = np.std(X, axis=axis)
    else:
        X_std = np.ones_like(X_bar)
    X_standardised = (X - X_bar) / X_std
    if return_mean_and_std:
        ans = (
            X_standardised,
            X_bar,
            X_std,
        )
    else:
        ans = X_standardised
    return ans
