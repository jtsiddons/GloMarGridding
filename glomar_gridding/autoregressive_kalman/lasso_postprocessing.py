"""
Post-process outputs from lasso_estimate
putting outputs in usable form

(Somewhat) similar to lasso_estimate:

N - number of grid points
M - number of valid grid points (N >= M)
T* - length of cross-validated errors at each grid point
"""

import numpy as np
import scipy as sp
import xarray as xr

import cov_diagonal


class LassoWeights:
    """
    Class to post process Lasso regression estimate weights
    e.g. the `self.coefficients` from class LassoEstimate_AR1
    Expanding them to full domain
    Convert them from numpy array or scipy sparse array
    to xarray DataArray

    Parameters
    ----------
    W: numpy.ndarray | scipy.sparse.sparray
        A 2D data weights array
        Should normally be sparse
        If it is not an instance sp.sparse.sparray,
        it will converted to one.
        dtype `float`
        shape `(N, N)`
    D: numpy.ndarray | scipy.sparse.sparray
        A 2D array indicating unmasked location
        Should normally be sparse
        If it is not an instance sp.sparse.sparray,
        it will converted to one.
        dtype `uint`, `bool` etc.
        shape `(N, M)`
    dtype: type
        numpy or python native float types that are used
        for xarray to store the outputs
    fillvalue: float
        The diagonal fill value of the weights
        Default 0.6
    auto_process: bool
        Allow the normal workflow of this class to be carried out
        automatically. That is to `expand_W` automatically.
        default `True`
    """

    def __init__(
        self,
        W: np.ndarray | sp.sparse.sparray,
        D: np.ndarray | sp.sparse.sparray,
        dtype: type = np.float32,
        fillvalue: float = 0.6,
        auto_process: bool = True,
    ):
        """__init__ for LassoWeights"""
        self.type = dtype
        self.fillvalue = fillvalue
        if sp.sparse.issparse(W):
            self.W = W
        elif isinstance(W, np.ndarray):
            self.W: sp.sparse.sparray = sp.sparse.csc_array(W, dtype=self.type)
        else:
            raise ValueError("W must be np.ndarray or sp.sparse.sparray")
        if sp.sparse.issparse(D):
            self.D = D
        elif isinstance(D, np.ndarray):
            self.D: sp.sparse.sparray = sp.sparse.csc_array(D, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown object {type(D)}.")
        self.expanded = False
        if auto_process:
            self.expand_W()

    def expand_W(self):  # noqa: N802
        """
        Expand W to include the full domain
        A fill value can be used to fill the diagonal of
        the originally fully empty rows.
        Note for VAR(1): 1 e-folding for t+2 implies a ~0.6
        in the diagonal which is the default of `LassoWeights`
        """
        if self.expanded:
            print("W is already expanded by D; no action taken.")
            return
        #
        # Note:
        # Both D and W are sp.sparse.spmatrix, and behaves and performs
        # differently than functions in cov_diagonal; (error) covariance
        # is, in contrast, a dense matrix.
        print("Expanding W into full matrix")
        self.W = self.D.T @ self.W @ self.D
        # See:
        # https://stackoverflow.com/questions/32743584/python-lil-matrix-vs-csr-matrix-in-extremely-large-sparse-matrices
        print(f"Filling diagonals of fully empty rows with {self.fillvalue}")
        empty_rows = self.W.getnnz(1) == 0
        where_empty = np.where(empty_rows)[0]
        n_empty_rows = np.sum(empty_rows)
        W_coo = self.W.tocoo()
        W_dat = W_coo.data.copy()
        W_row = W_coo.coords[0].copy()
        W_col = W_coo.coords[1].copy()
        del W_coo
        W_dat = np.append(W_dat, np.array([self.fillvalue] * n_empty_rows))
        W_row = np.append(W_row, where_empty)
        W_col = np.append(W_col, where_empty)
        self.W = sp.sparse.csr_matrix(
            (W_dat, (W_row, W_col)),
            shape=self.W.shape,
            dtype=self.type,
        )
        self.expanded = True

    def shrink_W(self):  # noqa: N802
        """
        If W is expanded, shrink back to its original shape
        fillvalues are removed by row-column subsampling
        """
        if not self.expanded:
            print("W is already D-subsampled; no action taken.")
            return
        print("Shrinking W back to its original shape")
        self.W = self.D @ self.W @ self.D.T
        self.expanded = False

    def to_xarray_da(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        name: str = "weights",
        attrs: dict | None = {"description": "weights", "units": "1"},
    ) -> xr.DataArray:
        """
        Convert expanded W into a xarray DataArray instance

        Example attrs:
        example attrs = {
            "lasso_lambda_hyperparm": 0.01,
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
            w as a DataArray that has metadata and written easily to netCDF
        """
        if not self.expanded:
            self.expand_W()
        if attrs is None:
            attrs = {}
        xx, yy = np.meshgrid(lons, lats)
        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        w_dim = xr.Coordinates(
            {
                "row": np.arange(self.W.shape[0], dtype=np.uint32),
                "col": np.arange(self.W.shape[0], dtype=np.uint32),
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
            data=self.W.toarray(),
            coords=w_dim,
            name=name,
            attrs=attrs,
        )
        return da


class LassoError:
    """
    Class to post process Lasso regression estimate residues
    e.g. the `self.residues` from class LassoEstimate_AR1
    The residues are recomputed into an estimated error covariance
    matrix. That is equivalent to:

    .. math::
       C - W C W^{T}

    part of the AR forecast error.

    Since LassoEstimate_AR1 by default normalises
    data before fitting, diagonal observed sigmas have to be multiplied
    back in (this is not performed here).

    Errors associated with other things
    like Kriging errors will still needed to be added in, such as
    :math:`W K W^{T}` (:math:`K` is Kriging covariance.)

    Convert output to xarray DataArray

    Convention for the residues matrix:
    rows - residues at each point (axis = 0)
    number of columns - number of time (in/out-of sample) residues
    per row (axis = 1)

    Parameters
    ----------
    residues_matrix: numpy.ndarray
        A 2D data residues
        dtype `float`
        shape `(N, T*)`
    dof_adj: float | int | None
        Degrees of freedom adjustment when computing covariance
        of the regression residues or forecast errors. This usually
        only applies to in-sample residue.
        The usual divisor for R @ R.T is sample size per time
        series/feature (dof == sample_size). However, it is in-sample
        residues of a regression fit:
        dof = sample_size - number_of_covariates
        if covariates/covariates are not correlated with each other nor
        the regression is regularized.
        If None (or 0 default), dof_adj will be set to 0,
        and R @ R.T will be divided by T*.
        Some ways to estimate dof_adj (if needed) can result
        in non-integer values, but single line OLS and Lasso
        always give integer adjustments to DOF.
    D: numpy.ndarray
        A 2D array indicating unmasked location
        Should normally be sparse
        If it is not an instance sp.sparse.sparray,
        it will converted to one.
        dtype `uint`, `bool` etc.
        shape `(N, M)`
    dtype: type
        numpy or python native float types that are used
        for xarray to store the outputs
    fillvalue: float
        The diagonal fill value of the error covariance
        Default 0.36
    auto_process: bool
        Allow the normal workflow of this class to be carried out
        automatically. That is to `estimate_errcov` and then
        `expand_R` if `D` is provided; default `True`
    """

    def __init__(
        self,
        residues_matrix: np.ndarray,
        dof_adj: int | float = 0,
        D: np.ndarray | None = None,
        dtype: type = np.float32,
        fillvalue: float = 0.36,
        auto_process: bool = True,
    ):
        """__init__ for LassoError"""
        self.type = dtype
        self.residues_matrix: np.ndarray = residues_matrix
        self.fillvalue = fillvalue
        if D is not None:
            self.D: sp.sparse.sparray = sp.sparse.csc_array(D, dtype=np.uint8)
        else:
            self.D = None
        self.dof_adj = dof_adj
        if self.dof_adj < 0:
            raise ValueError(f"dof_adj cannot be negative! {self.dof_adj = }")
        self.t = self.residues_matrix.shape[1]
        self.expanded = False
        if auto_process:
            self.estimate_errcov()
            if self.D is not None:
                self.expand_R()

    def estimate_errcov(self):
        """
        Estimate error covariance

        .. math::
           E = R @ R^{T} / (T^{*} - dof_adj)
        """
        print("Computing error covariance")
        dof = self.t - self.dof_adj
        print(f"{dof = }")
        print(f"{self.t = }; {self.dof_adj = }")
        print(f"{dof = }")
        self.R = (self.residues_matrix @ self.residues_matrix.T) / dof

    def expand_R(self):  # noqa: N802
        """
        Expand estimate errorcov to include the full domain
        A fill value can be used to fill the diagonal of
        the originally fully empty rows.
        """
        if self.D is None:
            raise ValueError("No D provided to class.")
        if not hasattr(self, "R"):
            err_msg = "Use method estimate_errcov or kwarg autoprocess "
            err_msg += "to compute errcov first."
            raise AttributeError(err_msg)
        if self.expanded:
            print("R is already expanded by D; no action taken.")
            return
        #
        print(f"Expanding R, filling 0 diagonals with {self.fillvalue}.")
        self.R = cov_diagonal.restore_diag_only_rows(
            self.R,
            self.D,
            diag_fillvalue=self.fillvalue,
        )
        self.expanded = True

    def to_xarray_da(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        name: str = "error_covariance",
        attrs: dict | None = None,
    ):
        """
        Convert estimated error covariance into a xarray DataArray instance

        attrs:
        example attrs = {
            "lasso_lambda_hyperparm": 0.01,
            "description": "VAR(1) estimated error covariance for SST",
            "units": "1"}

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
            errcov as a DataArray that has metadata and written easily to netCDF
        """
        if not hasattr(self, "R"):
            err_msg = "Use method estimate_errcov or kwarg autoprocess "
            err_msg += "to compute errcov first."
            raise AttributeError(err_msg)
        if not self.expanded:
            self.expand_R()
        if attrs is None:
            attrs = {
                "description": "standardised error_covariance_of_VAR(1)",
                "units": "1",
            }
        xx, yy = np.meshgrid(lons, lats)
        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        r_dim = xr.Coordinates(
            {
                "row": np.arange(self.R.shape[0], dtype=np.uint32),
                "col": np.arange(self.R.shape[0], dtype=np.uint32),
            }
        )
        r_dim.update(
            {
                "row_lat": ("row", yy),
                "col_lat": ("col", yy),
                "row_lon": ("row", xx),
                "col_lon": ("col", xx),
            }
        )
        da = xr.DataArray(
            data=self.R,
            coords=r_dim,
            name=name,
            attrs=attrs,
        )
        return da
