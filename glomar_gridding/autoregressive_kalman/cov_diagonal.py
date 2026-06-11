"""
Diagonal and non-diagonal matrix tools

remove_diag_only_rows:
- Auto-detect diagonal only rows/columns from an input matrix
- Purge them to form a smaller matrix and saves the matrix

restore_diag_only_rows:
- reverses the process, but one needs to give a filler value

diag_and_nondiag_rows_subsampler:
- get the subsampling matrix of diagonal and off-diagonal rows/columns
- option to return the actual subsampled array
"""

import logging
import numpy as np
import scipy as sp

EFFECTIVELY_ZERO_DEFAULT: float = 1e-6


def remove_diag_only_rows(
    cov: np.ndarray,
    zero_threshold: float = EFFECTIVELY_ZERO_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsampled the covariance matrix by the removal of
    diagonal-only elements. "Diagonal-only" elements are
    defined according to the zero_threshold

    Parameters
    ----------
    cov: numpy.ndarray
        covariance matrix with possible diagonal only elements
    zero_threshold: float
        The near-zero threshold

    Returns
    -------
    new_cov_arr : numpy.ndarray
        A new covariance without those diagonal-only elements
    D: numpy.ndarray
        The subsampling matrix that generates new_cor_arr
    """
    n_rows = cov.shape[0]
    logging.debug(f"{cov.shape = }")
    if not np.all(np.diag(cov) > zero_threshold):
        err_msg = f"There are elements below {zero_threshold = }"
        err_msg += " (negatives included) on the diagonal."
        raise ValueError(err_msg)
    has_off_diagonal_elements = np.sum(np.abs(cov) > zero_threshold, axis=0) > 1
    n_validrows = int(np.sum(has_off_diagonal_elements))
    logging.debug(f"{n_validrows = }")
    if n_validrows < 1:
        raise ValueError("cov is diagonal; no non-diagonal rows to work with")
    #
    D = sp.sparse.csr_matrix(
        np.eye(n_rows, dtype=np.uint8)[has_off_diagonal_elements, :]
    )
    logging.debug(f"{D.shape = }")
    #
    new_cov_arr = cov[has_off_diagonal_elements, :][
        :, has_off_diagonal_elements
    ]
    logging.debug(f"{new_cov_arr.shape = }")
    #
    return new_cov_arr, D


def restore_diag_only_rows(
    trimmed_cov_arr: np.ndarray,
    D: np.ndarray | sp.sparse.sparray,
    diag_fill_value: float = 1.2,
) -> np.ndarray:
    """
    Re-expanding subsampled covariance matrix prior to
    the removal of diagonal-only elements. Diagonal elements are
    filled with `diag_fill_value`.

    Parameters
    ----------
    trimmed_cov_arr: numpy.ndarray
        A trimmed numpy array that needs expanded
    D: numpy.ndarray
        The subsampling array that did the original purge
        (see remove_diag_only_rows); there should only be a
        single 1 or `True` per row (axis=0)
    diag_fill_value: float
        The diagonal fill value for restored rows and columns
    atol: float
        Instead of checking for exact 0s, this is the threshold
        to decide diag_fill_value replacement will occur

    Returns
    -------
    new_cov_arr: numpy.ndarray
        A larger (restored) covariance array
    """
    if isinstance(D, np.ndarray):
        has_off_diag = np.any(D, axis=0)
    elif sp.sparse.issparse(D):
        has_off_diag = np.any(D.toarray(), axis=0)
    else:
        err_msg = "D must be an instance numpy array or scipy sparse array"
        raise ValueError(err_msg)
    #
    logging.debug(f"{trimmed_cov_arr.shape = }")
    logging.debug(f"{D.shape = }")
    n = len(has_off_diag)
    if np.sum(has_off_diag) != trimmed_cov_arr.shape[0]:
        err_msg = "D shape is inconsistent with shape of trimmed_cov_arr."
        raise ValueError(err_msg)
    new_cov_arr = np.eye(n) * diag_fill_value
    fill_inds = np.logical_and.outer(has_off_diag, has_off_diag)
    np.place(new_cov_arr, fill_inds, trimmed_cov_arr)
    #
    old_diag_vals = np.diag(trimmed_cov_arr)
    diag_vals = np.diag(new_cov_arr)
    #
    logging.debug(f"post-restore trace: {np.sum(diag_vals) = }")
    logging.debug(f"pre-restore trace: {np.sum(old_diag_vals) = }")
    return new_cov_arr


def diag_and_nondiag_rows_subsampler(
    cov: np.ndarray,
    zero_threshold: float = EFFECTIVELY_ZERO_DEFAULT,
    return_subsampled_arr: bool = True,
) -> tuple[np.ndarray, None | np.ndarray, np.ndarray, None | np.ndarray]:
    """
    Get the subsampling matrices for rows and columns with
    only diagonal-only elements and off-diagonal elements.

    "Diagonal-only" elements are defined according to the zero_threshold.
    Any rows and columns that do not satisfy the "diagonal-only" definition
    is considered to have off-diagonal elements.

    Returns a tuple with up to 4 matrices including the subsampling
    matrices.

    Parameters
    ----------
    cov: numpy.ndarray
        Covariance matrix with possible diagonal only elements
    zero_threshold: float
        The near-zero threshold
    return_subsampled_arr: bool
        Set to `True` if one wants the split subsampled covariances,
        otherwise function will just return the subsampling operator matrices

    Returns
    -------
    d_off_diagonal: numpy.ndarray
        Sampling matrix operator for the off-diagonal rows
    the_denser_parts: numpy.ndarray | None
        A (somewhat denser) subsampled by that matrix
        Set to `None` if return_subsampled_arr is `False`
    d_diagonal_only: numpy.ndarray
        Sampling matrix operator for the diagonal only rows
    isolated_diag_vals: numpy.ndarray | None
        Vector with diagonal values of those diagonal rows
        Set to `None` if return_subsampled_arr is `False`
    """
    n_rows = cov.shape[0]
    logging.debug(f"{cov.shape = }")
    #
    # This returns True for rows that have off diagonal elements
    if not np.all(np.diag(cov) > zero_threshold):
        err_msg = f"There are elements below {zero_threshold = }"
        err_msg += " (negatives included) on the diagonal."
        raise ValueError(err_msg)
    has_off_diagonal_elements = np.sum(np.abs(cov) > zero_threshold, axis=0) > 1
    n_validrows = int(np.sum(has_off_diagonal_elements))
    n_diag_only = cov.shape[0] - n_validrows
    logging.debug(f"{n_validrows = }")
    logging.debug(f"{n_diag_only = }")
    if n_validrows < 1:
        raise ValueError("cov is diagonal; no non-diagonal rows to work with")
    #
    d_off_diagonal = sp.sparse.csr_matrix(
        np.eye(n_rows, dtype=np.uint8)[has_off_diagonal_elements, :]
    )
    d_diagonal_only = sp.sparse.csr_matrix(
        np.eye(n_rows, dtype=np.uint8)[~has_off_diagonal_elements, :]
    )
    logging.debug(f"{d_off_diagonal.shape = }")
    logging.debug(f"{d_diagonal_only.shape = }")
    #
    # Some versions of numpy (newer ones) return a view instead of array
    # Old versions of numpy will return an array
    diag_cov = np.array(np.diag(cov))
    if return_subsampled_arr:
        isolated_diag_vals = diag_cov[~has_off_diagonal_elements]
        the_denser_parts = cov[has_off_diagonal_elements, :][
            :, has_off_diagonal_elements
        ]
    else:
        isolated_diag_vals = None
        the_denser_parts = None
    #
    return (
        d_off_diagonal,
        the_denser_parts,
        d_diagonal_only,
        isolated_diag_vals,
    )
