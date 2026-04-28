"""
Docstring for cov_diagonal

remove_diag_only_rows:
- Auto-detect diagonal only rows/columns from an input matrix
- Purge them to form a smaller matrix and saves the matrix

restore_diag_only_rows:
- reverses the process, but one needs to give a filler value

diag_and_nondiag_rows_subsampler:
- get the subsampling matrix of diagonal and off-diagonal rows/columns
- option to return the actual subsampled array
"""

import numpy as np
import scipy as sp

EFFECTIVELY_ZERO_DEFAULT: float = 1e-6


def _more_than_one_element(
    row: np.ndarray, zero_threshold: float = EFFECTIVELY_ZERO_DEFAULT
):
    """Check if 1D vector more than one non-zero element"""
    return np.sum(np.abs(row) > zero_threshold) > 1


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
        a new covariance without those diagonal-only elements
    D: numpy.ndarray
        the subsampling matrix that generates new_cor_arr
    """
    n_rows = cov.shape[0]
    print(f"{cov.shape = }")
    n_validrows = 0
    if not np.all(np.diag(cov) > zero_threshold):
        err_msg = f'There are elements below {zero_threshold = }'
        err_msg += ' (negatives included) on the diagonal.'
        raise ValueError(err_msg)
    has_off_diagonal_elements = np.sum(np.abs(cov) > zero_threshold, axis=0) > 1
    n_validrows = int(np.sum(has_off_diagonal_elements))
    print(f"{n_validrows = }")
    if n_validrows < 1:
        raise ValueError(f"{n_validrows} must be at >= 1")
    #
    D = np.zeros((n_validrows, cov.shape[0]), dtype=np.uint8)
    row_count = 0
    for i in range(n_rows):
        if has_off_diagonal_elements[i] == 0:
            continue
        print(f"{row_count} {i}")
        D[row_count, i] = 1
        row_count += 1
    print(D)
    print(f"{D.shape = }")
    #
    new_cov_arr = np.matmul(D, cov)
    print(f"Progress update: {new_cov_arr.shape = }")
    new_cov_arr = np.matmul(new_cov_arr, D.T)
    print(new_cov_arr)
    print(f"{new_cov_arr.shape = }")
    #
    return new_cov_arr, D


def restore_diag_only_rows(
    trimmed_cov_arr: np.ndarray,
    D: np.ndarray,
    diag_fillvalue: float = 1.2,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Re-expanding subsampled covariance matrix prior to
    the removal of diagonal-only elements.
    Diagonal elements are filled with diag_fillvalue.

    Parameters
    ----------
    trimmed_cov_arr: numpy.ndarray
        A trimmed numpy array that needs expanded
    D: numpy.ndarray
        The subsampling array that did the original purge
        (see remove_diag_only_rows)
    diag_fillvalue: float
        The diagonal fillvalue for restored rows and columns
    atol: float
        Instead of checking for exact 0s,
        this is the threshold to decide diag_fillvalue replacement will occur

    Returns
    -------
    new_cov_arr: numpy.ndarray
        A larger (restored) covariance array
    """
    print(f"{trimmed_cov_arr.shape = }")
    print(f"{D.shape = }")
    #
    new_cov_arr = np.matmul(D.T, trimmed_cov_arr)
    print(f"Progress update: {new_cov_arr.shape = }")
    new_cov_arr = np.matmul(new_cov_arr, D)
    print(f"{new_cov_arr.shape = }")
    #
    diag_vals = np.array(np.diag(new_cov_arr))
    old_diag_vals = diag_vals.copy()
    diag_vals[diag_vals <= atol] = diag_fillvalue
    np.fill_diagonal(new_cov_arr, diag_vals)
    #
    print(new_cov_arr)
    print(f"{np.sum(diag_vals) = }")
    print(f"{np.sum(old_diag_vals) = }")
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
        covariance matrix with possible diagonal only elements
    zero_threshold: float
        The near-zero threshold
    return_subsampled_arr: bool
        Set to True if one wants the split subsampled covariances,
        otherwise function will just return the subsampling
        operator matrices

    Returns
    -------
    d_off_diagonal: numpy.ndarray
        sampling matrix operator for the off-diagonal rows
    the_denser_parts: numpy.ndarray | None
        a (somewhat denser) subsampled by that matrix
        Set to None if return_subsampled_arr is False
    d_diagonal_only: numpy.ndarray
        sampling matrix operator for the diagonal only rows
    isolated_diag_vals: numpy.ndarray | None
        vector with diagonal values of those diagonal rows
        Set to None if return_subsampled_arr is False
    """
    n_rows = cov.shape[0]
    print(f"{cov.shape = }")
    n_validrows = 0
    #
    # This returns True for rows that have off diagonal elements
    has_off_diagonal_elements = np.apply_along_axis(
        lambda row: _more_than_one_element(
            row,
            zero_threshold=zero_threshold,
        ),
        0,
        cov,
    )
    n_validrows = int(np.sum(has_off_diagonal_elements))
    n_diag_only = cov.shape[0] - n_validrows
    print(f"{n_validrows = }")
    print(f"{n_diag_only = }")
    if n_validrows < 1:
        raise ValueError(f"{n_validrows} must be at >= 1")
    #
    d_diagonal_only = np.zeros((n_diag_only, cov.shape[0]), dtype=bool)
    d_off_diagonal = np.zeros((n_validrows, cov.shape[0]), dtype=bool)
    row_count_off_diagonal = 0
    row_count_diagonal_only = 0
    for i in range(n_rows):
        if has_off_diagonal_elements[i] == 0:
            d_diagonal_only[row_count_diagonal_only, i] = True
            row_count_diagonal_only += 1
        else:
            d_off_diagonal[row_count_off_diagonal, i] = True
            row_count_off_diagonal += 1
    d_diagonal_only = sp.sparse.csr_matrix(d_diagonal_only)
    d_off_diagonal = sp.sparse.csr_matrix(d_off_diagonal)
    print(f"{type(d_off_diagonal) = }")
    print(f"{d_off_diagonal.shape = }")
    print(f"{type(d_diagonal_only) = }")
    print(f"{d_diagonal_only.shape = }")
    #
    diag_cov = np.diag(cov)
    diag_cov = np.array(diag_cov)
    if return_subsampled_arr:
        # isolated_diag_vals = np.matmul(
        #     d_diagonal_only.toarray(),
        #     diag_cov,
        # )
        # the_denser_parts = np.matmul(
        #     np.matmul(d_off_diagonal.toarray(), cov),
        #     d_off_diagonal.toarray().T,
        # )
        isolated_diag_vals = d_diagonal_only @ diag_cov
        the_denser_parts = d_off_diagonal @ cov @ d_off_diagonal.T
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
