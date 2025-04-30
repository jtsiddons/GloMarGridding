"""
Repair "damaged"/"improper" covariance matrices:
1) Un-invertible covariance matrices with 0 eigenvalues
2) Covariance matrices with eigenvalues less than zero

Known causes of damage:
1) Multicollinearity,
but nearly all very large cov matrices will have
rounding errors to have this occur
2) Number of spatial points >> length of time series
(for ESA monthly pentads: this ratio is about 150
3) Covariance is estimated using partial data

In our cases -- 2 and 3...?

Fixes:
1) Simple clipping:
Cut off the negative, zero, and small positive eigenvalues;
this is method used in statsmodels.stats.correlation_tools
but the version here has better thresholds based on
the accuracy of the eigenvalues, plus a iterivative
version which is slower but more stable with big matrices. The
iterivative version is recommended for SST/MAT covariances.

This is used for SST covariance matrices which have less dominant modes
than MAT; it also preserves more noise.

Trace (aka total variance) of the covariance matrix is not conserved,
but it is less disruptive than EOF chop off (method 3).

It is more difficult to use for covariance matrices with one large dominant mode
because that raises the bar of accuracy of the eigenvalues, so you have to clip
off a lot more eigenvectors.

2) Better (original) clipping:
Determine a noise eigenvalue threshold and
replace all eigenvalues below using the average of them,
preserving the original trace (aka total variance) of the
covariance matrix, but this will require a full computation
of all eigenvectors, which may be slow and cause
memory problems

3) EOF chop-off (fastest, trace/variance non-conserving):
Set a target explained variance (say 95%) for the empirical
orthogonal functions, compute the eigenvalues and
eigenvectors up to that explained variance. Reconstruct
the covariance keeping only EOFs up to the target. This
is very close to 2, but it reduces the total variance of
the covariance matrix. The original method requires solving
for ALL eigenvectors which may not be possible for massive
matrices (40000x40000 square matrices). This is currently done
for the MAT covariance matrices which have very large dominant
modes.

4) Other methods not implemented here
a) shrinkage methods
https://scikit-learn.org/stable/modules/covariance.html
b) reprojection (aka Higham's method)
https://github.com/mikecroucher/nearest_correlation
https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/
"""

from itertools import accumulate
from typing import Literal
import numpy as np
import scipy as sc
from statsmodels.stats import correlation_tools
from warnings import warn


def eof_chop(
    cov_arr: np.ndarray,
    target_explained_variance: float = 0.95,
):
    """
    Set a target explained variance (say 95%) for the empirical
    orthogonal functions, compute the eigenvalues and
    eigenvectors up to that explained variance. Reconstruct
    the covariance keeping only EOFs up to the target. This
    is very close to 2, but it reduces the total variance of
    the covariance matrix. The original method requires solving
    for ALL eigenvectors which may not be possible for massive
    matrices (40000x40000 square matrices). This is currently done
    for the MAT covariance matrices which have very large dominant
    modes.
    """
    if np.ma.isMaskedArray(cov_arr):
        cov_arr = cov_arr.data  # type: ignore

    # Compute all eigenvalues plus other useful diagonstics
    # Eigenvalues are returned in ascending order
    all_eigval = np.linalg.eigvalsh(cov_arr)
    N = len(all_eigval)
    min_eigval = all_eigval[0]
    max_eigval = all_eigval[-1]
    sum_eigval = np.sum(all_eigval)

    n_negatives = int(np.argmax(all_eigval >= 0))
    sum_of_negatives = np.sum(all_eigval[:n_negatives])

    p90_index = int(np.ceil(0.90 * N))
    p95_index = int(np.ceil(0.95 * N))
    sum_top10_eigval = np.sum(all_eigval[p90_index:])
    sum_top05_eigval = np.sum(all_eigval[p95_index:])

    top10_explained_var = 100.0 * sum_top10_eigval / sum_eigval
    top05_explained_var = 100.0 * sum_top05_eigval / sum_eigval
    explained_var_from_neg = 100.0 * sum_of_negatives / sum_eigval
    print("Pre-adjusted eigenvalue summary")
    print("Total variance=", sum_eigval)
    print("Largest=", max_eigval)
    print("Smallest/most negative=", min_eigval)
    print("Number of negative eigenvalues=", n_negatives)
    print("Sum=", sum_eigval)
    print("Explained variance from top 10%=", top10_explained_var, "%")
    print("Explained variance from top  5%=", top05_explained_var, "%")
    print(
        "Negative contributions from the negatives=",
        explained_var_from_neg,
        "%",
    )

    target_total_variance = target_explained_variance * sum_eigval
    all_eigval_R = all_eigval[::-1]
    eigenvals_2B_included = all_eigval_R[
        all_eigval_R.cumsum() <= target_total_variance
    ]
    n_eig_2B_included = len(eigenvals_2B_included)
    print("Target explained variance=", target_explained_variance)
    print("aka adjusted total variance=", target_total_variance)
    print("Requiring ", n_eig_2B_included, " eigenvalues")
    print(
        "Smallest eigenvalue in truncation=",
        eigenvals_2B_included[-1],
        "(aka threshold)",
    )
    print("Largest eigenvalue in truncation=", eigenvals_2B_included[0])

    print(
        "Computing eigenval & eigenvec up to the estimated number: "
        + str(n_eig_2B_included)
    )
    subset_by_index = [N - n_eig_2B_included, N - 1]
    # NOTE: Use SciPy here as can subset by index (numpy cannot)
    current_eigv, current_eigV = sc.linalg.eigh(
        cov_arr, subset_by_index=subset_by_index
    )
    print(current_eigv)
    print(current_eigV.shape)

    # This is new truncated covariance matrix...
    # it will/should have eigenvalues effectively 0
    cov_arr_adj = current_eigV @ np.diag(current_eigv) @ current_eigV.T

    n_vec = 10
    print("Computing adjusted eigenvalues, smallest " + str(n_vec))
    # NOTE: Use SciPy here as can subset by index (numpy cannot)
    new_eigv = sc.linalg.eigvalsh(cov_arr_adj, subset_by_index=[0, n_vec - 1])
    new_min_eigv = np.min(new_eigv)
    new_max_eigv = np.max(new_eigv)
    print("Largest eigenvalue=", new_max_eigv)
    print("Smallest eigenvalue=", new_min_eigv)
    print("Float32 precision=largest eigv x 1E-6=", 1e-6 * new_max_eigv, "]")

    new_det = np.linalg.det(cov_arr_adj)
    print("Determinant=", new_det)
    sum_eigval2 = np.sum(current_eigv)
    print(
        "Actual adjusted total variance after truncation=",
        sum_eigval2,
        "[Target = ",
        target_total_variance,
        "]",
    )

    meta_dict = {
        "target_explained_variance%": target_explained_variance * 100.0,
        "num_of_retained_eofs": n_eig_2B_included,
        "threshold": eigenvals_2B_included[-1],
        "smallest_eigv": new_min_eigv,
        "largest_eigv": new_max_eigv,
        "determinant": new_det,
        "total_variance": sum_eigval2,
    }
    return (cov_arr_adj, meta_dict)


def check_symmetric(
    a: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
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


def csum_up_to_val(
    vals: np.ndarray,
    target: float,
    reverse: bool = True,
    niter: int = 0,
    csum: float = 0.0,
) -> tuple[float, int]:
    """
    Find csum and sample index that target is surpassed. Displays a warning if
    the target is not exceeded or the input `vals` is empty.

    Can provide an initial `niter` and/or `csum` value(s), if working with
    multiple arrays in an iterative process.

    If `reverse` is set, the returned index will be negative and will correspond
    to the index required for the non-reversed array. Reverse is the default.

    Parameters
    ----------
    vals : numpy.ndarray
        Vector of values to sum cumulatively.
    target : float
        Value for which the cumulative sum must exceed.
    reverse : bool
        Reverse the array. The index will be negative.
    niter : int
        Initial number of iterations.
    csum : float
        Initial cumulative sum value.

    Returns
    -------
    csum : float
        The cumulative sum at the index when the target has been exceeded.
    niter : int
        The index of the value that results in the cumulative sum exceeding
        the target.

    Note
    ----
    It is actually faster to compute a full cumulative sum with `np.cumsum` and
    then look for the value that exceeds the target. This is not performed in
    this function.

    Examples
    --------
    >>> vals = np.random.rand(1000)
    >>> target = 301.1
    >>> csum_up_to_val(vals, target)
    """
    if vals.size == 0:
        warn("`vals` is empty")
        return csum, niter
    if len(vals) != vals.size:
        # Not a vector
        raise ValueError("`vals` must be a vector")

    vals = vals[::-1] if reverse else vals

    i = 0
    # accumulate defaults to sum
    for i, csum in enumerate(accumulate(vals, initial=csum), start=0):
        if csum > target:
            i = -i if reverse else i
            return csum, niter + i
    warn("Out of `vals`, target not exceeded.")
    i = -i if reverse else i
    return csum, niter + i


def clean_small(
    matrix: np.ndarray,
    atol: float = 1e-5,
) -> np.ndarray:
    """Set small values (abs(x) < atol) in an matrix to 0"""
    cleaned = matrix.copy()
    cleaned[np.abs(matrix) < atol] = 0.0
    return cleaned


def _find_index_explained_variance(eigvals, target=0.95):
    """
    Find the index of the eigenvalue for which the normalised cumulative sum
    exceeds a target variance.
    """
    total_variance = np.sum(eigvals)
    target_explained_variance = target * total_variance
    print((total_variance, target_explained_variance))
    csum, i2goal = csum_up_to_val(eigvals, target_explained_variance)
    if csum <= target_explained_variance:
        raise ValueError("Target Explained Variance not exceeded")
    return i2goal


def _find_index_aspect_ratio(
    eigvals: np.ndarray,
    num_grid_pts: int = 180 * 360,
    num_times: int = 41 * 6,
) -> int:
    """
    Defaults are based on:
    41 years ESA data
    6 pentads per month
    and 37000ish 1x1 deg grid points
    q ~ 150, threshold ~ 175

    For 5x5 data and 40-ish year of observations
    Observations are monthly
    q ~ 65, threshold ~ 82

    These parameters do not work in general
    must be determined from input data
    """
    _, threshold = _estimate_threshold(num_grid_pts, num_times)
    return -int(np.sum(eigvals > threshold))


def _estimate_threshold(
    num_grid_pts: int = 180 * 360,
    num_times: int = 41 * 6,
) -> tuple[float, float]:
    """
    See 7.2.2 in https://doi.org/10.1016/j.physrep.2016.10.005
    Eigenvalue threshold: threshold = (1.0 + SQRT(q))**2
    Below calculates q and threshold
    """
    q = num_grid_pts / num_times
    if q < 1.0:
        q = 1.0 / q
    threshold = (1.0 + np.sqrt(q)) ** 2.0
    return q, threshold


def eigenvalue_clip(
    mat: np.ndarray,
    method: Literal["explained_variance", "Laloux_2000"] = "explained_variance",
    method_parms={"target": 0.95},
):
    """
    Denoise symmetric damaged covariance/correlation matrix C
    by clipping eigenvalues

    This is the original method (?)
    https://www.worldscientific.com/doi/abs/10.1142/S0219024900000255

    Explained variance or aspect ratio based threshold
    Aspect ratios is based on dimensionless parameters
    (number of independent variable and observation size)
    q = N/T = (num of independent variable)
              / (num of observation per independent variable)
    Does not give the same results as in eig_clip

    explained_variance here does not have the same meaning.
    The trace of a correlation, by definition, equals the number of diagonal
    elements, which isn't intituatively linked to actual explained variance
    in climate science sense

    This is done by KEEPING the largest explained variance
    in which (number of basis vectors to be kept) >> (number of rows)
    In ESA data, keeping 95% variance means keeping top ~15% of the
    eigenvalues
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    print("[:5] =", eigvals[:5])
    print("[-5:] =", eigvals[-5:])
    n_eigvals = len(eigvals)

    match method:
        case "explained_variance":
            keep_i = _find_index_explained_variance(eigvals, **method_parms)
        case "Laloux_2000":
            keep_i = _find_index_aspect_ratio(eigvals, **method_parms)
        case _:
            raise ValueError("Unknown clipping method")

    clip_i = n_eigvals + keep_i  # Note i2keep is NEGATIVE
    print("Number of kept eigenvalues = ", -keep_i)
    print("Number of clipped eigenvalues = ", clip_i)

    # The total variance should be preserved after clipping
    # within precision error of the eigenvalues which is
    # O(Max(Eig) * float_accuracy)
    total_var = np.sum(eigvals)
    var_explained_by_i2keep = np.sum(eigvals[keep_i:])
    unexplained_var = total_var - var_explained_by_i2keep
    avg_eigenvals_4_unexplained = unexplained_var / clip_i

    # Find eigenvectors associated up to i2keep
    new_eigvals = eigvals.copy()
    new_eigvals[:keep_i] = avg_eigenvals_4_unexplained
    return eigvecs @ np.diag(new_eigvals) @ eigvecs.T
