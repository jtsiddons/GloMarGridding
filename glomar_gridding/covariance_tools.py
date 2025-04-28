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
perserving the original trace (aka total variance) of the
covariance matrix, but this will require a full computation
of all eigenvectors, which may be slow and cause
memory problems

3) EOF chop-off (fastest, trace/variance non-conserving):
Set a target explained variance (say 95%) for the emperical
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

import numpy as np
from scipy import linalg as linalg_scipy

from glomar_gridding.utils import cov_2_cor, cor_2_cov
from statsmodels.stats import correlation_tools


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


def _csum_up_to_val(
    samples: np.ndarray,
    target: float,
    reversed: bool = False,
    sort_data: bool = False,
) -> tuple[float, int]:
    """Find csum and sample index that target is surpassed."""
    if sort_data:
        samples = np.sort(samples)
    if reversed:
        samples = samples[::-1]
    niter = 0
    csum = 0.0
    for niter, sample in enumerate(samples):
        csum += sample
        print(niter, sample, csum)
        if csum > target:
            break
    return csum, niter


class CovarianceFixer:
    """
    https://www.worldscientific.com/doi/abs/10.1142/S0219024900000255

    Eigenvalue clipping based on number of thresholds and normalizations
    Can be done vs the actual covariance (or the correlation converted back to
    covariance)
    """

    def __init__(
        self,
        cov,
        clean_small_vals: bool = False,
        atol: float = 1e-5,
    ):
        print("CovarianceClean __init__")
        print(cov.shape)
        if clean_small_vals:
            self.cov = self.clean_small(cov, atol=atol)
        else:
            self.cov = cov
        if isinstance(self.cov, np.ma.MaskedArray):
            self.cov = self.cov.data
        print((np.max(self.cov), np.min(self.cov)))
        self.cor = cov_2_cor(self.cov)

    def clean_small(
        self,
        matrix: np.ndarray,
        atol: float = 1e-5,
    ) -> np.ndarray:
        """DOCUMENTATION"""
        small_stuff = np.abs(matrix) < atol
        ans = matrix.copy()
        ans[small_stuff] = 0.0
        return ans

    def find_index_explained_variance(self, eigvals, target=0.95):
        """DOCUMENTATION"""
        total_variance = np.sum(eigvals)
        target_explained_variance = target * total_variance
        print((total_variance, target_explained_variance))
        _, i2goal = _csum_up_to_val(eigvals, target_explained_variance)
        return i2goal

    def find_index_aspect_ratio(
        self,
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
        _, threshold = self.estimate_threshold(num_grid_pts, num_times)
        return -int(np.sum(eigvals > threshold))

    def estimate_threshold(
        self,
        num_grid_pts: int,
        num_times: int,
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

    def eig_clip_via_cor(
        self, method="explained_variance", method_parms={"target": 0.95}
    ):
        """
        Denoise symmetric damaged covariance/correlation matrix C
        by clipping eigenvalues

        This is the original method (?)
        https://www.worldscientific.com/doi/abs/10.1142/S0219024900000255

        Explained variance or aspect ratio based threshold
        Aspect ratios is based on dimensionless parameters
        (number of independent variable and observation size)
        q = N/T
          = (num of independent variable)
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
        print("Solving eigenvalues and vectors")
        Corr = cov_2_cor(self.cov)
        # print(np.trace(Corr), Corr.shape[0])
        eigvals, eigvec = linalg_scipy.eigh(Corr)
        sorted_order = np.argsort(eigvals)
        eigvals = eigvals[sorted_order]
        print("[:5] =", eigvals[:5])
        print("[-5:] =", eigvals[-5:])
        eigvecs = eigvec[:, sorted_order]
        n_eigvals = len(eigvals)
        thresh_ms = {
            "explained_variance": self.find_index_explained_variance,
            "Laloux_2000": self.find_index_aspect_ratio,
        }
        i2keep = thresh_ms[method](eigvals, **method_parms)
        i2clip = n_eigvals + i2keep  # Note i2keep is NEGATIVE
        print("Numbers of kept eigenvalues = ", -i2keep)
        print("Numbers of clipped eigenvalues = ", i2clip)
        #
        # The total variance should be perserved after clipping
        # within precision error of the eigenvalues which is
        # O(Max(Eig) * float_accuracy)
        total_var = np.sum(eigvals)
        var_explained_by_i2keep = np.sum(eigvals[i2keep:])
        unexplained_var = total_var - var_explained_by_i2keep
        avg_eigenvals_4_unexplained = unexplained_var / i2clip
        #
        # Find eigenvectors associated up to i2keep
        new_eigvals = eigvals.copy()
        new_eigvals[:i2keep] = avg_eigenvals_4_unexplained
        Corr_hat = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
        # print(np.trace(Corr_hat), Corr_hat.shape)
        #
        # Restore the trace (rounding error causes slight deviations)
        Corr_hat = cov_2_cor(Corr_hat)
        # print(np.trace(Corr_hat), Corr_hat.shape)
        #
        return cor_2_cov(Corr_hat, np.diag(self.cov))

    def eig_clip_via_cov(
        self, method="explained_variance", method_parms={"target": 0.95}
    ):
        """
        Denoise symmetric damaged covariance matrix C by clipping eigenvalues.
        Only works with explained variance approach?

        An example:
        https://ntrs.nasa.gov/api/citations/20170007918/downloads/20170007918.pdf
        https://core.ac.uk/download/pdf/95856791.pdf

        Laloux_2000 thresholds are for correlation and is clipping thresholds
        are dimensionless parameters (doesn't work with covariances)

        Since this is real explained variance, i.e. SST variances and EOFs
        it is more physically intituative

        This is done by KEEPING the largest explained variance
        in which (number of basis vectors to be kept) >> (number of rows)
        In ESA data, keeping 95% variance means keeping top ~15% of the
        eigenvalues
        """
        print("Solving eigenvalues and vectors")
        eigvals, eigvec = linalg_scipy.eigh(self.cov)
        sorted_order = np.argsort(eigvals)
        eigvals = eigvals[sorted_order]
        print("[:5] =", eigvals[:5])
        print("[-5:] =", eigvals[-5:])
        eigvecs = eigvec[:, sorted_order]
        n_eigvals = len(eigvals)
        thresh_ms = {"explained_variance": self.find_index_explained_variance}
        i2keep = thresh_ms[method](eigvals, **method_parms)
        i2clip = n_eigvals + i2keep  # Note i2keep is NEGATIVE
        print("Numbers of kept eigenvalues = ", -i2keep)
        print("Numbers of clipped eigenvalues = ", i2clip)
        #
        # The total variance should be perserved after clipping
        # within precision error of the eigenvalues which is
        # O(Max(Eig) * float_accuracy)
        total_var = np.sum(eigvals)
        var_explained_by_i2keep = np.sum(eigvals[i2keep:])
        unexplained_var = total_var - var_explained_by_i2keep
        avg_eigenvals_4_unexplained = unexplained_var / i2clip
        #
        # Find eigenvectors associated up to i2keep
        new_eigvals = eigvals.copy()
        new_eigvals[:i2keep] = avg_eigenvals_4_unexplained
        C_hat = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
        return C_hat


def main():
    """Main - keep calm and does nothing!"""
    print("--- Main ---")


if __name__ == "__main__":
    main()
