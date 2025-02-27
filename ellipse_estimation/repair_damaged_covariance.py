import numbers

import numpy as np
from numpy import ma
from numpy import linalg
from scipy import linalg as linalg_scipy

'''
This repairs "damaged"/"improper" covariance matrices:
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
because that raises the bar of accuracy of the eigenvalues, so you have to clip off a
lot more eigenvectors.

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
'''

#
# Simplified eigenvalue clipping
#
def simple_clipping(cov_arr, threshold='auto', method='iterative'):
    '''
    A modified version of:
    https://www.statsmodels.org/dev/generated/statsmodels.stats.correlation_tools.corr_nearest.html
    '''
    assert method in ['iterative', 'direct'], 'Invalid method kwargs'
    all_eigval = linalg.eigvals(cov_arr)
    all_eigval = np.sort(all_eigval)
    max_eigval = np.max(all_eigval)
    min_eigval = np.min(all_eigval)
    sum_eigval = np.sum(all_eigval)
    p90_index = int(0.1*len(all_eigval))
    sumtop10_eigval = np.sum(all_eigval[-p90_index:])
    top10_explained_var = 100.0*(sumtop10_eigval/sum_eigval)
    print('Pre-adjusted eigenvalue summary')
    print('Largest=', max_eigval)
    print('Smallest/most negative=', min_eigval)
    print('Sum=', sum_eigval)
    print('Explained variance from top 10%=', top10_explained_var, '%')

    if threshold == 'auto':
        # According to
        # https://stackoverflow.com/questions/13891225/precision-of-numpys-eigenvaluesh
        # https://www.netlib.org/lapack/lug/node89.html
        #
        # Accuracy of eigenvalue of lapack is greater or equal to
        # MAX(ABS(eigenvalues)) x floating_pt_accuracy of the float type
        #
        # e.g. for float32 np.array:
        # np.finfo(np.float32) -->
        # finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)
        # Typical SSTA covariance matricies has max eigv ~ 1000 degC**2
        # This gives a lower bound threshold on the order of 1E-6 x 1E3 ~ 1E-3
        #
        # Give some margin of safety to above approximation: we will do 5x of the above
        # For most climate science applications,
        # these eigenvalues are essentially noise to the data
        finfo = np.finfo(all_eigval.dtype)
        threshold = 5.0 * finfo.resolution * np.max(np.abs(all_eigval))  # pylint: disable=E1101
    elif threshold == 'statsmodels_default':
        # 1e-15 is the precision for np.float64
        # This is the default used in
        # https://www.statsmodels.org/dev/generated/statsmodels.stats.correlation_tools.corr_clipped.html
        #
        # It is NOT SUITABLE based on guidelines to the precision of
        # lapack eigenvalue decomposition, nor the input data can be assumed to be float64!
        threshold = 1e-15
    else:
        # Threshold must be a number otherwise, this checks for invalid inputs
        assert_msg = 'threshold must either be number, auto or statsmodels_default'
        assert isinstance(threshold, numbers.Number), assert_msg

    n_negative = np.sum(all_eigval < threshold)
    print('Minimum eigenvalue threshold = ', threshold)
    print('Estimated number of eigenvalues below threshold = ', n_negative)
    n_vec = n_negative

    print('Computing eigenvalues and eigenvector up to the estimated number: '+str(n_vec))
    current_eigv, current_eigV = linalg_scipy.eigh(cov_arr,
                                                   eigvals_only=False,
                                                   subset_by_index=[0, n_vec-1])
    print(current_eigv)
    print(current_eigV)
    print(current_eigV.shape)

    # Make a copy
    cov_arr_adj = cov_arr.copy()

    # Rank-n_vec update
    print('Fixing matrix by reverse clipping')
    if method == 'iterative':
        for iii in range(n_vec):
            if current_eigv[iii] > threshold:
                warning_msg = 'New estimate of eigenvalue is below threshold,'
                warning_msg += 'possibly due to precision; bypassing.'
                print((iii,
                       n_vec-1,
                       current_eigv[iii],
                       warning_msg))
                continue
            worst_eigV = current_eigV[:, iii][np.newaxis]
            VbadxVbadT = worst_eigV * worst_eigV.T
            # This is only if threshold == 0
            # r_peturb = VbadxVbadT * current_eigv[iii]
            # cov_arr_adj = cov_arr_adj - r_peturb
            r_peturb = VbadxVbadT * (threshold-current_eigv[iii])
            cov_arr_adj = cov_arr_adj + r_peturb
            print((iii,
                   n_vec-1,
                   current_eigv[iii],
                   threshold-current_eigv[iii],
                   np.max(np.diag(r_peturb))))
    elif method == 'direct':
        dL = threshold - all_eigval[:n_vec]
        if np.any(dL < 0.0):
            # This might be a problem when eigenvalues are re-estimated
            print('Some new estimates to eigenvalue are above threshold.')
            print('No adjustments will be made for those eigenvalues')
            dL = np.diag(np.max([np.zeroes_like(dL), dL], axis=0))
        else:
            dL = np.diag(dL)
        dC = np.matmul(np.matmul(current_eigV, dL), current_eigV.T)
        cov_arr_adj = cov_arr + dC

    print('Computing adjusted eigenvalues, smallest '+str(n_vec))
    new_eigv = linalg_scipy.eigh(cov_arr_adj,
                                 eigvals_only=True,
                                 subset_by_index=[0, n_vec-1])
    new_min_eigv = np.min(new_eigv)
    print('Eigenvalues that were smaller than threshold:')
    print('[:5] : ', new_eigv[:5])
    print('[-5:]: ', new_eigv[-5:])
    print('Smallest eigenvalue=', new_min_eigv)
    new_det = linalg.det(cov_arr_adj)
    print('Determinant=', new_det)
    total_var = np.sum(np.diag(cov_arr_adj))
    meta_dict = {'threshold': threshold,
                 'smallest_eigv': new_min_eigv,
                 'determinant': new_det,
                 'total_variance': total_var}
    return (cov_arr_adj, meta_dict)


#
# Direct EOF chopping
#
def eof_chop(cov_arr, target_explained_variance=0.95):
    #
    if ma.isMaskedArray(cov_arr):
        cov_arr = cov_arr.data
    #
    # Compute all eigenvalues plus other useful diagonstics
    all_eigval = linalg.eigvals(cov_arr)
    all_eigval = np.sort(all_eigval)
    max_eigval = np.max(all_eigval)
    min_eigval = np.min(all_eigval)
    sum_eigval = np.sum(all_eigval)
    p90_index = int(0.10*len(all_eigval))
    p95_index = int(0.05*len(all_eigval))
    n_negatives = np.sum(all_eigval < 0.0)
    sum_of_negatives = np.sum(all_eigval[all_eigval < 0.0])
    sumtop10_eigval = np.sum(all_eigval[-p90_index:])
    sumtop05_eigval = np.sum(all_eigval[-p95_index:])
    top10_explained_var = 100.0 * sumtop10_eigval / sum_eigval
    top05_explained_var = 100.0 * sumtop05_eigval / sum_eigval
    explained_var_from_neg = 100.0 * sum_of_negatives / sum_eigval
    print('Pre-adjusted eigenvalue summary')
    print('Total variance=', sum_eigval)
    print('Largest=', max_eigval)
    print('Smallest/most negative=', min_eigval)
    print('Number of negative eigenvalues=', n_negatives)
    print('Sum=', sum_eigval)
    print('Explained variance from top 10%=', top10_explained_var, '%')
    print('Explained variance from top  5%=', top05_explained_var, '%')
    print('Negative contributions from the negatives=', explained_var_from_neg, '%')
    #
    target_total_variance = target_explained_variance * sum_eigval
    all_eigval_R = all_eigval[::-1]
    eigenvals_2B_included = all_eigval_R[all_eigval_R.cumsum() <= target_total_variance]
    n_eig_2B_included = len(eigenvals_2B_included)
    print('Target explained variance=', target_explained_variance)
    print('aka adjusted total variance=', target_total_variance)
    print('Requiring ', n_eig_2B_included, ' eigenvalues')
    print('Smallest eigenvalue in truncation=', eigenvals_2B_included[-1], '(aka threshold)')
    print('Largest eigenvalue in truncation=', eigenvals_2B_included[0])
    #
    print('Computing eigenval & eigenvec up to the estimated number: '+str(n_eig_2B_included))
    subset_by_index = [len(all_eigval)-n_eig_2B_included,
                       len(all_eigval)-1]
    current_eigv, current_eigV = linalg_scipy.eigh(cov_arr,
                                                   eigvals_only=False,
                                                   subset_by_index=subset_by_index)
    print(current_eigv)
    print(current_eigV.shape)
    #
    # This is new truncated covariance matrix...
    # it will/should have eigenvalues effectively 0
    cov_arr_adj = current_eigV @ np.diag(current_eigv) @ current_eigV.T
    #
    n_vec = 10
    print('Computing adjusted eigenvalues, smallest '+str(n_vec))
    new_eigv = linalg_scipy.eigh(cov_arr_adj,
                                 eigvals_only=True,
                                 subset_by_index=[0, n_vec-1])
    new_min_eigv = np.min(new_eigv)
    new_max_eigv = np.max(new_eigv)
    print('Largest eigenvalue=', new_max_eigv)
    print('Smallest eigenvalue=', new_min_eigv)
    print('Float32 precision=largest eigv x 1E-6=', 1E-6*new_max_eigv, ']')

    new_det = linalg.det(cov_arr_adj)
    print('Determinant=', new_det)
    sum_eigval2 = np.sum(current_eigv)
    print('Actual adjusted total variance after truncation=', sum_eigval2,
          '[Target = ',target_total_variance ,']')
    #
    meta_dict = {'target_explained_variance%': target_explained_variance*100.0,
                 'num_of_retained_eofs': n_eig_2B_included,
                 'threshold': eigenvals_2B_included[-1],
                 'smallest_eigv': new_min_eigv,
                 'largest_eigv': new_max_eigv,
                 'determinant': new_det,
                 'total_variance': sum_eigval2}
    return (cov_arr_adj, meta_dict)


#
# The original eigenvalue clipping method
#
def csum_up_to_val(samples,
                   target,
                   pop=-1,
                   csum=0.0,
                   niter=0,
                   sort_data=False):
    '''
    Find csum and sample index that target is surpassed.
    '''
    if sort_data:
        samples = np.sort(samples)
    if pop not in [-1, 1]:
        raise ValueError('pop must be either at the end (-1) or at start (1)!')
    csum = 0.0
    niter = 0
    nsamples = len(samples)
    while True:
        if pop == 1:
            pop_val = samples[niter]
        else:
            pop_val = samples[niter-1]
        csum += pop_val
        niter += pop
        print(niter, nsamples, pop_val, csum)
        if csum > target:
            break
        assert abs(niter) < nsamples, 'Out of samples!'
    return (csum, niter)


def csum_up_to_val_recursive(samples,
                             target,
                             pop=-1,
                             csum=0.0,
                             niter=0):
    '''
    Recursive and callback solution to csum_up_to_val
    Python has issues with 1000+ recusion. Hence not suitable for large data
    Also no sort... you probably don't want to call np.sort many times...
    '''
    if pop not in [-1, 1]:
        raise ValueError('pop must be either at the end (-1) or at start (1)!')
    if pop == 1:
        csum += samples[0]
    else:
        csum += samples[-1]
    niter += pop
    if csum > target:
        return (csum, niter)
    print(len(samples))
    assert len(samples) > 1, 'Out of samples!'
    if pop == -1:
        return csum_up_to_val(samples[:-1],
                              target,
                              pop=pop,
                              csum=csum,
                              niter=niter)
    return csum_up_to_val(samples[1:],
                          target,
                          pop=pop,
                          csum=csum,
                          niter=niter)


class Laloux_CovarianceClean:
    '''
    https://www.worldscientific.com/doi/abs/10.1142/S0219024900000255
    '''
    def __init__(self, C, clean=True, atol=None):
        print('CovarianceClean __init__')
        print(C.shape)
        self.C = self.clean_small(C, atol=atol) if clean else C
        print((np.max(self.C), np.min(self.C)))

    def clean_small(self, matrix, atol=None):
        if atol is None:
            atol = 1E-5
        small_stuff = np.abs(matrix) < atol
        ans = matrix.copy()
        ans[small_stuff] = 0.0
        return ans

    def find_index_explained_variance(self, eigvals, target=0.95):
        total_variance = np.sum(eigvals)
        target_explained_variance = target * total_variance
        print((total_variance, target_explained_variance))
        __, i2goal = csum_up_to_val(eigvals, target_explained_variance)
        return i2goal

    def find_index_aspect_ratio(self, eigvals, N=37000, T=41*6):
        '''
        Defaults are based on 41 years of pentad ESA data...
        it does not work in general
        '''
        __, threshold = self.estimate_threshold(N, T)
        i2goal = -np.sum(eigvals > threshold)
        return i2goal

    def estimate_threshold(self, N, T):
        '''
        See 7.2.2 in https://doi.org/10.1016/j.physrep.2016.10.005
        Eigenvalue threshold: threshold = (1.0 + SQRT(q))**2
        Below calculates q and threshold
        '''
        q = N/T
        if q < 1.0:
            q = 1.0/q
        threshold = (1.0 + np.sqrt(q))**2.0
        return (q, threshold)

    def eig_clip(self,
                 method='explained_variance',
                 method_parms={'target': 0.95}):
        '''
        Denoise symmetric damaged covariance/correlation matrix C
        by clipping eigenvalues
        This is done by KEEPING the largest explained variance
        in which (number of basis vectors to be kept) >> (number of rows)
        In ESA data, keeping 95% variance means keeping top ~15% of the
        eigenvalues
        '''
        print('Solving eigenvalues and vectors')
        self.eigvals, eigvec = linalg_scipy.eigh(self.C)
        sorted_order = np.argsort(self.eigvals)
        self.eigvals = self.eigvals[sorted_order]
        print('[:5] =', self.eigvals[:5])
        print('[-5:] =', self.eigvals[-5:])
        eigvecs = eigvec[:, sorted_order]
        self.n_eigvals = len(self.eigvals)
        thresh_ms = {'explained_variance': self.find_index_explained_variance,
                     'Laloux_2000': self.find_index_aspect_ratio}
        self.i2keep_m = method
        self.i2keep_m_kwargs = method_parms
        self.i2keep = thresh_ms[self.i2keep_m](self.eigvals, **self.i2keep_m_kwargs)
        self.i2clip = self.n_eigvals + self.i2keep  # Note i2keep is NEGATIVE
        print('Numbers of kept eigenvalues = ', -self.i2keep)
        print('Numbers of clipped eigenvalues = ', self.i2clip)
        #
        # The total variance should be perserved after clipping
        # within precision error of the eigenvalues which is O(Max(Eig) * float_accuracy)
        self.total_var = np.sum(self.eigvals)
        self.var_explained_by_i2keep = np.sum(self.eigvals[self.i2keep:])
        self.unexplained_var = self.total_var - self.var_explained_by_i2keep
        self.avg_eigenvals_4_unexplained = self.unexplained_var / self.i2clip
        #
        # Find eigenvectors associated up to i2keep
        self.new_eigvals = self.eigvals.copy()
        self.new_eigvals[:self.i2keep] = self.avg_eigenvals_4_unexplained
        self.C_hat = eigvecs @ np.diag(self.new_eigvals) @ eigvecs.T


def main():
    ''' Main - keep calm and does nothing! '''
    print('--- Main ---')


if __name__ == "__main__":
    main()
