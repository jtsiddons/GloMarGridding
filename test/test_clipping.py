import pytest
import iris
import numpy as np
from scipy import linalg
from ellipse_estimation import repair_damaged_covariance as rdc
from ellipse_estimation import simulate_ellipse as s_e

sst_cov_path = '/noc/mpoc/surface_data/ESA_CCI5deg_month_extra/ANOMALY/SpatialScales/locally_build_covariances/'

@pytest.mark.parametrize(
    "unclipped, clipped",
    [
        (sst_cov_path+'covariance_09_v_eq_1p5_sst_without_psd_check.nc', 
         sst_cov_path+'covariance_09_v_eq_1p5_sst_clipped.nc'), 
    ]
)
def test_clipping(unclipped, clipped):
    unclipped_cov_cube = iris.load_cube(unclipped, 'covariance')
    print(unclipped_cov_cube.data)
    print(np.trace(unclipped_cov_cube.data))
    clipped_cov_cube = iris.load_cube(clipped, 'covariance')
    print(clipped_cov_cube.data)
    print(np.trace(clipped_cov_cube.data))
    #
    # Correlation based clipping
    cleaner = rdc.Laloux_CovarianceClean(unclipped_cov_cube.data)
    #
    print('Test 1')
    cleaned_cov_via_cor_ev95 = cleaner.eig_clip_via_cor()
    print(cleaned_cov_via_cor_ev95)
    print(np.trace(cleaned_cov_via_cor_ev95))
    print(np.abs(np.trace(cleaned_cov_via_cor_ev95) - np.trace(unclipped_cov_cube.data)),
          np.abs(np.trace(cleaned_cov_via_cor_ev95) - np.trace(unclipped_cov_cube.data)) < 1E-5)
    assert np.abs(np.trace(cleaned_cov_via_cor_ev95) - np.trace(unclipped_cov_cube.data)) < 1E-5
    #
    print('Test 2')
    cleaned_cov_via_cor_L20 = cleaner.eig_clip_via_cor(method='Laloux_2000', 
                                                       method_parms={'N': unclipped_cov_cube.shape[0],
                                                                     'T': 40})
    print(cleaned_cov_via_cor_L20)
    print(np.trace(cleaned_cov_via_cor_L20))
    print(np.abs(np.trace(cleaned_cov_via_cor_L20) - np.trace(unclipped_cov_cube.data)),
          np.abs(np.trace(cleaned_cov_via_cor_L20) - np.trace(unclipped_cov_cube.data)) < 1E-5)
    assert np.abs(np.trace(cleaned_cov_via_cor_L20) - np.trace(unclipped_cov_cube.data)) < 1E-5
    #
    # This is the method used in the ellipse covariances
    print('Test 3')
    cleaned_cov_via_cov_ev95 = cleaner.eig_clip_via_cov()
    print(cleaned_cov_via_cov_ev95)
    # Check explained variance to be the same
    print('Check for nearly unchanged explained variance')
    print(np.trace(cleaned_cov_via_cov_ev95))
    print(np.abs(np.trace(cleaned_cov_via_cov_ev95)/np.trace(unclipped_cov_cube.data)-1.0),
          np.abs(np.trace(cleaned_cov_via_cov_ev95)/np.trace(unclipped_cov_cube.data)-1.0) < 1E-5)
    assert np.abs(np.trace(cleaned_cov_via_cov_ev95)/np.trace(unclipped_cov_cube.data)-1.0) < 1E-5
    # Check no negative eigenvalues remain
    print('Check all eigenvalues are positive')
    eigv5 = linalg.eigh(cleaned_cov_via_cov_ev95, eigvals_only=True, subset_by_index=[0,4])
    print(f'5 smallest eigenvalues = {eigv5}')
    assert np.all(eigv5 > 0)
    # This is just checking, we know they should be the same!
    print('Final Check: against a known answer')
    print(cleaned_cov_via_cov_ev95)
    print(clipped_cov_cube.data)
    max_diff = np.max(np.abs(cleaned_cov_via_cov_ev95 - clipped_cov_cube.data))
    print(max_diff)
    assert np.max(np.abs(cleaned_cov_via_cov_ev95 - clipped_cov_cube.data)) < 1E-5
    W, p_val = s_e.chisq_test_using_likelihood_ratios_4_covariance(cleaned_cov_via_cov_ev95,
                                                                   clipped_cov_cube.data,
                                                                   40)
    print(W, p_val)
    assert p_val > 0.99
