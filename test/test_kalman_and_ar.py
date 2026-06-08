"""
Kalman filter test:
- Based on examples: https://kalmanfilter.net/ 

AR1: uses manually checkable result + modified result from example 8.3 in Wilks 2006:
- 3 points with Lag1-autocorrelation of 0.7, 0.6, 0.67
- climatological value at each point 1.0, -1.0, 20.23
- climatological sigma at each point 2.0, 3.0, 8.961 (4.0, 9.0, 80.292 variance)
- start value 0.0, 0.0, 25.0 for
- Correct answers:
- (0.7 x (-1.0 - 0.0)) + 1.0) = 0.3
- (0.6 x (1.0 - 0.0)) + (-1.0) = 0.4
- (0.67 x (25.0 - 20.23)) + 20.23 = 23.4259
- variance:
- (1 - 0.7 x 0.7) x 4.0 = 2.04
- (1 - 0.6 x 0.6) x 9.0 = 5.76
- (1 - 0.67 x 0.67) x 80.292 = 44.24 (pg 356 example 8.1)

VAR1:
- roughly based on example 64 in https://faculty.washington.edu/ezivot/econ584/notes/varModels.pdf

Also always check outputted error covariance are symmetric
"""

import numpy as np
import scipy as sp

from glomar_gridding.autoregressive_kalman import (
    inv_sigma_wgts,
    forecast_linear_ar1,
)


def test_kalman():
    """Iteration 1 in https://kalmanfilter.net/"""
    #
    # Approximate (rounded) correct answers
    correct_kalman_gain = np.array([
        [0.4048, 0.6377],
        [0.0399, 0.3144],
    ])  # K(1,1)
    correct_z11 = np.array([11009.37, 201.43])  # x_hat(1,1)
    correct_r11 = np.array([
        [14.57, 1.43],
        [1.43, 0.71],
    ])  # P(1,1)
    #
    z0 = np.array([11000.0, 200.0])  # x_hat(1, 0)
    r0 = np.array([
        [28.5, 3.75],
        [3.75, 1.25],
    ])  # P(1,0)
    z1 = np.array([11020.0, 202.0])  # z(1)
    r1 = np.diag([36.0, 2.25])  # R(1)
    #
    # Compute
    kalman_test = inv_sigma_wgts.KalmanOut(
        z0, z1, r0, r1, None, use_diag_only=False,
    )
    kalman_test.compute_outputs()
    #
    # Assert check
    assert np.all(np.isclose(
        np.round(kalman_test.kalman_gain_from_new_obs, 4),
        correct_kalman_gain,
    ))
    assert np.all(np.isclose(
        np.round(kalman_test.wgt_mean, 2),
        correct_z11,
    ))
    assert np.all(np.isclose(
        np.round(kalman_test.errcov, 2),
        correct_r11,
    ))

def test_ar():
    """
    AR1: uses manually checkable result + modified example 8.3 in Wilks 2006:
    - 3 points with Lag1-autocorr        0.7,  0.6,  0.67
    - start value at each point          0.0,  0.0, 25.00
    - climatological value at each point 1.0, -1.0, 20.23
    - climatological variance at each pt 4.0,  9.0, 80.292
    - (sigma: 2.0, 3.0, 8.961)
    - Correct answers:
    - (0.7 x (0.0 - 1.0)) + 1.0)      =  0.3
    - (0.6 x (0.0 + 1.0)) - 1.0)      = -0.4
    - (0.67 x (25.0 - 20.23)) + 20.23 ~ 23.43 (rounded to 2)
    - variance:
    - (1 - 0.7 x 0.7) x 4.0      = 2.04
    - (1 - 0.6 x 0.6) x 9.0      = 5.76
    - (1 - 0.67 x 0.67) x 80.292 ~ 44.25 (pg 356 example 8.1) (rounded to 2)
    """
    phi = np.array([0.7, 0.6, 0.67])
    climatology = np.array([1.0, -1.0, 20.23])
    errcov_analysis = np.zeros((3, 3))
    errcov_analysis_non_zero = np.diag([0.36, 0.36, 0.36])
    variance = np.array([4.0, 9.0, 80.292])
    sigma = np.sqrt(variance)
    z0 = np.array([0.0, 0.0, 25.0])
    #
    t1 = forecast_linear_ar1.Autoregressive1ForecastUncorr(
        z0,
        errcov_analysis,
        phi,
        climatology,
        variance,
    )
    t1.compute_forecast_local()
    correct_z01 = np.array([0.3, -0.4, 23.43])
    correct_var = np.diag([2.04, 5.76, 44.25])
    assert np.all(np.isclose(np.round(t1.forecast, 2), correct_z01))
    assert np.all(np.isclose(np.round(t1.errcov, 2), correct_var))
    #
    t2 = forecast_linear_ar1.Autoregressive1ForecastUncorr(
        z0,
        np.diag(errcov_analysis),
        phi,
        climatology,
        sigma,
        errcov_analysis_is_sdev=True,
        clim_covar_is_sdev=True,
    )
    t2.compute_forecast_local()
    assert np.all(np.isclose(np.round(t2.forecast, 2), correct_z01))
    assert np.all(np.isclose(np.round(t2.errcov, 2), correct_var))
    #
    # Add 0.36 (0.6 ** 2) to z0 error
    t3 = forecast_linear_ar1.Autoregressive1ForecastUncorr(
        z0,
        errcov_analysis_non_zero,
        phi,
        climatology,
        variance,
    )
    t3.compute_forecast_local()
    # Note: correct_var is already rounded..., so need to round the other bit too
    correct_var_new = (
        correct_var +
        np.round(np.diag(phi) @ errcov_analysis_non_zero @ np.diag(phi), 2)
    )
    assert np.all(np.isclose(np.round(t3.forecast, 2), correct_z01))
    assert np.all(np.isclose(np.round(t3.errcov, 2), correct_var_new))


def test_var():
    """
    Loosely based on example 64 in (weights are changed):
    https://faculty.washington.edu/ezivot/econ584/notes/varModels.pdf
    Should produce answer
    -0.7 + 0.7 x (-1.0 - (-0.7)) + 0.2 x (1.0 - 1.3) = -0.97
     1.3 + 0.0 x (-1.0 - (-0.7)) + 0.7 x (1.0 - 1.3) =  1.09
    """
    climatology = np.array([-0.7, 1.3])
    W = np.array([
        [0.7, 0.2],
        [0.0, 0.7],
    ])
    W = sp.sparse.csc_array(W)
    z0 = np.array([-1.0, 1.0])
    clim_covar = np.diag([0.6, 0.6])
    #
    t1 = forecast_linear_ar1.Autoregressive1ForecastVector(
        z0,
        np.zeros((2, 2)),
        W,
        climatology,
        clim_covar,
    )
    t1.compute_forecast_vector(
        check_wgt_stability=False,
        check_errcov_psd=False,)
    correct_ans = np.array([-0.97, 1.09])
    correct_errcov = clim_covar - W @ clim_covar @ W.T
    assert np.all(t1.forecast == correct_ans)
    assert np.all(np.isclose((t1.errcov - t1.errcov.T), np.zeros((2, 2))))
    assert np.all(correct_errcov, t1.errcov)
