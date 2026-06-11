Auto Regressive Forecast
------------------------

Introduction and purpose
========================

Most marine datasets are what we called a 2DVAR analysis - the current analysis is based on the
present observations only (2D as in x, y; in contrast of 3DVAR seen in historical weather
forecasting literature: x, y, z).

This limits the amount of the observations one can used, resulting in higher uncertainty. In many
cases, this is mitigated by temporally aggregating the observations (like doing monthly analysis
instead of daily analysis) or blending multiple input sources (like high-level remote sensing data
product). Sometimes one is stuck with such spotty observations; like one really wants a daily
analysis based purely on daily observations. How we can reduce the uncertainties?

This leads to the introduction of time dimension into the analysis. This is what we called 4DVAR (x,
y, z, t) weather forecasting (or 3DVAR if one is dealing with only surface variables - x, y, t).

A popular way to do so is via Kalman Filter -- a recursive method that blends a forecast to the
present state (aka prior) with present observations, resulting a posterior analysis that is more
accurate with just the forecast or the observations. The method is widely used in autopilot and
positioning. For Earth sciences, they are often associated with numerical weather prediction and
reanalyses. Kalman filter has been used in producing more accurate weather forecasts and historical
reanalysis by the ECWMF and the UK Met Office
(https://www.met.reading.ac.uk/~darc/training/ecmwf_collaborative_training/EnKF_AFowler.pdf).

Producing the prior using a simple autoregressive statistical model
===================================================================

This lead to the question how would one produce a forecast priors. The priors/first guesses are
often based on outputs of dynamical models like in numerical weather prediction and reanalysis; even
in satnav or autopilot, simple dynamics are included (dead-reckoning along roads). Such dynamics are
often not part of the development of the gridded observations; GlomarGridding is a package for
statistical modelling only.

A simple statistical approach is implemented here to produce such forecast, basing on the simple
idea is that current condition (described by the variable `a` below) will generally tend toward
climatology (:math:`E(a)`). The simplest way to model that behavior is the local 1st order autoregressive
model, in which the expected value for :math:`a(t)` is a linear function of the observed lag-1 correlation
and its last observed value (:math:`a(t-1)`), resulting in exponential relaxation toward :math:`E(a)`, plus a
normal-distributed uncertainty epsilon:

.. math::
    a_{t} = \Phi (a_{t-1} - E(a)) + \epsilon

This is often referred as the (non-vectored) `AR1 model
<https://en.wikipedia.org/wiki/Autoregressive_model>`__ . More sophisticated approach can be used
like full vectorised form (requires full lagged autocovariance matrix), multi-lag, or autoregressive
moving average (ARMA).

The AR1 model produces its own uncertainty. Input uncertainties can be incorporated following
standard propagation of error rules. This is essential as uncertainty lie in the heart of Kalman
Filter. Let say :math:`\Phi` is a diagonal matrix with local lag-1 correlation, :math:`\Sigma_\text{climatology}`
is the diagonal matrix with climatological variance (no off-diagonal contribution; this requires full
vectorised AR1), :math:`\Sigma_{t}` is a diagonal matrix of the diagonal of error covariance for the current
observations, the full uncertainty can be estimated as:

.. math::
    \epsilon_{\text{AR1}} \sim \text{MVN}(0, \Phi\Sigma_{t}\Phi^T +
    \Sigma_\text{climatology}-\Phi\Sigma_\text{climatology}\Phi^T)

The above form is correct even for weights that has off-diagonal terms. In that case, :math:`\Phi` is no longer a
diagonal autocorrelation matrix but the weights for fully vectorised multi-variate model :math:`\mathbf{W}`;
:math:`\mathbf{W} = \Phi` is a merely a special case.

.. math::
    a_{t} = \mathbf{W} (a_{t-1} - E(a)) + \epsilon

In the local and diagonal case, the above is equivalent to the below at each grid point, like the equations that one can
find in Chapter 8 of [Wilks]:

.. math::
    \epsilon_{\text{AR1}} \sim \text{N}(0, \phi^2\sigma^2_{t} +
    (1-\phi^2)\sigma^2_\text{climatology})

.. Work in progress: Ideally, this should be done in vectorised form.

`Wikipedia intro <https://en.wikipedia.org/wiki/Vector_autoregression>`__

`Wikipedia intro <https://en.wikipedia.org/wiki/Lasso_(statistics)>`__

For the diagonal :math:`\Phi` case, the lagged autocorrelation is easy to compute from observations; its application
to statistical forecasting is highly stable and quick. The weights for the multi-variate correlated case are
much more complicated. It is usually estimated by fitting each grid point with a regression model, taking advantage of
the seemingly unrelated regression of vector autoregression. For high resolution
datasets, fitting such models often results in overfitting and low predictive power. This can be mitigated using
regularized regression. The one implemented here uses LASSO regression [Tibshirani_Lasso] via ``scikit-learn``;
LASSO generally leads to sparse :math:`\mathbf{W}`, makes them easier to handle memory and computational cost via
sparse matrix classes and function within ``scipy``. What LASSO does is that it adds a penalty term to the
the least squares cost function that is proportional to the sum of the absolute values of all regression coefficients. The
sum is scaled by a tunable hyperparameter; it is usually called :math:`\lambda` in statistics literature (including the Wikipedia
article) but is called :math:`\alpha` instead in ``scikit-learn`` documentation.

.. This requires a modeled or computed lagged autocovariance and contemporary covariance; the latter
.. can use the `ellipse` covariance (part of `GlomarGridding`... former not yet!). The predicted value
.. for a point would now depends on values of other points. This also require computation of a weight
.. matrix that is similar to Kriging with full error covariance that is contains off-diagonal values.

Uncertainty weighted average (Kalman Filter)
============================================

Kalman Filter, in its heart, is a weighted average -- an average for two input streams: the
prior/first-guess/forecast what the current state probably be, and what the most recent observations
say. The weights are determined only by uncertainties as neither the first-guess and observations
are perfect. In simple English, produce a better observations by taking multiple sources, weighting
in favour the more accurate one.

Let say the uncertainty of the forecast and observations can be described by two different uni- or
multi-variate zero-averaged normal distribution with some scalar variance / error covariance matrix.

.. math::
    \epsilon_{\text{forecast}} \sim N(0, \sigma_{\text{forecast}}^2) \text{or} MVN(0,
    \Sigma_{\text{forecast}})

.. math::
    \epsilon_{\text{obs}} \sim N(0, \sigma_{\text{obs}}^2) \text{or} MVN(0, \Sigma_{\text{obs}})

Assuming observations are on the same grid as the forecast (true for the exercise), the Kalman gain
(weight for new observations) is:

Univariate:

.. math::
    K = \frac{\sigma_{\text{forecast}}^2 } { \sigma_{\text{forecast}}^2 + \sigma_{\text{obs}}^2}

Multi-variate:

.. math::
    K = \Sigma_{\text{forecast}} (\Sigma_{\text{forecast}} + \sigma_{\text{obs}})^{-1} =
    \sigma_{\text{obs}}^{-1} (\Sigma_{\text{forecast}}^{-1} + \sigma_{\text{obs}}^{-1})^{-1}

in which K is solution to the following linear equation, this is how it is solved in the code using
`numpy.linalg.solve`. No actual error covariance inversions is needed.

.. math::
    (\Sigma_{\text{forecast}} + \sigma_{\text{obs}}) K^{T} = \Sigma_{\text{forecast}}

noting that Sigma is symmetrical but :math:`K` is not (similar to Kriging weights).

Once the weights are known, the posterior analysis is a weighted average of forecast and observations

.. math::
    \tilde{a}(t) = K a_{\text{obs}} + (I - K)  a_{\text{forecast}} = K (a_{\text{obs}} -
    a_{\text{forecast}} ) +  a_{\text{forecast}}

with posterior error covariance of:

.. math::
    \Sigma_{\text{analysis}} = (\textbf{I} - \textbf{K}) \Sigma_{\text{forecast}}

If observations are not on the same grid as the forecast, additional mapping multiplications are
needed; that leads the usual form one will see in text books or Wikipedia.

The workflow here will be:

1. Produce an initial :math:`t=0` analysis
2. Do a t+1 forecast
3. Krige the observations for :math:`t+1`
4. Blend the forecast with Kriging result using Kalman filter
5. Go back to step 2 using the posterior analysis and error covariance

Code and usage
==============

The main processing involve two modules

- `autoregresive_kalman.forecast_linear_ar1`
- `autoregresive_kalman.inv_sigma_wgts`

There is a helping code module called `autoregressive_kalman.cov_diagonal` which can be used to
separate the analysis between we expect to not to be spatially correlated with bits that are
spatially correlated.

For the all the main classes, the computation uses a method that is called `compute_XXXXX`.

`autoregressive_kalman.forecast_linear_ar1`
===========================================

.. autoclass:: glomar_gridding.autoregressive_kalman.forecast_linear_ar1.Autoregressive1ForecastUncorr
   :members:

.. autoclass:: glomar_gridding.autoregressive_kalman.forecast_linear_ar1.Autoregressive1ForecastVector
   :members:

`autoregressive_kalman.inv_sigma_wgts`
======================================

For Kalman Filtering, using inverse error covariance weighting (:math:`sigma`) weights.

.. autoclass:: glomar_gridding.autoregressive_kalman.inv_sigma_wgts.KalmanOut
   :members:

.. autoclass:: glomar_gridding.autoregressive_kalman.inv_sigma_wgts.KalmanOutUncorrCorrSplit
   :members:

`autoregressive_kalman.cov_diagonal`
====================================

.. automodule:: glomar_gridding.autoregressive_kalman.cov_diagonal
   :members:

`autoregressive_kalman.lasso_estimate`
======================================

.. autoclass:: glomar_gridding.autoregressive_kalman.lasso_estimate.LassoEstimate_AR1
   :members:

`autoregressive_kalman.lasso_postprocessing`
============================================

.. autoclass:: glomar_gridding.autoregressive_kalman.lasso_postprocessing.LassoWeights
   :members:

.. autoclass:: glomar_gridding.autoregressive_kalman.lasso_postprocessing.LassoError
   :members:

`autoregressive_kalman.compute_autocorrelation`
====================================

.. automodule:: glomar_gridding.autoregressive_kalman.compute_autocorrelation
   :members:
