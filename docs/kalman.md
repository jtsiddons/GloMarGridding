# Local autoregressive forecast and basic Kalman filtering

## Introduction and purpose

Most marine datasets are what we called a 2DVAR analysis - the current analysis is based on the present observations only (2D as in x, y; in contrast of 3DVAR seen in historical weather forecasting literature: x, y, z).

This limits the amount of the observations one can used, resulting in higher uncertainty. In many cases, this is mitigated by temporally aggregating the observations (like doing monthly analysis instead of daily analysis) or blending multiple input sources (like high-level remote sensing data product). Sometimes one is stuck with such spotty observations; like one really wants a daily analysis based purely on daily observations. How we can reduce the uncertainties?

This leads to the introduction of time dimension into the analysis. This is what we called 4DVAR (x, y, z, t) weather forecasting (or 3DVAR if one is dealing with only surface variables - x, y, t).

A popular way to do so is via Kalman Filter -- a recursive method that blends a forecast to the present state (aka prior) with present observations, resulting a posterior analysis that is more accurate with just the forecast or the observations. The method is widely used in autopilot and positioning. For Earth sciences, they are often associated with numerical weather prediction and reanalyses. Kalman filter has been used in producing more accurate weather forecasts and historical reanalysis by the ECWMF and the UK Met Office (https://www.met.reading.ac.uk/~darc/training/ecmwf_collaborative_training/EnKF_AFowler.pdf).

# Producing the prior using a simple autoregressive statistical model

This lead to the question how would one produce a forecast priors. The priors/first guesses are often based on outputs of dynamical models like in numerical weather prediction and reanalysis; even in satnav or autopilot, simple dynamics are included (dead-reckoning along roads). Such dynamics are often not part of the development of the gridded observations; GlomarGridding is a package for statistical modelling only.

A simple statistical approach is implemented here to produce such forecast, basing on the simple idea is that current condition (described by the variable `a` below) will generally tend toward climatology (`E(a)`). The simplest way to model that behavior is the local 1st order autoregressive model, in which the expected value for `a(t)` is a linear function of the observed lag-1 correlation and its last observed value (`a(t-1)`), resulting in exponential relaxation toward `E(a)`, plus a normal-distributed uncertainty epsilon:

$$
a_{t} = \Phi (a_{t-1} - E(a)) + \epsilon
$$

This is often referred as the (non-vectored) [AR1 model](https://en.wikipedia.org/wiki/Autoregressive_model). More sophisticated approach can be used like full vectorised form (requires full lagged autocovariance matrix), multi-lag, or autoregressive moving average (ARMA).

The AR1 model produces its own uncertainty. Input uncertainties can be incorporated following standard propagation of error rules. This is essential as uncertainty lie in the heart of Kalman Filter. Let say Phi is a diagonal matrix with local lag-1 correlation, Sigma(climatology) is the diagonal matrix with climatological variance (no off-diagonal contribution; this requires full vectorised AR1), Sigma t is a diagonal matrix of the diagonal of error covariance for the current observations, the full uncertainty can be estimated as:

$$
\Sigma_{\text{AR1}} \sim MVN(0, \Phi\Sigma_{t}\Phi + \sqrt{(\textbf{1}-\Phi\Phi)}\Sigma_\text{climatology}\sqrt{(\textbf{1}-\Phi\Phi)})
$$

Work in progress: Ideally, this should be done in vectorised form.

[Wikipedia intro](https://en.wikipedia.org/wiki/Vector_autoregression)

This requires a modeled or computed lagged autocovariance and contemporary covariance; the latter can use the `ellipse` covariance (part of `GlomarGridding`... former not yet!). The predicted value for a point would now depends on values of other points. This also require computation of a weight matrix that is similar to Kriging with full error covariance that is contains off-diagonal values.

# Uncertainty weighted average (Kalman Filter)

Kalman Filter, in its heart, is a weighted average -- an average for two input streams: the prior/first-guess/forecast what the current state probably be, and what the most recent observations say. The weights are determined only by uncertainties as neither the first-guess and observations are perfect. In simple English, produce a better observations by taking multiple sources, weighting in favour the more accurate one.

Let say the uncertainty of the forecast and observations can be described by two different uni- or multi-variate zero-averaged normal distribution with some scalar variance / error covariance matrix.

$$
\epsilon_{\text{forecast}} \sim N(0, \sigma_{\text{forecast}}^2) \text{or} MVN(0, \Sigma_{\text{forecast}})
$$

$$
\epsilon_{\text{obs}} \sim N(0, \sigma_{\text{obs}}^2) \text{or} MVN(0, \Sigma_{\text{obs}})
$$

Assuming observations are on the same grid as the forecast (true for the exercise), the Kalman gain (weight for new observations) is:

Univariate:

$$
K = \frac{\sigma_{\text{forecast}}^2  (sigma_{\text{forecast}}^2 + \sigma_{\text{obs}}^2)}
$$

Multi-variate:

$$
K = \Sigma_{\text{forecast}} (\Sigma_{\text{forecast}} + \sigma_{\text{obs}})^{-1} = \sigma_{\text{obs}}^{-1} (\Sigma_{\text{forecast}}^{-1} + \sigma_{\text{obs}}^{-1})^{-1}
$$

in which K is solution to the following linear equation, this is how it is solved in the code using `numpy.linalg.solve`. No actual error covariance inversions is needed.

$$
(\Sigma_{\text{forecast}} + \sigma_{\text{obs}}) K^{T} = \Sigma_{\text{forecast}}
$$

noting that Sigma is symmetrical but K is not (similar to Kriging weights).

Once the weights are known, the posterior analysis is a weighted average of forecast and observations

$$
\tilde{a}(t) = K a_{\text{obs}} + (I - K)  a_{\text{forecast}} = K (a_{\text{obs}} - a_{\text{forecast}} ) +  a_{\text{forecast}}
$$

with posterior error covariance of:

$$
\Sigma_{\text{analysis}} = (\textbf{I} - \textbf{K}) \Sigma_{\text{forecast}}}
$$

If observations are not on the same grid as the forecast, additional mapping multiplications are needed; that leads the usual form one will see in text books or Wikipedia.

The workflow here will be:

1. Produce an initial t=0 analysis
2. Do a t+1 forecast
3. Krige the observations for t+1
4. Blend the forecast with Kriging result using Kalman filter
5. Go back to step 2 using the posterior analysis and error covariance

# Code and usage

The main processing involve two Python code

- `forecast_linear_ar1.py`
- `inv_sigma_wgts.py`

There is a helping code file called `cov_diagonal.py` which can be used to separate the analysis between we expect to not to be spatially correlated with bits that are spatially correlated.

For the all the main classes, the computation uses a method that is called `compute_XXXXX`.

## `forecast_linear_ar1.py`

This does the local 1st-order linear autoregressive forecast.

There is only one class in the code. All the processing (methods) needed are within the class. Only the local AR1 is implemented for now.

### Autoregressive1Forecast

- `obs`: Kriging outputs that serves as inputs to this class
- `errcov_obs`: Uncertainty to `obs`. Can be matrix or vector depending on usage. Current version will force it 1D vector as vector AR1 is not yet implemented.
- `lag_1_autocov`: A 1D vector of lag-1 correlation (or 2D full autocovariance matrix... when that is implemented; class will force it to 1D for now)
- `clim_mean`: The climatological mean for `obs`; while `obs` is usually already in anomalies (0-mean relative to some arbitrary baseline), this allows one to redefine that to a current quasi-stationary timeslice)
- `clim_covar`: The climatological covariance. Can be 1 or 2D depending on usage. Current version will force it to 1D vector as vector AR1 is not yet implemented.
- `errcov_obs_is_sdev`: `climatology_variance` is standard deviation. This defaults to False. Only works for 1D. The class will square the values for you. This is here to make sure you check your variable's units; did you use squared or non-squared values!?
- `clim_covar_is_sdev`: `clim_covar` is standard deviation. This defaults to False. Only works for 1D. The class will square the values for you.
- `predict_local`: Forces local AR1 forecasts. This is only implemented method for now. Like it or not, it will be forced to True (!) for now.

.. autoclass:: glomar_gridding.autoregressive_kalman.forecast_linear_ar1.Autoregressive1Forecast

## `inv_sigma_wgts.py`

For Kalman Filtering, using inverse error covariance weighting (`sigma`) weights.

### KalmanOut

Kalman filtering if observations and forecast are on the same grid. This class and its computation is general and is left open for different types of observations and prior forecast as long as the grid is the same.

- `forecast_vector`: The vector of forecasts, such as outputs from `Autoregressive1Forecast`.
- `obs_vector`: The vector of observations, such as outputs from `OrdinaryKriging`.
- `errcov_forecast`: Error covariance of the forecast/prior - for the purpose here, this will be the error covariance from `Autoregressive1Forecast`, but future development may render that obsolete.
- `errcov_obs`: Error covariance of the observations - normally, this will be the error covariance from the Kriging analysis.
- `cov_forecast_and_obs`: The covariance between the forecast and observations.
- `ez_covariances`: This disables the use of off-diagonal/correlated components of the error covariances.

.. autoclass:: glomar_gridding.autoregressive_kalman.inv_sigma_wgts.KalmanOut

### KalmanOutUncorrCorrSplit

For most part, this works the same way as KalmanOut, but it is possible to specify which bit one wants to use `ez_covariances` mode in which bit one does not. In the past, GlomarGridding Kriging classes are generally used to produce global outputs including estimates into terrain that needed to be treated differently (marine temperatures onto sea ice or land vice versa). This can be used to filter such results automatically.

The only addition `arg` is:

- `arr_2_decide_if_points_are_isolated`: this works with the type of covariances produced by ellipse classes which allows expanding covariances globally. If one passes the post-expanded global spatial covariance into this, methods within `KalmanOutUncorrCorrSplit` auto-detects which rows/columns are filled in by looking for diagonal rows, and conducts Kalman-Filtering separately using the `ez_covariances` mode.

.. autoclass:: glomar_gridding.autoregressive_kalman.inv_sigma_wgts.KalmanOutUncorrCorrSplit
