# ellipse_estimation

Workflow:

1. Fit ellipses where there are observations
    1. Run script: run_scripts/ellipse_mle_*/fit_ellipse_fivedegree_monthly/process_basin_satellite_monthly_climatology_matern_distances_*
    2. Model: cube_covariance.py
2. In-fill missing ellipse parameters (required for global kriging)
    1. Run scripts: run_scripts/general/coast_lake_pixels_scale_infilling/infill_coasts_and_lakes.py
3. Stich the ellipses into covariances
    1. Run script: run_scripts/ellipse_mle_*/stiching/API_mask_*_cov_5p0_*.py
    2. Model: cube_covariance_nonstationary_stich.py
4. Check positive definite, repair if needed
    1. Run script: run_scripts/ellipse_mle_*/clipping/clip_*_cov.*
    2. Model: repair_damaged_covariance.py
5. Optional step to make covariance -180 180 to 0 360
    1. Run script: run_scripts/general/remap_m180_180_2_0_360.py


Other code:
1. cube_io_10x10.py:
    1. Region look up table
    2. Some I/O tools
    3. (Old) file look up table
2. simulate_ellipse.py: 
    1. Multivariate normal distribution simulations
    2. Construct artifical data using ellipse parameters
    3. Used in certain unit tests and can conduct idealised covariance simulations
3. distance_util.py
    1. Some distance computing functions
