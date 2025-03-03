#!/bin/csh
setenv OMP_NUM_THREADS 8
setenv OPENBLAS_NUM_THREADS 8
setenv MKL_NUM_THREADS 8
setenv VECLIB_MAXIMUM_THREADS 8
setenv NUMEXPR_NUM_THREADS 8
python -u API_mask_esa_cov_5p0_lsat.py 1 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 2 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 3 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 4 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 5 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 6 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 7 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 8 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 9 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 10 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 11 1.5 0 0 1
python -u API_mask_esa_cov_5p0_lsat.py 12 1.5 0 0 1
