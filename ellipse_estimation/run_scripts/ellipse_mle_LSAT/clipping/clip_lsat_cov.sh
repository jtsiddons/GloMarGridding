#!/bin/csh
setenv OMP_NUM_THREADS 8
setenv OPENBLAS_NUM_THREADS 8
setenv MKL_NUM_THREADS 8
setenv VECLIB_MAXIMUM_THREADS 8
setenv NUMEXPR_NUM_THREADS 8
python -u clip_lsat_cov.py
echo 'Complete'