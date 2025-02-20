#!/bin/sh
#SBATCH -p par-single
#SBATCH --array=[1-12]
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --job-name=matern_global_lsat
#SBATCH --output=/work/scratch-pw2/schan016/NOC-hostace/ESA_CCI5deg_month/logs/matern_global_lsat_%A_%a.out
#SBATCH --error=/work/scratch-pw2/schan016/NOC-hostace/ESA_CCI5deg_month/logs/matern_global_lsat_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --mem=2048
##
## Execution the Python script
python -u ../../ellipse_mle/fit_ellipse_fivedegree_monthly/process_basin_satellite_monthly_climatology_matern_physical_distances_Global.py lsat ${SLURM_ARRAY_TASK_ID} 1.5 anistropic_rotated_pd
##
## original config
## -n 4 -N 1 --time=24:00:00
