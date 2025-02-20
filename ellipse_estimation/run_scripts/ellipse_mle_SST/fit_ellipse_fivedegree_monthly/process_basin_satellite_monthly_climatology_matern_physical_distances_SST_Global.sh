#!/bin/sh
#SBATCH -p par-single
#SBATCH --array=[1-12]
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --job-name=matern_global_sst
#SBATCH --output=/work/scratch-pw2/schan016/NOC-hostace/ESA_CCI5deg_month/logs/matern_global_sst_%A_%a.out
#SBATCH --error=/work/scratch-pw2/schan016/NOC-hostace/ESA_CCI5deg_month/logs/matern_global_sst_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=2048
##
## Execution the Python script
python -u process_basin_satellite_monthly_climatology_matern_physical_distances_Global.py sst ${SLURM_ARRAY_TASK_ID} 1.5 anistropic_rotated_pd
##
