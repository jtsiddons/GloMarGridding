#!/bin/bash
#SBATCH -p short-serial
#SBATCH --array=[2-18]
#SBATCH --job-name=esa_sst_ten
#SBATCH --output=/work/scratch-pw2/schan016/NOC-hostace/temp/run_10deg_%j-%A_%a.o
#SBATCH --error=/work/scratch-pw2/schan016/NOC-hostace/temp/run_10deg_%j-%A_%a.e
#SBATCH --time=6:00:00
#SBATCH --mem=20000
##
module load jasr
Rscript prep_vg_data.R $SLURM_ARRAY_TASK_ID

