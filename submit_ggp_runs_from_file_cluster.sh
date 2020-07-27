#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=8GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --error=/scratch1/bon136/slurm-%A_%a.err
#SBATCH --output=/scratch1/bon136/slurm-%A_%a.out
#
# Submit no graph experiments to cluster (a feature-based prior is constructed )
#
# submit many jobs as an array of jobs
# use e.g. sbatch -a 0-999 submit_all_noisy_experiments_bracewell.sh
# where 0-999 are the range of the indices of the jobs
#
# GPUs not requested above since the environment was installed with tensorflow cpu

module load cuda/10.0.130
module load cudnn/v7.5.0-cuda92
source ~/graphGP-env/bin/activate

IFS=$'\n' read -d '' -r -a lines < ${1}

# Submit job
if [ ! -z "$SLURM_ARRAY_TASK_ID" ]
then
    i=$SLURM_ARRAY_TASK_ID
    echo ${lines[i]}
    eval "${lines[i]}"
else
    echo "Error: Missing array index as SLURM_ARRAY_TASK_ID"
fi



