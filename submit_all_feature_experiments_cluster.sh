#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=8GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --error=/scratch1/bon136/slurm-%A_%a.err
#SBATCH --output=/scratch1/bon136/slurm-%A_%a.out
#
# submit many jobs as an array of jobs
# use e.g. sbatch -a 0-999 submit_all_noisy_experiments_bracewell.sh
# where 0-999 are the range of the indices of the jobs
#

module load cuda/10.0.130
module load cudnn/v7.5.0-cuda92
source ~/graphGP-env/bin/activate


all_n_hidden='16 32'
all_n_neighbour='8 16 32'
all_seed_val='1 10 2 20 3 30 4 40 5 50'

all_dataset_name='cora citeseer'
all_model='ggp'


RESULTS_DIR=$SCRATCH1DIR # output for tensorboard

# Creates all the experiments settings into a single big array
c=0
for seed_val in ${all_seed_val}
do
    for dataset_name in ${all_dataset_name}
    do
        for model in ${all_model}
        do
            for n_hidden in ${all_n_hidden}
            do
                for n_neighbour in ${all_n_neighbour}
                do
                    ptr_dataset_name[c]=$dataset_name
                    ptr_model[c]=$model
                    ptr_n_hidden[c]=$n_hidden
                    ptr_n_neighbour[c]=$n_neighbour
                    ptr_seed_val[c]=$seed_val
                    let c=c+1
                done
            done
        done
    done
done

EPOCHS=10000
RUN_FILE='ssl_exp_noisy.py'
RANDOM_SEED=1
ADD_VAL=1


# Submit job
if [ ! -z "$SLURM_ARRAY_TASK_ID" ]
then
    i=$SLURM_ARRAY_TASK_ID
    DATASET_NAME=${ptr_dataset_name[$i]}
    MODEL=${ptr_model[$i]}
    N_HIDDEN=${ptr_n_hidden[$i]}
    N_NEIGHBOUR=${ptr_n_neighbour[$i]}
    SEED_VAL=${ptr_seed_val[$i]}

    BASENAME=$DATASET_NAME'_graph_supervised_nhidden'$N_HIDDEN'_neighbours'$N_NEIGHBOUR
    ADJ_MATRIX='Dataset/featured_based_datasets_compatible/'$DATASET_NAME'/'$BASENAME'.gpickle'

    str_options='--dataset='$DATASET_NAME' --epochs='$EPOCHS' --adjacency='$ADJ_MATRIX' --random-seed-np='$RANDOM_SEED' --random-seed-tf='$RANDOM_SEED' --fixed-split  --add-val --add-val-seed='$SEED_VAL

    name=$DATASET_NAME'/'$MODEL'/feature_based''/n_hidden'$N_HIDDEN'/n_neighbour'$N_NEIGHBOUR'/v'$SEED_VAL

    RESULTS_DIR=$RESULTS_DIR'/'$name
    mkdir -p $RESULTS_DIR

    str_options=$str_options' --results-dir='$RESULTS_DIR
    echo $str_options

    PYTHONPATH=. python $RUN_FILE $str_options

else
    echo "Error: Missing array index as SLURM_ARRAY_TASK_ID"
fi


