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
all_seed_random='853241 599831 27397 598716 277457 223802 777732 169406 481024 621272 454421 924398 155053 759545 621362 930787 397468 726836 449891 270299 614710 400516 375765 951214 208938 194723 618467 596379 946045 451935 127923 37496 945978 658017 20385 419256 631851 942747 204899 24597 211578 674092 388491 350927 202770 82784 373257 599330 540866 108095 957614 208404 401554 184939 41255 167059 748510 513383 208168 558758 866063 960600 401866 158903 489125 519866 694845 50843 900305 438892 184153 745074 470605 139258 148082 810130 905757 686390 367250 289005 474316 979516 709568 314351 26833 960475 850276 765347 880904 239623 102864 333886 694480 847794 563508 341065 329352 846424 450680 511197'

all_dataset_name='cora citeseer'
all_model='ggp'

split_size_cora='0.948 0.805'
split_size_citeseer='0.964 0.844'

RESULTS_DIR=$SCRATCH1DIR'/random_splits' # output for tensorboard

# Creates all the experiments settings into a single big array
c=0
for seed_random in ${all_seed_random}
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
                    ptr_seed_random[c]=$seed_random
                    let c=c+1
                done
            done
        done
    done
done

EPOCHS=10000
RUN_FILE='ssl_exp_noisy.py'
RANDOM_SEED_NP=1 # global random seed. Does not really matter
ADD_VAL=1
SEED_VAL=1

# Submit job
if [ ! -z "$SLURM_ARRAY_TASK_ID" ]
then
    i=$SLURM_ARRAY_TASK_ID
    DATASET_NAME=${ptr_dataset_name[$i]}
    MODEL=${ptr_model[$i]}
    N_HIDDEN=${ptr_n_hidden[$i]}
    N_NEIGHBOUR=${ptr_n_neighbour[$i]}
    RANDOM_SPLIT_SEED=${ptr_seed_random[$i]}

    BASENAME=$DATASET_NAME'_graph_supervised_nhidden'$N_HIDDEN'_neighbours'$N_NEIGHBOUR
    ADJ_MATRIX='Dataset/featured_based_datasets_compatible/'$DATASET_NAME'/'$BASENAME'.gpickle'


    if [ "$DATASET_NAME" = "cora" ]; then
        SPLIT_SIZE=$split_size_cora
     else
        SPLIT_SIZE=$split_size_citeseer
    fi

    str_options='--dataset='$DATASET_NAME' --epochs='$EPOCHS' --adjacency='$ADJ_MATRIX' --random-seed-np='$RANDOM_SEED_NP' --random-seed-tf='$RANDOM_SEED_NP' --random-split  --add-val --add-val-seed='$SEED_VAL' --random-split-seed='$RANDOM_SPLIT_SEED' --split-sizes='$SPLIT_SIZE

    name=$DATASET_NAME'/'$MODEL'/feature_based''/n_hidden'$N_HIDDEN'/n_neighbour'$N_NEIGHBOUR'/v'$RANDOM_SPLIT_SEED

    RESULTS_DIR=$RESULTS_DIR'/'$name
    mkdir -p $RESULTS_DIR

    str_options=$str_options' --results-dir='$RESULTS_DIR
    echo $str_options

    PYTHONPATH=. python $RUN_FILE $str_options

else
    echo "Error: Missing array index as SLURM_ARRAY_TASK_ID"
fi


