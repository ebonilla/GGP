#!/usr/bin/env bash

# Run gcn_bayesian with a corrupted dataset
DATASET_NAME=$1
RESULTS_DIR=$2 # output for tensorboard
PERC_CORRUPTION=$3
CORRUPTION_METHOD='adding'
ITER=${4:-1}

RUN_FILE='ssl_exp_noisy.py'
ADJ_MATRIX='Dataset/corrupted_datasets_compatible/'$DATASET_NAME'/'$CORRUPTION_METHOD'/'$DATASET_NAME'_'$CORRUPTION_METHOD'_'$PERC_CORRUPTION'_v'$ITER'.gpickle'
name='GGP-noisy-'$DATASET_NAME'-PERC_CORRUPTION'$PERC_CORRUPTION'-'$ITER


str_options=$DATASET_NAME' 0 '$ADJ_MATRIX' '$RESULTS_DIR

echo $str_options
PYTHONPATH=. python $RUN_FILE $str_options 2>&1 | tee  $RESULTS_DIR'/'$name.log


