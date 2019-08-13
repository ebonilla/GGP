#!/usr/bin/env bash
# Run ggp on compatible feature-based (graph) datasets
DATASET_NAME=$1
RESULTS_DIR=$2 # output for tensorboard
N_HIDDEN=$3
N_NEIGHBOURS=$4
ITER=${5:-1}

RUN_FILE='ssl_exp_noisy.py'
BASENAME=$DATASET_NAME'_graph_supervised_nhidden'$N_HIDDEN'_neighbours'$N_NEIGHBOURS
ADJ_MATRIX='Dataset/featured_based_datasets_compatible/'$DATASET_NAME'/'$BASENAME'.gpickle'
name='GGP-feature-based-'$BASENAME'-seed-'$ITER

str_options=$DATASET_NAME' '$ITER' '$ADJ_MATRIX' '$RESULTS_DIR

echo $str_options
PYTHONPATH=. python $RUN_FILE $str_options 2>&1 | tee  $RESULTS_DIR'/'$name.log


