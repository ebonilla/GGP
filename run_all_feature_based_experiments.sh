#!/usr/bin/env bash
DATASETS='
cora
citeseer
'
RESULTS_DIR='results'
N_HIDDEN='16 32'
N_NEIGHBOURS='8 16 32'

for data in $DATASETS
do
    for n_hidden in $N_HIDDEN
    do
        for n_neighbours in $N_NEIGHBOURS
        do
            for iter in $(seq 1 1 10)
            do
                #str_cmd='./run_feature_based_experiment.sh '$data' '$RESULTS_DIR' '$n_hidden' '$n_neighbours' '$iter
                ./run_feature_based_experiment.sh $data $RESULTS_DIR $n_hidden $n_neighbours $iter
            done
         done
    done
done

