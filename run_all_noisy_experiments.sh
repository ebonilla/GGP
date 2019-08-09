#!/usr/bin/env bash
DATASETS='
cora
citeseer
'
RESULTS_DIR='results'
PERC_CORRUPTION='0.5 1.0 2.0 3.0 4.0 5.0 6.0'

for data in $DATASETS
do
    for perc_corruption in $PERC_CORRUPTION
    do
        for iter in $(seq 1 1 10)
        do
            echo $data-$perc_corruption-$iter
            ./run_noisy_experiment.sh $data $RESULTS_DIR $perc_corruption $iter
        done
    done
done

