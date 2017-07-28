#!/bin/bash

set -x

if [[ $# > 1 ]]; then
    echo "usage: ./run.sh [test]"
    exit 1
elif [[ $# == 1 ]]; then
    if [[ $1 != 'test' ]]; then
        echo "uage: ./run.sh [test]"
        exit 1
    fi

    python model/main.py  \
        --batch_size 6 \
        --epoch 2 \
        --bitdepth 32 \
        --network model.py \
        --networkDirectory . \
        --optimization adam \
        --lr_base_rate 0.0001 \
        --lr_polcy fixed \
        --train_db data/train \
        --validation_db data/val \
        --save log/train \
        --save_vars all \
        --summaries_dir log/summaries \
        --seed 10 \
        --shuffle True\
        --snapshotPrefix gan \
        --snapshortInterval 1 \
        --noserving_export
        #--noshuffle \
        #--log_device_placement \
        #--log_runtime_stats_per_step 2 \

elif [[ $# < 1 ]]; then
    python model/main.py  \
        --batch_size 64 \
        --epoch 200000 \
        --bitdepth 32 \
        --network model.py \
        --networkDirectory . \
        --optimization adam \
        --lr_base_rate 0.00001 \
        --lr_polcy fixed \
        --train_db /gfs/atlastts/StandFemale_22K/tfrecords/train \
        --validation_db /gfs/atlastts/StandFemale_22K/tfrecords/val \
        --save /gfs/atlastts/StandFemale_22K/log/train \
        --save_vars all \
        --summaries_dir /gfs/atlastts/StandFemale_22K/log/summaries \
        --seed 10 \
        --shuffle True\
        --snapshotPrefix gan \
        --snapshortInterval 2 \
        --noserving_export
        #--noshuffle \
        #--log_device_placement \
        #--log_runtime_stats_per_step 2 \
fi

