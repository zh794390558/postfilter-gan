#!/bin/bash

set -x

if [[ $# > 1 ]]; then
    echo "usage: ./evaluation.sh [test]"
    exit 1
elif [[ $# == 1 ]]; then
    if [[ $1 != 'test' ]]; then
        echo "uage: ./evaluation.sh [test]"
        exit 1
    fi

    python model/main.py \
        --batch_size 2\
        --epoch 1 \
        --seed 10 \
        --shuffle False \
        --network model.py \
        --networkDirectory . \
        --inference_db data/test\
        --inference_save data/lsf \
        --bitdepth 32 \
        --weights log/train/gan_2.ckpt \
        --summaries_dir log/summaries \
        --noserving_export #--log_device_placement --log_runtime_stats_per_step 2
        #--snapshortInterval 1 \
        #--save log/train \
        #--optimization adam \
        #--lr_base_rate 0.0001 \
        #--lr_polcy fixed \

    python gen_wav.py \
        --feature_dir data/lsf \
        --f0_dir /gfs/atlastts/StandFemale_22K/nature/postf0 \
        --wav_dir data/wav \
        --tool_dir tools \

elif [[ $# < 1 ]]; then
    python model/main.py  \
        --batch_size 64 \
        --epoch 200000 \
        --bitdepth 32 \
        --network model.py \
        --networkDirectory . \
        #--optimization adam \
        #--lr_base_rate 0.00001 \
        #--lr_polcy fixed \
        #--train_db /gfs/atlastts/StandFemale_22K/tfrecords/train \
        #--validation_db /gfs/atlastts/StandFemale_22K/tfrecords/val \
        --inference_db /gfs/atlastts/StandFemale_22K/tfrecords/test \
        --inference_save /gfs/atlastts/StandFemale_22K/eval_output/lsf \
        --save /gfs/atlastts/StandFemale_22K/log/train \
        --save_vars all \
        --summaries_dir /gfs/atlastts/StandFemale_22K/log/summaries \
        --seed 10 \
        --shuffle False \
        --snapshotPrefix gan \
        --snapshortInterval 1 \
        --noserving_export
        #--noshuffle \
        #--log_device_placement \
        #--log_runtime_stats_per_step 2 \
fi

