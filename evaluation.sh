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

    # The src must split batch_size with gpu numbers,
    # so we use GPU0 to evalution, then the batch_size can be one without any error,
    # or Must satisfy `batch_size % n_gpu == 0`
    CUDA_VISIBLE_DEVICES=1 python model/main.py \
        --batch_size 1\
        --epoch 1 \
        --seed 10 \
        --bitdepth 32 \
        --noshuffle \
        --network model.py \
        --networkDirectory . \
        --weights log/train/gan_2.ckpt  \
        --inference_db data/test\
        --inference_save data/lsf \
        --summaries_dir log/summaries \
        --noserving_export #--log_device_placement --log_runtime_stats_per_step 2
        #--shuffle False \
        #--snapshortInterval 1 \
        #--save log/train \
        #--optimization adam \
        #--lr_base_rate 0.0001 \
        #--lr_polcy fixed \

elif [[ $# < 1 ]]; then
    CUDA_VISIBLE_DEVICES=1 python model/main.py  \
        --batch_size 1 \
        --epoch 1 \
        --seed 10 \
        --bitdepth 32 \
        --noshuffle \
        --network model.py \
        --networkDirectory . \
        --weights /gfs/atlastts/StandFemale_22K/log/train/gan_64.ckpt \
        --inference_db /gfs/atlastts/StandFemale_22K/tfrecords/test \
        --inference_save /gfs/atlastts/StandFemale_22K/lsf \
        --summaries_dir /gfs/atlastts/StandFemale_22K/log/summaries \
        --noserving_export
        #--snapshotPrefix gan \
        #--snapshortInterval 1 \
        #--shuffle True \
        #--optimization adam \
        #--lr_base_rate 0.00001 \
        #--lr_polcy fixed \
        #--train_db /gfs/atlastts/StandFemale_22K/tfrecords/train \
        #--validation_db /gfs/atlastts/StandFemale_22K/tfrecords/val \
        #--save /gfs/atlastts/StandFemale_22K/log/train \
        #--save_vars all \
        #--log_device_placement \
        #--log_runtime_stats_per_step 2 \
fi

