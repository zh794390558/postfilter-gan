#!/bin/bash

set -x

if [[ $# > 1 ]]; then
    echo "usage: ./make.sh [test]"
elif [[ $# == 1 ]]; then
    if [[ $1 != 'test' ]]; then
        echo "usage: ./make.sh [test]"
        exit 1
    fi

    python gen_wav.py \
        --feature_dir data/lsf \
        --f0_dir /gfs/atlastts/StandFemale_22K/nature/postf0 \
        --wav_dir data/wav \
        --tool_dir tools

elif [[ $# < 1 ]]; then
    python gen_wav.py \
        --feature_dir /gfs/atlastts/StandFemale_22K/lsf \
        --f0_dir /gfs/atlastts/StandFemale_22K/nature/postf0 \
        --wav_dir /gfs/atlastts/StandFemale_22K/wav \
        --tool_dir tools
fi



