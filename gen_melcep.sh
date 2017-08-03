#!/bin/bash

set -x

if [[ $# > 1 ]]; then
    echo "usage: ./make.sh [test]"
elif [[ $# == 1 ]]; then
    if [[ $1 != 'test' ]]; then
        echo "usage: ./make.sh [test]"
        exit 1
    fi

    # nature
    python gen_mcep.py \
        --feature_dir data/nature/mcep \
        --feature_suffix mcep \
        --f0_dir /gfs/atlastts/StandFemale_22K/nature/postf0 \
        --wav_dir data/wav \
        --tool_dir tools

elif [[ $# < 1 ]]; then
    # nature
    python gen_mcep.py \
        --feature_dir /gfs/atlastts/StandFemale_22K/nature/melcep \
        --feature_suffix mcep \
        --f0_dir /gfs/atlastts/StandFemale_22K/nature/postf0 \
        --wav_dir /gfs/atlastts/StandFemale_22K/nature/wav_nature \
        --tool_dir tools
fi



