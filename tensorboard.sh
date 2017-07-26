#!/bin/bash

set -x

if [[ $# > 1 ]]; then
    echo "usage: ./tensorboard.sh [test]"
elif [[ $# == 1 ]]; then
    if [[ $1 != 'test' ]]; then
        echo "usage: ./tensorboard.sh [test]"
        exit 1
    fi

    tensorboard --logdir log &
elif [[ $# < 1 ]]; then
    tensorboard --logdir /gfs/atlastts/StandFemale_22K/log/ &
fi



