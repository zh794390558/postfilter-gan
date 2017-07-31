#!/bin/bash

set -x

if [[ $# > 1 ]]; then
    echo "usage: ./make.sh [test]"
elif [[ $# == 1 ]]; then
    if [[ $1 != 'test' ]]; then
        echo "usage: ./make.sh [test]"
        exit 1
    fi

    # test with 30 examples
    python make_tfrecords.py --examples 30 --save_path data --cfg cfg/postfilter.toml --test_size 0.1 --val_size 0.1 --force-gen
elif [[ $# < 1 ]]; then
    python make_tfrecords.py --save_path /gfs/atlastts/StandFemale_22K/tfrecords --cfg cfg/postfilter.toml --test_size 0.1 --val_size 0.1 --force-gen
fi



