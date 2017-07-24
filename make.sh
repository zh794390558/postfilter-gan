#!/bin/bash

set -x

python make_tfrecords.py --save_path /gfs/atlastts/StandFemale_22K/tfrecords  --out_file postfilter.tfrecords --force-gen

#python make_tfrecords.py --examples 30 --save_path data --out_file postfilter.tfrecords --force-gen
