#!/bin/bash

set -x

python model/main.py --batch_size 5 --epoch 2 --network model.py --networkDirectory . \
	--optimization adam --save log --seed 10 --noshuffle --snapshortInterval 1 \
	--train_db data --weights log --bitdepth 32 \
	--lr_base_rate 0.01 --lr_polcy fixed --save_vars all --summaries_dir log \
	--noserving_export --log_device_placement --log_runtime_stats_per_step 2 
