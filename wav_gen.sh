#!/bin/bash

python gen_wav.py \
    --feature_dir data/lsf \
    --f0_dir /gfs/atlastts/StandFemale_22K/nature/postf0 \
    --wav_dir data/wav \
    --tool_dir tools
