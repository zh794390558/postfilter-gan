#!/bin/bash

usage(){
    echo "usage: $0 f0file lsffile wavfile"
    exit 1
}

if [[ $# != 3 ]];then
    usage
fi

# 40 is for 41 dim features
./straight_mceplsf  -f 22050 -lsf -order 40 -shift 5 -f0file $1 -syn $2  $3
