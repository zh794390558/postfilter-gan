#!/bin/bash

usage(){
    echo "usage: $0 f0file wavfile melcepfile"
    exit 1
}

if [[ $# != 3 ]];then
    usage
fi

# 40 is for 41 dims feature
./straight_mceplsf -f 22050 -shift 5 -mcep -pow -order 40 -f0file $1 -ana $2 $3
