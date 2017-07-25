#!/bin/bash

set -x 

#./atlasctl create  -n demo -i bootstrapper:5000/zhanghui/tensorflow-cpu  \
#		-g 0 -v "nfs,10.10.10.251:/volume1/gfs,/gfs" \
#		-a "echo 'hello';sleep 20" -v 'hostpath,/tmp, /tmp'

usage(){
	echo "usage: ./create.sh [gpu|cpu]"
	exit 1
}

if [[ $# != 1 ]]; then
	usage
elif [[ $# == 1 ]]; then
	if [[ $1 == 'gpu' ]]; then
		kubectl create -f k8s/gpu.yaml
		kubectl create -f k8s/service.yaml
	elif [[ $1 == 'cpu' ]]; then
		kubectl create -f k8s/cpu.yaml
		kubectl create -f k8s/service.yaml
	fi	
fi

