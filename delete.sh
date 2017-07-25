#!/bin/bash 

set -x

usage(){
	echo "usage: ./delete.sh [gpu|cpu]"
	exit 1
}

if [[ $# != 1 ]]; then
	usage
elif [[ $# == 1 ]]; then
	if [[ $1 == 'gpu' ]]; then
		kubectl delete -f k8s/gpu.yaml
		kubectl delete -f k8s/service.yaml
	elif [[ $1 == 'cpu' ]]; then
		kubectl delete -f k8s/cpu.yaml
		kubectl delete -f k8s/service.yaml
	fi	
fi

