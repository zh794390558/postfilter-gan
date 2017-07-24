#!/bin/bash

set -x

if [[ $# < 1 ]]; then
	echo 'usage: make [gpu|cpu]'
	exit 1
fi

if [[ $1 == cpu ]]; then
	FILE=docker/Dockerfile
elif [[ $1 == gpu ]]; then
	FILE=docker/Dockerfile.gpu
else
	echo "invalid pararm: $1"
	exit 1
fi


docker build --force-rm=false --no-cache=false -f $FILE -t harbor.ail.unisound.com/zhanghui/tensorflow-$1-1.2.1 docker 

if [ $? != 0 ]; then
	exit 1
fi

#docker rmi harbor.ail.unisound.com/zhanghui/tensorflow-$1-1.2.1:latest
docker push harbor.ail.unisound.com/zhanghui/tensorflow-$1-1.2.1 
