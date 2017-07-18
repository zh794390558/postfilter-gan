#!/bin/bash

set -x 

docker build --force-rm=false --no-cache=false -f docker/Dockerfile -t harbor.ail.unisound.com/zhanghui/tensorflow-cpu-1.2.1 docker 

if [ $? != 0 ]; then
	exit 1
fi

docker rmi harbor.ail.unisound.com/zhanghui/tensorflow-cpu-1.2.1:latest
docker push harbor.ail.unisound.com/zhanghui/tensorflow-cpu-1.2.1 
