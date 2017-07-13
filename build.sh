#!/bin/bash

set -x 

docker build --force-rm=fasle --no-cache=false -f docker/Dockerfile -t harbor.ail.unisound.com/zhanghui/tensorflow-cpu-1.2.1 docker 

docker push harbor.ail.unisound.com/zhanghui/tensorflow-cpu-1.2.1 
