#!/bin/bash

set -x 

docker build -f docker/Dockerfile -t bootstrapper:5000/zhanghui/tensorflow-cpu docker 

docker push bootstrapper:5000/zhanghui/tensorflow-cpu
