#!/bin/bash

set -x 

./atlasctl create  -n demo -i bootstrapper:5000/zhanghui/tensorflow-cpu  \
		-g 0 -v "nfs,10.10.10.251:/volume1/gfs,/gfs" \
		-a "echo 'hello';sleep 20" -v 'hostpath,/tmp, /tmp'
