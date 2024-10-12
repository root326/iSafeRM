#!/bin/bash

entry_ip=$1
workload=$2
benchmark="social_network"
cd ~/tuning_microservice/deploy/social_network/scripts

wrk -D exp -t 1 -c 10 -d 60 -R $workload -L -s ./compose-post.lua http://"$entry_ip":8082/wrk2-api/post/compose  > ~/tuning_microservice/.tmp/."$benchmark"_tmp/output/compose_post.log 2>&1
