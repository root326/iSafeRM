#!/bin/bash

# 命令行参数
entry_ip=$1
workload=$2
benchmark="hotel_reservation"
# 负载规格

# shellcheck disable=SC2164
cd ~/tuning_microservice/deploy/hotel_reservation/scripts



 wrk -D exp -t 1 -c 10 -d 60 -R $workload -L -s ./search.lua http://"$entry_ip":5000  > ~/tuning_microservice/.tmp/."$benchmark"_tmp/output/search.log 2>&1
