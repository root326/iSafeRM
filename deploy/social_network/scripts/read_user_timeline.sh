#!/bin/bash

entry_ip=$1
benchmark="social_network"
workload=(10 20 30 40 50 60 )

multiplier=3

for ((i=0; i<${#workload[@]}; i++))
do
    workload[$i]=$((workload[$i] * multiplier))
done

cd ~/tuning_microservice/DeathStarBench/socialNetwork

i=0
for wk in "${workload[@]}"; do
  wrk -D exp -t 1 -c 10 -d 100 -R $wk -L -s ./wrk2/scripts/social-network/read-user-timeline.lua http://"$entry_ip":8080/wrk2-api/user-timeline/read > ~/tuning_microservice/.tmp/."$benchmark"_tmp/output/read_user_timeline_$i.log 2>&1
  ((i++))
done