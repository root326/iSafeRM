#!/bin/bash

# 命令行参数
entry_ip=$1
workload=$2
benchmark="social_network"
# 负载规格
#workload=(15 20  25 30 35 40 35 30 25 20 15 20 25 30 35 40 35 30 25 20 15)
#workload=(10 20 30 40 50 60 )
#workload=(100 110 120 130 140 150 160 170 180 190 200 210 220 230 250)
#workload=(10 10)
# shellcheck disable=SC2164
cd ~/tuning_microservice/deploy/social_network/scripts

# 遍历数组
# shellcheck disable=SC2034
#i=0
#for wk in "${workload[@]}"; do
##  echo "workload: $wk"
#  wrk -D exp -t 1 -c 10 -d 200 -R $wk -L -s ./wrk2/scripts/social-network/compose-post.lua http://"$entry_ip":8080/wrk2-api/post/compose  > ~/tuning_microservice/.tmp/."$benchmark"_tmp/output/compose_post_$i.log 2>&1
##  echo http://$deploy_ip:8080/wrk2-api/post/compose
#  ((i++))
#done

 wrk -D exp -t 1 -c 10 -d 60 -R $workload -L -s ./compose-post.lua http://"$entry_ip":8082/wrk2-api/post/compose  > ~/tuning_microservice/.tmp/."$benchmark"_tmp/output/compose_post.log 2>&1
