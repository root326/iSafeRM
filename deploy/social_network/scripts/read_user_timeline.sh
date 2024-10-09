#!/bin/bash

# 命令行参数
entry_ip=$1
benchmark="social_network"
# 负载规格
#workload=(15 20  25 30 35 40 35 30 25 20 15 20 25 30 35 40 35 30 25 20 15)
workload=(10 20 30 40 50 60 )
#workload=(100 110 120 130 140 150 160 170 180 190 200 210 220 230 250)
#workload=(50 )

# 定义一个倍数
multiplier=3

# 定义一个数组
# 遍历数组并乘以倍数
for ((i=0; i<${#workload[@]}; i++))
do
    # shellcheck disable=SC2004
    workload[$i]=$((workload[$i] * multiplier))
done

# shellcheck disable=SC2164
cd ~/tuning_microservice/DeathStarBench/socialNetwork

# 遍历数组
# shellcheck disable=SC2034
i=0
for wk in "${workload[@]}"; do
#  echo "workload: $wk"
  wrk -D exp -t 1 -c 10 -d 100 -R $wk -L -s ./wrk2/scripts/social-network/read-user-timeline.lua http://"$entry_ip":8080/wrk2-api/user-timeline/read > ~/tuning_microservice/.tmp/."$benchmark"_tmp/output/read_user_timeline_$i.log 2>&1
#  echo http://$deploy_ip:8080/wrk2-api/post/compose
  ((i++))
done