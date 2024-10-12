#!/bin/bash

entry_ip=$1
workload=$2
benchmark="hotel_reservation"


cd ~/tuning_microservice/deploy/hotel_reservation/scripts



 wrk -D exp -t 1 -c 10 -d 60 -R $workload -L -s ./recommendation.lua http://"$entry_ip":5000