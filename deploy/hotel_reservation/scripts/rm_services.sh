#!/bin/bash

services=('frontend' 'search' 'geo' 'rate' 'reservation' 'profile'
'mongodb-profile' 'memcached-profile' 'mongodb-rate' 'memcached-rate' 'mongodb-geo' 'mongodb-reservation' 'memcached-reserve'
)



for wk in "${services[@]}"; do
  docker service rm "hr_$wk"
done