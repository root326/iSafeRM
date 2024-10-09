#!/bin/bash

#key=$1

# search
services=('frontend' 'search' 'geo' 'rate' 'reservation' 'profile'
'mongodb-profile' 'memcached-profile' 'mongodb-rate' 'memcached-rate' 'mongodb-geo' 'mongodb-reservation' 'memcached-reserve'
)

#services=('frontend' 'search' 'geo' 'rate' 'reservation' 'profile'
#'memcached-profile' 'mongodb-geo' 'mongodb-reservation' 'memcached-reserve' 'mongodb-rate' 'memcached-rate'
#)

#case $key in
#    "key")
#        # 匹配 pattern1 时执行的命令
#        services=('frontend' 'search' 'geo' 'rate' 'reservation' 'profile'
#'memcached-profile' 'mongodb-geo' 'mongodb-reservation' 'memcached-reserve'
#)
#        ;;
#    "bottleneck")
#        # 匹配 pattern2 时执行的命令
#        services=('frontend' 'search' 'geo' 'rate' 'reservation' 'profile'
#'memcached-profile' 'mongodb-geo' 'mongodb-rate' 'memcached-rate'
#)
#
#        ;;
#    *)
#        # 默认情况下执行的命令
#        services=('frontend' 'search' 'geo' 'rate' 'reservation' 'profile'
#'memcached-profile' 'mongodb-geo' 'mongodb-reservation' 'memcached-reserve' 'mongodb-rate' 'memcached-rate'
#)
#        ;;
#esac


# recommendation
#services=('frontend' 'recommendation' 'profile'
#'mongodb-recommendation' 'mongodb-profile' 'memcached-profile')

for wk in "${services[@]}"; do
  docker service rm "hr_$wk"
done