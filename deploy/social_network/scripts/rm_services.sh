#!/bin/bash

services=('compose-post-service' 'url-shorten-service' 'compose-post-redis' 'post-storage-mongodb'
'user-mention-service' 'text-service' 'nginx-web-server' 'user-timeline-service'
'post-storage-service' 'media-service' 'unique-id-service' 'user-service' 'social-graph-service'
'write-home-timeline-service' 'post-storage-memcached'
'user-timeline-mongodb' 'social-graph-redis' 'home-timeline-redis' )

for wk in "${services[@]}"; do
  docker service rm "sn_$wk"
done