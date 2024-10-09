#!/bin/bash

#services=('compose-post-service' 'url-shorten-service' 'compose-post-redis' 'post-storage-mongodb'
#'user-mention-service' 'text-service' 'user-timeline-mongodb' 'nginx-web-server' 'user-timeline-service'
#'post-storage-service' 'media-service' 'unique-id-service' 'user-service' 'social-graph-service'
#'write-home-timeline-service' 'post-storage-memcached')

# key
services=('compose-post-service' 'url-shorten-service' 'compose-post-redis' 'post-storage-mongodb'
'user-mention-service' 'text-service' 'nginx-web-server' 'user-timeline-service'
'post-storage-service' 'media-service' 'unique-id-service' 'user-service' 'social-graph-service'
'write-home-timeline-service' 'post-storage-memcached'
'user-timeline-mongodb' 'social-graph-redis' 'home-timeline-redis' )

# all
#services=('compose-post-service' 'url-shorten-service' 'user-mention-service' 'text-service' 'nginx-web-server'
#'user-timeline-service' 'post-storage-service' 'media-service' 'unique-id-service' 'user-service' 'social-graph-service'
#'write-home-timeline-service'
# 'compose-post-redis' 'url-shorten-memcached' 'url-shorten-mongodb' 'user-timeline-mongodb' 'user-timeline-redis'
# 'post-storage-memcached' 'post-storage-mongodb' 'media-memcached' 'media-mongodb' 'user-memcached' 'user-mongodb'
# 'social-graph-mongodb' 'social-graph-redis'
#)

for wk in "${services[@]}"; do
  docker service rm "sn_$wk"
done