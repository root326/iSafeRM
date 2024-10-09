graph [
  directed 1
  node [
    id 0
    label "media-service_UploadMedia_1"
  ]
  node [
    id 1
    label "compose-post-service_UploadMedia_2"
  ]
  node [
    id 2
    label "user-service_UploadUserWithUserId_3"
  ]
  node [
    id 3
    label "compose-post-service_UploadCreator_4"
  ]
  node [
    id 4
    label "post-storage-service_StorePost_5"
  ]
  node [
    id 5
    label "post-storage-service_MongoInsertPost_6"
  ]
  node [
    id 6
    label "nginx-web-server_/wrk2-api/post/compose_7"
  ]
  node [
    id 7
    label "nginx-web-server_/wrk2-api/post/compose_8"
  ]
  node [
    id 8
    label "nginx-web-server_ComposePost_9"
  ]
  node [
    id 9
    label "unique-id-service_UploadUniqueId_10"
  ]
  node [
    id 10
    label "text-service_UploadText_11"
  ]
  node [
    id 11
    label "compose-post-service_UploadUniqueId_12"
  ]
  node [
    id 12
    label "url-shorten-service_UploadUrls_13"
  ]
  node [
    id 13
    label "user-mention-service_UploadUserMentions_14"
  ]
  node [
    id 14
    label "compose-post-service_UploadText_15"
  ]
  node [
    id 15
    label "compose-post-service_RedisHashSet_16"
  ]
  node [
    id 16
    label "compose-post-service_RedisHashSet_17"
  ]
  node [
    id 17
    label "compose-post-service_RedisHashSet_18"
  ]
  node [
    id 18
    label "compose-post-service_UploadUrls_19"
  ]
  node [
    id 19
    label "compose-post-service_UploadUserMentions_20"
  ]
  node [
    id 20
    label "compose-post-service_RedisHashSet_21"
  ]
  node [
    id 21
    label "compose-post-service_RedisHashSet_22"
  ]
  node [
    id 22
    label "compose-post-service_RedisHashSet_23"
  ]
  node [
    id 23
    label "user-timeline-service_WriteUserTimeline_24"
  ]
  node [
    id 24
    label "user-timeline-service_MongoInsert_25"
  ]
  node [
    id 25
    label "user-timeline-service_RedisUpdate_26"
  ]
  edge [
    source 0
    target 8
  ]
  edge [
    source 1
    target 0
  ]
  edge [
    source 2
    target 8
  ]
  edge [
    source 3
    target 2
  ]
  edge [
    source 4
    target 14
  ]
  edge [
    source 5
    target 4
  ]
  edge [
    source 7
    target 6
  ]
  edge [
    source 8
    target 7
  ]
  edge [
    source 9
    target 8
  ]
  edge [
    source 10
    target 8
  ]
  edge [
    source 11
    target 9
  ]
  edge [
    source 12
    target 10
  ]
  edge [
    source 13
    target 10
  ]
  edge [
    source 14
    target 10
  ]
  edge [
    source 15
    target 1
  ]
  edge [
    source 16
    target 3
  ]
  edge [
    source 17
    target 11
  ]
  edge [
    source 18
    target 12
  ]
  edge [
    source 19
    target 13
  ]
  edge [
    source 20
    target 18
  ]
  edge [
    source 21
    target 19
  ]
  edge [
    source 22
    target 14
  ]
  edge [
    source 23
    target 14
  ]
  edge [
    source 24
    target 23
  ]
  edge [
    source 25
    target 23
  ]
]
