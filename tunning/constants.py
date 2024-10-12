from pathlib import Path
import time

_PROJECT_ROOT_DIR = Path(__file__).parent.parent.resolve()
TMP_DIR = _PROJECT_ROOT_DIR / '.tmp'
TMP_OUTPUT_DIR = _PROJECT_ROOT_DIR / '.tmp' / 'output'
RESULT_DIR = _PROJECT_ROOT_DIR / 'results'
CONFIGS_DIR = _PROJECT_ROOT_DIR / 'configs'

RESULT_FILE_COLUMN = ['task_id', 'rep_run_time', 'latency', 'resource', 'latency_list', 'observation', 'reward', 'cost',
                      'curr_workload_list']

YCSB_BENCHMARK = ['redis', 'mongodb', 'memcached', 'mysql']
WRK_BENCHMARK = ['nginx']
SOFTWARE_DEPLOY_DIR = _PROJECT_ROOT_DIR / 'deploy' / 'software'

HOST_CONFIG = {
    'host1': ['192.168.1.237', 'aituningnode1'],
    'host2': ['192.168.1.238', 'aituningnode2'],
    'host3': ['192.168.1.239', 'aituningnode3'],
    'host4': ['192.168.1.183', 'node3']
}

GLOBAL_CONFIG = {
    'python_interpreter': '/home/XXXX-1/miniconda3/envs/mytunning/bin/python',
    'prometheus_host': f"http://{HOST_CONFIG['host1'][0]}:9090",
    'jaeger_host': f"http://{HOST_CONFIG['host1'][0]}:16686",
    'results_dir': _PROJECT_ROOT_DIR / 'output/results',

    'configs_dir': _PROJECT_ROOT_DIR / 'configs',
    'tmp_dir': _PROJECT_ROOT_DIR / '.tmp',
    'deploy_dir': _PROJECT_ROOT_DIR / 'deploy',
    'ycsb_benchmark': ['redis', 'mongodb', 'memcached', 'mysql'],
    'wrk_benchmark': ['nginx'],
    'result_file_column': ['task', 'latency', 'resource', 'observation', 'reward',
                           'cost', 'workload','conf'],
    'key_config':  _PROJECT_ROOT_DIR / 'configs'/'key_config.csv',
    'envs_id': [
        'wk10-sn_key_services_all_parameters',
        'wk10-sn_key_services_key_parameters',
        'wk10-sn_key_services_resource',
        'wk-sn_key_services_key_parameters',
        "wk20-sn_5causal_3parameters"
    ],
    "project_root_dir": _PROJECT_ROOT_DIR,
    "project_name": "iSafeRM"
}

BENCHMARK_CONFIG = {
    'software': {
        'prometheus': {
            'node_exporter': [f"{HOST_CONFIG['host1'][0]}:9100", f"{HOST_CONFIG['host2'][0]}:9100",
                              f"{HOST_CONFIG['host3'][0]}:9100", f"{HOST_CONFIG['host4'][0]}:9100"],
            'cadvisor': [f"{HOST_CONFIG['host1'][0]}:9105", f"{HOST_CONFIG['host2'][0]}:9105",
                         f"{HOST_CONFIG['host3'][0]}:9105", f"{HOST_CONFIG['host4'][0]}:9105"],
            'mongodb_exporter': f"{HOST_CONFIG['host1'][0]}:9216",
            'nginx_exporter': f"{HOST_CONFIG['host1'][0]}:9113",
            'memcached_exporter': f"{HOST_CONFIG['host1'][0]}:9150",
            'redis_exporter': f"{HOST_CONFIG['host1'][0]}:9121",
            'mysql_exporter': f"{HOST_CONFIG['host1'][0]}:9104",
            'playbook_dir': SOFTWARE_DEPLOY_DIR / 'prometheus' / 'ansible',
            'deploy_host': HOST_CONFIG['host1'][1]
        },
        'nginx': {
            'playbook_dir': '',
            'deploy_host': HOST_CONFIG['host1'][1],
            'service_ip': HOST_CONFIG['host1'][0],
            'exporter_url': f"http://{HOST_CONFIG['host1'][0]}:9113/metrics",
            'interval': 90,
            'port': '80',
            'key_parameters': [],
        },
        'memcached': {
            'playbook_dir': '',
            'deploy_host': HOST_CONFIG['host2'][1],
            'service_ip': HOST_CONFIG['host2'][0],
            'exporter_url': f"http://{HOST_CONFIG['host1'][0]}:9150/metrics",
            'interval': 80,
            'port': '11211',
            'key_parameters': [],

        },
        'mongodb': {
            'playbook_dir': '',
            'deploy_host': HOST_CONFIG['host3'][1],
            'service_ip': HOST_CONFIG['host3'][0],
            'exporter_url': f"http://{HOST_CONFIG['host1'][0]}:9216/metrics",
            'interval': 120,
            'port': '27017',
            'key_parameters': [],

        },
        'redis': {
            'playbook_dir': '',
            'deploy_host': HOST_CONFIG['host2'][1],
            'service_ip': HOST_CONFIG['host2'][0],
            'exporter_url': f"http://{HOST_CONFIG['host1'][0]}:9121/metrics",
            'interval': 150,
            'port': '6379',
            'key_parameters': [],

        },
        'mysql': {
            'playbook_dir': '',
            'deploy_host': HOST_CONFIG['host1'][1],
            'service_ip': HOST_CONFIG['host1'][0],
            'exporter_url': f"http://{HOST_CONFIG['host1'][0]}:9104/metrics",
            'interval': 250,
            'port': '3306',
            'key_parameters': [],

        },
    },
    'microservices': {
        'social_network': {
            'entry_point': 'nginx-web-server',
            'jaegerHost': f"http://{HOST_CONFIG['host1'][0]}:16688",
            'entry_ip': f"{HOST_CONFIG['host1'][0]}",
            'operations': {
                "ComposePost": "/wrk2-api/post/compose",
            },
            'operation': "/wrk2-api/post/compose",
            'output': GLOBAL_CONFIG['tmp_dir'] / '.social_network_tmp' / 'output',
            'workload': [20],
            'configs_dir': GLOBAL_CONFIG['configs_dir'] / 'social_network',
            'playbook_dir': GLOBAL_CONFIG['deploy_dir'] / 'social_network' / 'ansible',
            'workload_sh': GLOBAL_CONFIG['deploy_dir'] / 'social_network' / 'scripts' / 'composepost.sh',
            'obs_metrics': 3 + 3 + 1 + 1 + 1 + 1 + 1,
            'slo': 200,
            'workload_duration': 4,
            'experiments_columns': ['ComposePost', 'HomeTimeline', 'UserTimeline', 'workload1', 'workload2',
                                    'workload3'],
            'experiments_results': GLOBAL_CONFIG['results_dir'] / 'experiments' / 'social_network',
            'results': GLOBAL_CONFIG['results_dir'] / 'social_network',
            'bench_log': 'compose_post.log',
            'p_latency': '950000',
            'default_conf': '/home/XXXX-1/tuning_microservice/deploy/social_network/ansible/default_conf.yml',
            'span_to_ms': {
                'nginx-web-server': 'nginx-web-server_/wrk2-api/post/compose',
                'compose-post-service': 'nginx-web-server_ComposePost',
                'media-service': 'media-service_UploadMedia',
                'media-memcached': 'media-service_UploadMedia',
                'unique-id-service': 'unique-id-service_UploadUniqueId',
                'user-service': 'user-service_UploadUserWithUserId',
                'user-memcached': 'user-service_UploadUserWithUserId',
                'text-service': 'text-service_UploadText',
                'url-shorten-service': 'url-shorten-service_UploadUrls',
                'url-shorten-memcached': 'url-shorten-service_UploadUrls',
                'user-timeline-service': 'user-timeline-service_WriteUserTimeline',
                'user-timeline-mongodb': 'user-timeline-service_MongoInsert',
                'user-timeline-redis': 'user-timeline-service_RedisUpdate',
                'compose-post-redis': 'compose-post-service_RedisHashSet',
                'user-mention-service': 'user-mention-service_UploadUserMentions',
                'post-storage-service': 'post-storage-service_StorePost',
                'post-storage-memcached': 'post-storage-service_StorePost',
                'write-home-timeline-service': 'write-home-timeline-service_FanoutHomeTimelines',
                'post-storage-mongodb': 'post-storage-service_MongoInsertPost',
                'social-graph-service': 'social-graph-service_GetFollowers',
                'write-home-timeline-redis': 'write-home-timeline-service_RedisUpdate',
                'social-graph-redis': 'social-graph-service_RedisGet',
                'social-graph-mongodb': 'social-graph-service_MongoFindUser',
            }
        },
    }
}
