from tunning.constants import BENCHMARK_CONFIG
import subprocess


def prometheus_deploy():
    clean_playbook = BENCHMARK_CONFIG['software']['prometheus']['playbook_dir'] / 'prometheus_clean.yml'
    deploy_playbook = BENCHMARK_CONFIG['software']['prometheus']['playbook_dir'] / 'prometheus_deploy.yml'
    node_exporter = BENCHMARK_CONFIG['software']['prometheus']['node_exporter']
    node_exporter_0 = BENCHMARK_CONFIG['software']['prometheus']['node_exporter'][0]
    node_exporter_1 = BENCHMARK_CONFIG['software']['prometheus']['node_exporter'][1]
    node_exporter_2 = BENCHMARK_CONFIG['software']['prometheus']['node_exporter'][2]
    node_exporter_3 = BENCHMARK_CONFIG['software']['prometheus']['node_exporter'][3]
    cadvisor_0 = BENCHMARK_CONFIG['software']['prometheus']['cadvisor'][0]
    cadvisor_1 = BENCHMARK_CONFIG['software']['prometheus']['cadvisor'][1]
    cadvisor_2 = BENCHMARK_CONFIG['software']['prometheus']['cadvisor'][2]
    cadvisor_3 = BENCHMARK_CONFIG['software']['prometheus']['cadvisor'][3]
    deploy_host = BENCHMARK_CONFIG['software']['prometheus']['deploy_host']
    mongodb_exporter = BENCHMARK_CONFIG['software']['prometheus']['mongodb_exporter']
    nginx_exporter = BENCHMARK_CONFIG['software']['prometheus']['nginx_exporter']
    memcached_exporter = BENCHMARK_CONFIG['software']['prometheus']['memcached_exporter']
    redis_exporter = BENCHMARK_CONFIG['software']['prometheus']['redis_exporter']
    mysql_exporter = BENCHMARK_CONFIG['software']['prometheus']['mysql_exporter']
    nginx_ip = BENCHMARK_CONFIG['software']['nginx']['service_ip'] + ':' + BENCHMARK_CONFIG['software']['nginx']['port']
    memcached_ip = BENCHMARK_CONFIG['software']['memcached']['service_ip'] + ':' + BENCHMARK_CONFIG['software']['memcached']['port']
    mongodb_ip = BENCHMARK_CONFIG['software']['mongodb']['service_ip'] + ':' + BENCHMARK_CONFIG['software']['mongodb']['port']
    redis_ip = BENCHMARK_CONFIG['software']['redis']['service_ip'] + ':' + BENCHMARK_CONFIG['software']['redis']['port']
    mysql_ip = BENCHMARK_CONFIG['software']['mysql']['service_ip'] + ':' + BENCHMARK_CONFIG['software']['mysql']['port']
    mysql_host = BENCHMARK_CONFIG['software']['mysql']['service_ip']
    mysql_port = BENCHMARK_CONFIG['software']['mysql']['port']

    clean_cmd = [
        "ansible-playbook",
        clean_playbook,
    ]
    subprocess.run(
        clean_cmd, check=True
    )

    deploy_cmd = [
        "ansible-playbook",
        deploy_playbook,
        f"-e "
        f"node_exporter_0={node_exporter_0} "
        f"node_exporter_1={node_exporter_1} "
        f"node_exporter_2={node_exporter_2} "
        f"node_exporter_3={node_exporter_3} "
        f"cadvisor_0={cadvisor_0} "
        f"cadvisor_1={cadvisor_1} "
        f"cadvisor_2={cadvisor_2} "
        f"cadvisor_3={cadvisor_3} "
        f"deploy_host={deploy_host} "

        f"mongodb_exporter={mongodb_exporter} "
        f"nginx_exporter={nginx_exporter} "
        f"memcached_exporter={memcached_exporter} "
        f"redis_exporter={redis_exporter} "
        f"mysql_exporter={mysql_exporter} "
        f"nginx_ip={nginx_ip} "
        f"memcached_ip={memcached_ip} "
        f"mongodb_ip={mongodb_ip} "
        f"mysql_ip={mysql_ip} "
        f"mysql_host={mysql_host} "
        f"mysql_port={mysql_port} "
        f"redis_ip={redis_ip} ",
    ]

    subprocess.run(
        deploy_cmd, check=True
    )


if __name__ == "__main__":
    prometheus_deploy()
