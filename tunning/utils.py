import os
from pathlib import Path
from typing import Any, Dict
import json
import yaml
import re
import requests
import pandas as pd
from tunning.constants import GLOBAL_CONFIG
import logging
import time
import numpy as np
import random
import pickle
from pathlib import Path
current_file_path = os.path.abspath(__file__)
tuning_folder = os.path.dirname(os.path.dirname(current_file_path))

my_logger = logging.Logger("file_logger")
# current_path = os.getcwd()
# parent_path = o
log_file = Path(f"{tuning_folder}/output/log/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.log")
# log_file = os.path.join(current_path, log_file)
if Path(log_file).exists():
    os.remove(log_file)
my_handler = logging.FileHandler(log_file, encoding='utf-8')

my_handler.setLevel(logging.INFO)
my_format = logging.Formatter("log time: %(asctime)s level: %(levelname)s message: %(message)s line: %(lineno)d")

# 把handler添加到对应的logger中去。
my_handler.setFormatter(my_format)
my_logger.addHandler(my_handler)

you_handler = logging.StreamHandler()

you_handler.setLevel(logging.DEBUG)
you_format = logging.Formatter("log time: %(asctime)s level: %(levelname)s message: %(message)s line: %(lineno)d")

you_handler.setFormatter(you_format)
my_logger.addHandler(you_handler)

you_handler.setFormatter(you_format)
my_logger.addHandler(you_handler)


def parse_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f'config file {config_path} not found')
    return yaml.safe_load(config_path.read_text())


def rm_path(path: Path) -> None:
    for p in path.iterdir():
        if p.is_dir():
            rm_path(p)
        else:
            p.unlink()
    path.rmdir()


def format_time(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def result2json(result_dir, iteration):
    res_json = {}
    conf_dict = {}
    task_dict = {}
    for i in range(iteration + 1):
        file_name = result_dir / f"{i}_conf.yml"
        with open(file_name, "r") as file:
            # content = file.read()
            # content = content.replace('\n', '')
            yml_obj = yaml.safe_load(file)
            # json_str = json.dumps(yml_obj)
            # content = file.read()
            # print(yml_obj)
            conf_dict[f"{i}_conf.yml"] = yml_obj
    res_json["conf_result"] = conf_dict
    res_sum_file = result_dir / "_results_summary.txt"
    latency = re.compile(r"-latency:\s+(\d+\.+\d)")
    resource = re.compile(r"-resource:\s+(\d+\.+\d)")
    task_id = re.compile(r"task_\d+")
    with open(res_sum_file, "r") as sum_file:
        for line in sum_file.readlines():
            match_latency = latency.search(line)
            match_resource = resource.search(line)
            match_task = task_id.search(line)
            # content = res_sum_file.read_text()
            # print(match_latency.group(1))
            # print(match_resource.group(1))
            # print(match_task.group(0))
            task = {"latency": float(match_latency.group(1)), "resource": float(match_resource.group(1))}
            task_dict[match_task.group(0)] = task
    res_json["task_result"] = task_dict
    return res_json


def fetch_prometheus(
        host,
        prometheus_query,
        query_type,
        step=None,
        start_time=None,
        end_time=None,
        time=None,
):
    request_data = {
        "query": prometheus_query,
    }
    # request_data["query"] = "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode='idle'}[5m])) * 100) "
    # query = f'instance:node_cpu_utilisation:rate1m{{instance=~"{".*|".join(nodes)}.*"}}'

    if query_type == "range":
        request_data["step"] = step
        request_data["start"] = start_time
        request_data["end"] = end_time
    elif query_type == "point":
        request_data["time"] = time
    # print(time)
    url_suffix = {"range": "query_range", "point": "query"}[query_type]
    res = requests.get(f"{host}/api/v1/{url_suffix}", params=request_data)
    return res


def get_prometheus_response_value(res):
    res_list = []
    for i in range(len(res['data']['result'])):
        # print(res['data']['result'][i])
        res_list.append(round(float(res['data']['result'][i]['value'][1]), 2))
    # print(res_list)
    return res_list


def get_node_cpu_usage(host: str = GLOBAL_CONFIG['prometheus_host'],
                       time_interval: int = 5,
                       job_name: str = "prometheus"):
    query = f"(1 - avg(irate(node_cpu_seconds_total{{job='{job_name}',mode='idle'}}[{time_interval}m]))by(instance))*100"
    # print(query)
    return get_prometheus_response_value(fetch_prometheus(host, query, "point").json())


def get_node_memory_usage(host: str = GLOBAL_CONFIG['prometheus_host'],
                          time_interval: int = 5,
                          job_name: str = "prometheus"):
    query = f"100 * (1 - ((avg_over_time(node_memory_MemFree_bytes{{job='{job_name}'}}[{time_interval}m]) " \
            f"+ avg_over_time(node_memory_Cached_bytes{{job='{job_name}'}}[{time_interval}m]) " \
            f"+ avg_over_time(node_memory_Buffers_bytes{{job='{job_name}'}}[{time_interval}m])) " \
            f"/ avg_over_time(node_memory_MemTotal_bytes{{job='{job_name}'}}[{time_interval}m])))"
    # print(fetch_prometheus(host, query, "point").json())
    return get_prometheus_response_value(fetch_prometheus(host, query, "point").json())


def get_prometheus_response_container_value(res):
    # print(res)
    res_list = []
    for i in range(len(res['data']['result'])):
        # print(res['data']['result'][i])
        tmp = {}
        tmp['container_name'] = res['data']['result'][i]['metric']['name']
        tmp['container_value'] = round(float(res['data']['result'][i]['values'][0][1]), 2)
        res_list.append(tmp)
    # print(res_list)
    value_list = []
    for v in res_list:
        value_list.append(v['container_value'])
    return round(sum(value_list) / len(value_list), 2)
    # return res_list


def get_svc_count(svcs):
    dic = {}
    for svc in svcs:
        dic[svc] = get_container_svc_count(start_time=time.time() - 10000, end_time=time.time(), service_name=svc)
    return dic


def get_container_svc_count(host: str = GLOBAL_CONFIG['prometheus_host'],
                            start_time=None,
                            end_time=None,
                            service_name: str = ''):
    query = f"count(container_memory_usage_bytes{{container_label_com_docker_swarm_service_name='{service_name}'}})"
    # print(query)
    res_cpu = []
    # res_cpu = fetch_prometheus(host, query, "point").json()
    # print(res_cpu)
    count = 1
    while count > 0:
        res_cpu = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # res_cpu = fetch_prometheus(host, query, "point").json()
        # print(res_cpu)
        if len(res_cpu['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_cpu) + 1


def get_container_cpu_limit(host: str = GLOBAL_CONFIG['prometheus_host'],
                            start_time=None,
                            end_time=None,
                            service_name: str = ''):
    query = f"container_spec_cpu_quota{{container_label_com_docker_swarm_service_name='{service_name}'}} /100000 "
    # print(query)
    res_cpu = []
    # res_cpu = fetch_prometheus(host, query, "point").json()
    # print(res_cpu)
    count = 1
    while count > 0:
        res_cpu = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # res_cpu = fetch_prometheus(host, query, "point").json()
        # print(res_cpu)
        if len(res_cpu['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_cpu)


def get_container_cpu_usage(host: str = GLOBAL_CONFIG['prometheus_host'],
                            start_time=None,
                            end_time=None,
                            service_name: str = ''):
    query = f"sum(irate(container_cpu_usage_seconds_total{{container_label_com_docker_swarm_service_name='{service_name}'}}[3m]))without(cpu)*100"
    # print(query)
    res_cpu = []
    # res_cpu = fetch_prometheus(host, query, "point").json()
    # print(res_cpu)
    count = 1
    while count > 0:
        res_cpu = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # res_cpu = fetch_prometheus(host, query, "point").json()
        # print(res_cpu)
        if len(res_cpu['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_cpu)


def get_container_memory_usage(host: str = GLOBAL_CONFIG['prometheus_host'],
                               start_time=None,
                               end_time=None,
                               service_name: str = ''):
    # query = f"container_memory_usage_bytes/container_spec_memory_limit_bytes{{container_label_com_docker_swarm_service_name='{service_name}'}}*100"
    # print(query)
    query = f"avg_over_time(container_memory_usage_bytes{{container_label_com_docker_swarm_service_name='{service_name}'}}[3m]) / (container_spec_memory_limit_bytes{{container_label_com_docker_swarm_service_name='{service_name}'}}) * 100"
    res_memory = []
    count = 1
    while count > 0:
        res_memory = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # print(res_memory)
        if len(res_memory['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_memory)


def get_container_memory_limit(host: str = GLOBAL_CONFIG['prometheus_host'],
                               start_time=None,
                               end_time=None,
                               service_name: str = ''):
    # query = f"container_memory_usage_bytes/container_spec_memory_limit_bytes{{container_label_com_docker_swarm_service_name='{service_name}'}}*100"
    # print(query)
    query = f"container_spec_memory_limit_bytes{{container_label_com_docker_swarm_service_name='{service_name}'}} / (1024*1024)"
    res_memory = []
    count = 1
    while count > 0:
        res_memory = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # print(res_memory)
        if len(res_memory['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_memory)


def get_container_fs_usage(host: str = GLOBAL_CONFIG['prometheus_host'],
                           start_time=None,
                           end_time=None,
                           service_name: str = ''):
    # query = f"container_memory_usage_bytes/container_spec_memory_limit_bytes{{container_label_com_docker_swarm_service_name='{service_name}'}}*100"
    # print(query)
    query = f"rate(container_fs_usage_bytes{{container_label_com_docker_swarm_service_name='{service_name}'}}[1m]) / (1024*1024)"
    res_memory = []
    count = 1
    while count > 0:
        res_memory = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # print(res_memory)
        if len(res_memory['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_memory)


def get_container_fs_write(host: str = GLOBAL_CONFIG['prometheus_host'],
                           start_time=None,
                           end_time=None,
                           service_name: str = ''):
    # query = f"container_memory_usage_bytes/container_spec_memory_limit_bytes{{container_label_com_docker_swarm_service_name='{service_name}'}}*100"
    # print(query)
    query = f"rate(container_fs_write_seconds_total{{container_label_com_docker_swarm_service_name='{service_name}'}}[1m])"
    res_memory = []
    count = 1
    while count > 0:
        res_memory = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # print(res_memory)
        if len(res_memory['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_memory)


def get_container_fs_read(host: str = GLOBAL_CONFIG['prometheus_host'],
                          start_time=None,
                          end_time=None,
                          service_name: str = ''):
    query = f"rate(container_fs_read_seconds_total{{container_label_com_docker_swarm_service_name='{service_name}'}}[1m])"
    res_memory = []
    count = 1
    while count > 0:
        res_memory = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # print(res_memory)
        if len(res_memory['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_memory)


def get_container_net_receive(host: str = GLOBAL_CONFIG['prometheus_host'],
                              start_time=None,
                              end_time=None,
                              service_name: str = ''):
    query = f"rate(container_network_receive_bytes_total{{container_label_com_docker_swarm_service_name='{service_name}'}}[1m]) / 1024"
    res_memory = []
    count = 1
    while count > 0:
        res_memory = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # print(res_memory)
        if len(res_memory['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_memory)


def get_container_net_trainsmit(host: str = GLOBAL_CONFIG['prometheus_host'],
                                start_time=None,
                                end_time=None,
                                service_name: str = ''):
    query = f"rate(container_network_transmit_bytes_total{{container_label_com_docker_swarm_service_name='{service_name}'}}[1m]) / 1024"
    res_memory = []
    count = 1
    while count > 0:
        res_memory = fetch_prometheus(host, query, "range", 10, start_time, end_time).json()
        # print(res_memory)
        if len(res_memory['data']['result']) != 0:
            break
        time.sleep(2)
        count -= 1
    if count == 0:
        return -1
    else:
        return get_prometheus_response_container_value(res_memory)


# 递归删除文件夹及其内容
def delete_folder_recursive(folder_path):
    try:
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(folder_path)
    except Exception as e:
        print(f"error: {e}")


def init_config(csv_path: str, result_dir: Path, performance: int == 1):
    # 创建存储收集数据的文件
    path_list = [result_dir, result_dir / 'conf', result_dir / 'log', result_dir / 'trace']

    for p in path_list:
        if not p.exists():
            p.mkdir()
        else:
            delete_folder_recursive(p)
            p.mkdir()

    parameters = pd.read_csv(csv_path)
    key_services = parameters['microservice'].unique().tolist()
    curr_conf = {}
    config_dict = {}
    # 从基础配置文件中初始化配置字典，给出默认值
    for index, row in parameters.iterrows():
        if pd.notnull(row['parameter']):
            temp_dict = {}
            parameter_name = row['microservice'].replace('-', '_') + '_' + row['parameter'].replace('-', '_')
            range_list = row['range'].strip().split(';')
            if row['categorical'] == 1:
                range_list = [str(x) for x in range_list]
                # curr_conf[parameter_name] = range_list[0]
                curr_conf[parameter_name] = row['default']
                temp_dict['type'] = 'categorical'
                # temp_dict['length'] = len(range_list)
                temp_dict['range_list'] = range_list
                temp_dict['step'] = int(row['step'])
                # temp_dict['value'] = curr_conf[parameter_name]
                temp_dict['value'] = range_list.index(curr_conf[parameter_name])
            elif row['discrete'] == 1:
                range_list = [int(x) for x in range_list]
                # 由于已经出现性能违规，那么我们首先就应该放大资源使用量，先保证第一步不违规
                # 如果资源过剩的情况，那么就缩小资源
                if 'replicas' in parameter_name:
                    curr_conf[parameter_name] = int(row['default']) + 1
                    # curr_conf[parameter_name] = int(range_list[1]) if performance == 1 else int(range_list[0])
                    if 'nginx' in parameter_name or 'frontend' in parameter_name:
                        curr_conf[parameter_name] = 1
                elif 'resource' in parameter_name:
                    curr_conf[parameter_name] = int(row['default']) + 2 * int(row['step']) if performance == 1 else int(
                        int(row['default']) / 2)
                    prefix = parameter_name.split('resource')[0]
                    curr_conf[prefix + 'cpus'] = round(curr_conf[parameter_name] * 0.1, 2)
                    curr_conf[prefix + 'memory'] = round(curr_conf[parameter_name] * 0.2, 2)
                    # print(curr_conf[parameter_name])
                else:
                    # curr_conf[parameter_name] = int(sum(range_list) / len(range_list))
                    curr_conf[parameter_name] = int(row['default'])
                temp_dict['type'] = 'discrete'
                temp_dict['step'] = int(row['step'])
                temp_dict['value'] = curr_conf[parameter_name]
                temp_dict['min'] = range_list[0]
                temp_dict['max'] = range_list[1]
                # 动态更新resource的上下限，尽可能减少不必要的资源探索
                if 'resource' in parameter_name:
                    temp_dict['min'] = max(int(row['default']) - 10 * int(row['step']), 1)
                    temp_dict['max'] = int(row['default']) + 3 * int(row['step'])
            elif row['float'] == 1:
                range_list = [float(x) for x in range_list]
                # curr_conf[parameter_name] = round(float(sum(range_list) / len(range_list)), 2)
                curr_conf[parameter_name] = float(row['default'])
                temp_dict['type'] = 'float'
                temp_dict['step'] = float(row['step'])
                temp_dict['value'] = curr_conf[parameter_name]
                temp_dict['min'] = range_list[0]
                temp_dict['max'] = range_list[1]
            # print(temp_dict)
            # print(parameter_name)
            config_dict[parameter_name] = temp_dict

    # my_logger.info("init config:" + str(curr_conf))
    conf_path = result_dir / 'conf' / f"{0}_conf.yml"
    conf_path.write_text(yaml.safe_dump(curr_conf))
    return key_services, config_dict, conf_path


def init_config_from_key_services(csv_path: str, result_dir: Path, key_services,default_conf, performance: int == 1,
                                  parameters_type: int == 1):
    # 创建存储收集数据的文件
    path_list = [result_dir, result_dir / 'conf', result_dir / 'log', result_dir / 'trace']

    for p in path_list:
        if not p.exists():
            p.mkdir()
        else:
            delete_folder_recursive(p)
            p.mkdir()

    parameters = pd.read_csv(csv_path)
    # key_services = parameters['microservice'].unique().tolist()
    curr_conf = {}
    config_dict = {}
    default_conf = read_yaml(default_conf)
    for svc in key_services:
        para = None
        if parameters_type == 1:
            if 'redis_write' in svc:
                para = parameters[((parameters['microservice'] == 'redis') & (parameters['type'] == 'write')) | (
                        parameters['type'] == 'resource')]
            if 'redis_read' in svc:
                para = parameters[((parameters['microservice'] == 'redis') & (parameters['type'] == 'read')) | (
                        parameters['type'] == 'resource')]
            if 'mongodb_write' in svc:
                para = parameters[((parameters['microservice'] == 'mongodb') & (parameters['type'] == 'write')) | (
                        parameters['type'] == 'resource')]
            if 'mongodb_read' in svc:
                para = parameters[((parameters['microservice'] == 'mongodb') & (parameters['type'] == 'read')) | (
                        parameters['type'] == 'resource')]
            if 'memcached_write' in svc:
                para = parameters[((parameters['microservice'] == 'memcached') & (parameters['type'] == 'write')) | (
                        parameters['type'] == 'resource')]
            if 'memcached_read' in svc:
                para = parameters[((parameters['microservice'] == 'memcached') & (parameters['type'] == 'read')) | (
                        parameters['type'] == 'resource')]
            if 'nginx' in svc or 'frontend' in svc:
                para = parameters[(parameters['microservice'] == 'nginx') | (parameters['type'] == 'resource')]
            if 'service' in svc:
                para = parameters[(parameters['type'] == 'resource')]
        else:
            para = parameters[(parameters['type'] == 'resource')]
        # 从基础配置文件中初始化配置字典，给出默认值
        for index, row in para.iterrows():
            if pd.notnull(row['parameter']):
                temp_dict = {}
                parameter_name = svc.split('_')[0].replace('-', '_') + '_' + row['parameter'].replace('-', '_')
                range_list = row['range'].strip().split(';')
                if row['categorical'] == 1:
                    range_list = [str(x) for x in range_list]
                    # curr_conf[parameter_name] = range_list[0]
                    curr_conf[parameter_name] = row['default']
                    temp_dict['type'] = 'categorical'
                    # temp_dict['length'] = len(range_list)
                    temp_dict['range_list'] = range_list
                    temp_dict['step'] = int(row['step'])
                    # temp_dict['value'] = curr_conf[parameter_name]
                    temp_dict['value'] = range_list.index(curr_conf[parameter_name])
                elif row['discrete'] == 1:
                    range_list = [int(x) for x in range_list]
                    # 由于已经出现性能违规，那么我们首先就应该放大资源使用量，先保证第一步不违规
                    # 如果资源过剩的情况，那么就缩小资源
                    if 'replicas' in parameter_name:
                        curr_conf[parameter_name] = int(row['default']) + 1
                        # curr_conf[parameter_name] = int(range_list[1]) if performance == 1 else int(range_list[0])
                        if 'nginx' in parameter_name or 'frontend' in parameter_name:
                            curr_conf[parameter_name] = 1
                    elif 'resource' in parameter_name:
                        curr_conf[parameter_name] = int(row['default']) + 5 * int(
                            row['step']) if performance == 1 else int(
                            int(row['default']) / 2)
                        prefix = parameter_name.split('resource')[0]
                        curr_conf[prefix + 'cpus'] = round(curr_conf[parameter_name] * 0.1, 2)
                        curr_conf[prefix + 'memory'] = round(curr_conf[parameter_name] * 0.2, 2)
                        # print(curr_conf[parameter_name])
                    else:
                        # curr_conf[parameter_name] = int(sum(range_list) / len(range_list))
                        curr_conf[parameter_name] = int(row['default'])
                    temp_dict['type'] = 'discrete'
                    temp_dict['step'] = int(row['step'])
                    temp_dict['value'] = curr_conf[parameter_name]
                    temp_dict['min'] = range_list[0]
                    temp_dict['max'] = range_list[1]
                    # 动态更新resource的上下限，尽可能减少不必要的资源探索，当前服务默认配置开始
                    if 'resource' in parameter_name:
                        temp_dict['min'] = max(default_conf[parameter_name]-5, 1)
                        temp_dict['max'] = int(default_conf[parameter_name]) + 20 * int(row['step'])
                        # temp_dict['min'] = max(int(row['default']) - 5 * int(row['step']), 1)
                        # temp_dict['max'] = int(row['default']) + 10 * int(row['step'])
                elif row['float'] == 1:
                    range_list = [float(x) for x in range_list]
                    # curr_conf[parameter_name] = round(float(sum(range_list) / len(range_list)), 2)
                    curr_conf[parameter_name] = float(row['default'])
                    temp_dict['type'] = 'float'
                    temp_dict['step'] = float(row['step'])
                    temp_dict['value'] = curr_conf[parameter_name]
                    temp_dict['min'] = range_list[0]
                    temp_dict['max'] = range_list[1]
                # print(temp_dict)
                # print(parameter_name)
                config_dict[parameter_name] = temp_dict

    # my_logger.info("init config:" + str(curr_conf))
    print(config_dict)
    # return config_dict
    conf_path = result_dir / 'conf' / f"{0}_conf.yml"
    conf_path.write_text(yaml.safe_dump(curr_conf))
    return key_services, config_dict, conf_path


def read_yaml(conf_path: Path):
    with open(conf_path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def get_conf_row_count(conf_path: Path) -> int:
    with open(conf_path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        # Get the number of lines
        num_lines = len(data)
    return num_lines


def min_max(data_list: list, map_range: float = 1):
    min_num = min(data_list)
    max_num = max(data_list)
    mapped_data = map(lambda x: (x - min_num) * map_range / (max_num - min_num), data_list)
    return list(mapped_data)


def get_resource(conf_path: Path):
    # cpu_re = re.compile(r"(\w+)_(cpus):\s+(\d*\.?\d+)")
    resource_re = re.compile(r"(\w+)_(resource):\s+(\d*\.?\d+)")
    # memory_re = re.compile(r"(\w+)_(memory):\s+(\d*\.?\d+)")
    replicas_re = re.compile(r"(\w+)_(replicas):\s+(\d*\.?\d+)")
    # cpus_list = []
    # memory_list = []
    resource_list = []
    replicas_list = []
    content = conf_path.read_text()
    res = [0.0, 0.0]

    # for match in cpu_re.finditer(content):
    #     cpus_list.append(float(match.group(3)))
    # for match in memory_re.finditer(content):
    #     memory_list.append(float(match.group(3)))
    for match in replicas_re.finditer(content):
        replicas_list.append(float(match.group(3)))
    for match in resource_re.finditer(content):
        resource_list.append(float(match.group(3)))

    return sum([x * y for x, y in zip(resource_list, replicas_list)])
    # res[1] = sum([x * y for x, y in zip(memory_list, replicas_list)])

    # return res


def get_reward(pre_conf_dict: dict, curr_conf_path: Path):
    reward = []
    pre_resource = [0.0, 0.0]
    pre_cpu = []
    pre_memory = []
    pre_replicas = []
    # with open(curr_conf_path, "r") as f:
    #     curr_conf = yaml.safe_load(f)
    # print(curr_conf)
    for k, v in pre_conf_dict.items():
        if k.endswith('cpu'):
            pre_cpu.append(pre_conf_dict[k]['value'])
        if k.endswith('memory'):
            pre_memory.append(pre_conf_dict[k]['value'])
        if k.endswith('replicas'):
            pre_replicas.append(pre_conf_dict[k]['value'])
    pre_resource[0] = sum([x * y for x, y in zip(pre_cpu, pre_replicas)])
    pre_resource[1] = sum([x * y for x, y in zip(pre_memory, pre_replicas)])

    curr_resource = get_resource(curr_conf_path)
    reward.append(curr_resource[0] - pre_resource[0])
    reward.append(curr_resource[1] - pre_resource[1])

    # if k.endswith('cpu') or k.endswith('memory'):
    #     cost.append(curr_conf[k] - pre_conf_dict[k]['value'])
    # cost[0] = 10
    # print(len(cost))
    all_zero = all(x == 0 for x in reward)
    if all_zero:
        return 0
    else:
        reward = (min_max(reward, 2))
        return reward[0] * 0.6 + reward[1] * 0.4


def get_config_dict_resource(config_dict):
    resource = 0
    for k, v in config_dict.items():
        if 'resource' in k:
            resource += v['value'] * config_dict[k.split('resource')[0] + 'replicas']['value']
    return round(resource, 2)


def get_conf_vector(config_dict):
    # 得到每次配置的值，用于相似度计算
    conf_vector = []
    for k, v in config_dict.items():
        conf_vector.append(v['value'])
    return conf_vector


def compute_similarity(replay_path, curr_conf_vector, config_dict, threshold, key_services, workload, slo):
    replay_csv = pd.read_csv(replay_path)
    if replay_csv.shape[0] == 0:
        return False, None, None, None
    conf_vector_list = []
    for index, row in replay_csv.iterrows():
        conf = []
        tmp = []
        obs = []
        for i in row['conf'].split(';'):
            conf.append(float(i))
        for i in row['observation'].split(';'):
            obs.append(float(i))
        tmp.append(conf)
        tmp.append(float(row['latency']))
        tmp.append(float(row['resource']))
        tmp.append(obs)
        conf_vector_list.append(tmp)
    # similarity = []
    max_similarity = 0
    latency = 0
    resource = get_config_dict_resource(config_dict)
    observation = None
    for vector in conf_vector_list:
        cosine_similarity = np.dot(curr_conf_vector, vector[0]) / (
                np.linalg.norm(curr_conf_vector) * np.linalg.norm(vector[0]))
        if cosine_similarity > max_similarity:
            max_similarity = cosine_similarity
            latency = vector[1]
            # resource = vector[2]
            observation = vector[3]

        # similarity.append(cosine_similarity)
    if max_similarity > threshold:
        new_obs = []
        for i in range(2 * len(key_services)):
            noise = round(np.random.normal(0, 0.1), 2)
            new_obs.append(observation[i] + noise)
        noise = round(np.random.normal(0, 2), 2)
        latency += noise
        new_obs += [resource] + workload + [latency]
        # 定义随机值，以95%的概率启用历史数据，5%的概率重新部署
        values = [0, 1]
        weights = [0.1, 0.9]
        # 使用 choices() 函数生成随机值
        replay = random.choices(values, weights)[0]
        if replay == 1 and latency < slo:
            latency = round(latency, 2)
            new_obs = [round(i, 2) for i in new_obs]
            return True, latency, resource, new_obs
        else:
            return False, 0, 0, new_obs
    else:
        return False, 0, 0, []


def get_bench_name(_env_id):
    bench = _env_id.split('-')[1].split('_')[0]
    if bench == 'sn' or bench == 'social_network':
        return 'social_network'
    if bench == 'mm':
        return 'media_microservices'
    if bench == 'tt':
        return 'train_ticket'
    if bench == 'hr':
        return 'hotel_reservation'


import argparse


def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="automatic optimization framework for microservices"
    )
    parser.add_argument(
        "-tn",
        "--task_name",
        help="the name of the optimization task. can use for generating result directory.",
    )

    return parser.parse_args()


def clear_folder(folder_path):
    # 获取文件夹下的所有文件
    file_list = os.listdir(folder_path)
    # 遍历文件夹下的所有文件，并删除
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


# 检测文件是否存在
def check_file_exists(file_path):
    if os.path.exists(file_path):
        pass
        # print(f"文件'{file_path}'存在。")
    else:
        print(f"文件'{file_path}'不存在。")


def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# 复制文件并改名
def copy_and_rename_file(source_file, destination_folder, new_filename):
    try:
        with open(source_file, 'rb') as f_source:
            with open(os.path.join(destination_folder, new_filename), 'wb') as f_destination:
                f_destination.write(f_source.read())
        os.rename(os.path.join(destination_folder, new_filename), os.path.join(destination_folder, new_filename))
        # print(f"file'{source_file}' rename to '{new_filename}'")
    except Exception as e:
        print(f"error: {e}")


# 把list写入txt
def list2txt(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(str(item) + '\n')


def txt2list(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(str(line.strip()))
    return data

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data