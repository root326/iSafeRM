import requests
import time
import argparse
from tunning.constants import BENCHMARK_CONFIG,TMP_DIR




def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="software runtime metrics collect"
    )
    parser.add_argument(
        "--software_name",
        help="the name of software,such as mongodb ,redis ",
    )
    parser.add_argument(
        "--files_num",
        help="抓取多少次指标",
        default=3
    )
    return parser.parse_args()

def get_request_txt(benchmark, count):
    interval = BENCHMARK_CONFIG['software'][benchmark]['interval']
    metrics_url = BENCHMARK_CONFIG['software'][benchmark]['exporter_url']
    time.sleep(interval)
    txt_list = [f"{TMP_DIR}/.{benchmark}_tmp/output/metrics_{i + 1}.txt" for i in range(count)]
    for i in range(count):
        response = requests.get(metrics_url)
        content = response.text
        with open(txt_list[i], "w") as file:
            file.write(content)
        if i < count - 1:
            time.sleep(10)
    return txt_list


def metrics_filtered(source_file, result_file):
    with open(source_file, 'r') as file:
        lines = file.readlines()
        filtered_lines = [line.strip() for line in lines if not line.startswith('#')]
        del_line = []
        for line in filtered_lines:
            if line.split(' ')[1] == '0' or 'commands' in line.split(' ')[0] or 'e' in line.split(' ')[1] \
                    or float(line.split(' ')[1]) < 10 or float(line.split(' ')[1]) > 100000:
                del_line.append(line)
        filtered_lines = [line for line in filtered_lines if line not in del_line]
        with open(result_file, "w") as file:
            for line in filtered_lines:
                file.write(line + '\n')


def get_metrics_from_txt(txt):
    with open(txt, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    result = []
    for line in lines:
        if len(line.split('{')) != 1:
            line = line.split('{')[0] + ' ' + line.split(' ')[1]
        result.append(line)
    return result