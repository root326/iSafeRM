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
    # 发送GET请求获取网页内容
    # url = "http://192.168.1.180:9216/metrics"
    txt_list = [f"{TMP_DIR}/.{benchmark}_tmp/output/metrics_{i + 1}.txt" for i in range(count)]
    for i in range(count):
        # 运行时指标，随着负载在变化，抓取多次取metrics平均值
        response = requests.get(metrics_url)
        content = response.text
        # 将内容保存到txt文件
        with open(txt_list[i], "w") as file:
            file.write(content)
        # prometheus 15s抓取一次
        if i < count - 1:
            time.sleep(10)
    return txt_list


def metrics_filtered(source_file, result_file):
    # 去掉首字符是#的每一行内容
    with open(source_file, 'r') as file:
        lines = file.readlines()
        filtered_lines = [line.strip() for line in lines if not line.startswith('#')]
        del_line = []
        for line in filtered_lines:
            if line.split(' ')[1] == '0' or 'commands' in line.split(' ')[0] or 'e' in line.split(' ')[1] \
                    or float(line.split(' ')[1]) < 10 or float(line.split(' ')[1]) > 100000:
                del_line.append(line)
        filtered_lines = [line for line in filtered_lines if line not in del_line]
        # print(filtered_lines)
        # for line in filtered_lines:
        #     print(line)
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
        # print(line)
    return result

# if __name__ == "__main__":
#     args = vars(parser_args())
#     metrics_url = ''
#     if args.get('software_name') == 'mongodb':
#         metrics_url = MONGODB_EXPORTER_URL
#     get_request_txt(metrics_url,args.get('files_num'))
