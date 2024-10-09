import pandas as pd
import json
from tunning.utils import check_file_exists, save_dict, copy_and_rename_file, load_dict,load_json,my_logger
from tunning.PBSLocator.rca import compute_causal_services, compute_causal_services_dict
from pathlib import Path
import omnisafe
import os
from tunning.bench.microservices_benchmark import MicroservicesBenchmark
from tunning.constants import GLOBAL_CONFIG
# current_file_path = os.path.abspath(__file__)
# tuning_folder = os.path.dirname(os.path.dirname(current_file_path))
tuning_folder = GLOBAL_CONFIG["project_root_dir"]
import argparse
from tunning.datacollector.trace_collector import TraceCollector
import time
# 在线运行打开
# from tunning.benchenvs.microservices_env import MicroservicesENV
# 离线运行打开
from tunning.benchenvs.microservices_env_offline import MicroservicesENVOffline

def coast_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} coast time:{time.perf_counter() - t:.8f} s')
        return result

    return fun

def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="automatic optimization framework for microservices"
    )
    parser.add_argument(
        "-tn",
        "--task_name",
        help="The microservice name of the optimization task.",
        default="sn",
    )
    parser.add_argument(
        "-on",
        "--online",
        help="Whether to select online optimization. no or yes",
        default="no",
    )
    parser.add_argument(
        "-rca",
        "--root_cause_analysis",
        help="Whether to root cause analysis. no or yes",
        default="no",
    )
    parser.add_argument(
        "-wk",
        "--workload",
        help="Current microservice workloads.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-dc",
        "--default_config",
        help="Default configuration file for the current microservice",
        default=f"{tuning_folder}/deploy/social_network/ansible/default_conf.yml",
    )
    parser.add_argument(
        "-ts",
        "--total_steps",
        help="total number of steps to train",
        type=int,
        default=256,
    )
    parser.add_argument(
        "-spe",
        "--steps_per_epoch",
        help="number of steps to update the policy",
        type=int,
        default=64,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help=" batch size for each iteration",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-ui",
        "--update_iters",
        help="number of iterations to update the policy",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-tk",
        "--top_k",
        help="Select the first k service optimizations",
        type=int,
        default=9,
    )
    parser.add_argument(
        "-rcadir",
        "--rca_dir",
        default=f"{tuning_folder}/data/rca/",
    )
    parser.add_argument(
        "-traindir",
        "--train_dir",
        default=f"{tuning_folder}/output/run/",
    )

    return parser.parse_args()


class safescaler:
    def __init__(self,args):
        self._env = args["task_name"]
        self._args = args
        self.benchmark, self._namespace = self.get_bench_name()
        self._bench = self.get_benchmark()
        self._rca_dir = args["rca_dir"]
        self._trace_collector = TraceCollector(
            bench_name=self.benchmark,
            resultPath=self._rca_dir,
            max_traces=1000
        )
        self._workload = args["workload"]
        self._default_conf = args["default_config"]


    def deploy_ms(self):
        # 启动服务
        self._bench.cleanup(0)
        start_time = time.time()-300
        self._bench.run(conf_path=self._default_conf, workload=[self._workload], task_id=0)

        for i in range(3):
            self._bench.run_workload(self._workload)

        # 收集延迟数据
        self._trace_collector.set_task_id(0)
        self._trace_collector.collect_ms_data(start_time)

        # self._ms_data = self._trace_collector.get_ms_data_csv()
        # self._trace_collector.close_pool()

        # root cause analysis
        # self.services_rca(self._rca_dir)




    def get_benchmark(self):
        return MicroservicesBenchmark(self.benchmark)

    def get_bench_name(self):
        bench = self._env
        if bench == 'sn':
            return 'social_network', bench
        if bench == 'mm':
            return 'media_microservices', bench
        if bench == 'tt':
            return 'train_ticket', bench
        if bench == 'hr':
            return 'hotel_reservation', bench

    def services_rca(self, data_path=None):
        if data_path is None:
            data_path = self._args["rca_dir"]
        data_path = Path(data_path)
        file_list = [data_path / 'ms_data.csv', data_path / 'causal_graph.gml']

        for file in file_list:
            check_file_exists(file)

        top_services = compute_causal_services_dict(file_list[0], file_list[1])

        with open(data_path / 'top_services.json', 'w') as file:
            json.dump(top_services, file)

        return top_services.keys()


    def train_online(self):
        # args = self._args
        env_id = 'wk-sn_key_services_key_parameters'
        custom_cfgs = {
            'train_cfgs': {
                'total_steps': self._args["total_steps"],  # 256
                'vector_env_nums': 1,
                'parallel': 1,
                'torch_threads': 16
            },
            'algo_cfgs': {
                'steps_per_epoch': self._args["steps_per_epoch"],  # 64
                'safety_budget': 100,
                'update_iters': self._args["update_iters"],  # 4
                'batch_size': self._args["batch_size"],  # 16
                'unsafe_reward': -1.0
                # 'start_learning_steps': 0,
            },
            'logger_cfgs': {
                'log_dir': self._args["train_dir"],
                'save_model_freq': 1
            },
        }
        agent = omnisafe.Agent('PPOSaute', env_id, custom_cfgs=custom_cfgs)
        agent.learn()

    @coast_time
    def train_offline(self):
        # args = self._args
        env_id = 'wk10-sn_key_services_key_parameters'
        custom_cfgs = {
            'train_cfgs': {
                'total_steps': self._args["total_steps"],  # 256
                'vector_env_nums': 1,
                'parallel': 1,
                'torch_threads': 16
            },
            'algo_cfgs': {
                'steps_per_epoch': self._args["steps_per_epoch"],  # 64
                'safety_budget': 100,
                'update_iters': self._args["update_iters"],  # 4
                'batch_size': self._args["batch_size"],  # 16
                'unsafe_reward': -2.0
                # 'start_learning_steps': 0,
            },
            'logger_cfgs': {
                'log_dir': self._args["train_dir"],
                'save_model_freq': 1
            },
        }
        agent = omnisafe.Agent('PPOSaute', env_id, custom_cfgs=custom_cfgs)
        agent.learn()




if __name__ == "__main__":


    args = vars(parser_args())
    # print(list(load_json("/home/lilong/iSafeRM/data/rca/top_services.json").keys()))

    # print(load_dict(tuning_folder+'/output/config'))
    model = safescaler(args)

    if args["online"] == "no":
        if args["root_cause_analysis"] == "yes":
            model.services_rca(args["rca_dir"])
            # model.train_offline()
        else:

            model.train_offline()
    else:
        my_logger.info("deploying microservices")
        # model.deploy_ms()
        my_logger.info("root cause analysis")
        # model.services_rca()
        args["key_services"] = list(load_json("//data/rca/top_services.json").keys())[
                               :args["top_k"]]
        save_dict(args, f"{tuning_folder}/output/config")
        my_logger.info("iSafeRM online")
        model.train_online()

