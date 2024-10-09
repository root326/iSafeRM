from tunning.datacollector.trace_collector import TraceCollector
from tunning.constants import BENCHMARK_CONFIG
import yaml
import pandas as pd
from pathlib import Path
import time
import networkx as nx
from dowhy import gcm
import pandas as pd
from scipy.stats import halfnorm
import numpy as np
from tunning.bench.microservices_benchmark import MicroservicesBenchmark
from tunning.utils import get_bench_name, parser_args
import re
# from experiments.SafeScaler.PBA import PBScaler
# from PBScaler.Config import Config

def get_benchmark(benchmark):
    return MicroservicesBenchmark(benchmark)


def coast_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} coast time:{time.perf_counter() - t:.8f} s')
        return result
    return fun


class BenchMark:
    def __init__(self, benchmark, env, steps, task_name):
        # self._benchmark = benchmark
        self._benchmark = get_bench_name(env)
        self._env = env
        self._steps = steps
        env_dir = self._env.split('-')[0]
        self._workload = [int(re.findall(r'\d+', env_dir)[0])]
        self._csv_path = BENCHMARK_CONFIG['microservices'][self._benchmark][
                             'configs_dir']  / f"{self._env.split('-')[1]}.csv"
        # self._csv_path = BENCHMARK_CONFIG['microservices'][self._benchmark]['configs_dir'] / f"{self._env}.csv"
        self._result_path = BENCHMARK_CONFIG['microservices'][self._benchmark][
                                'experiments_results'] / f"{task_name}-{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        self._bench = get_benchmark(self._benchmark)
        self._output = BENCHMARK_CONFIG['microservices'][self._benchmark]['output']
        self._task_id = 0
        self._conf_dict = None
        self._conf = None
        self.init_config()
        self.data_collector = TraceCollector(
            bench_name=self._benchmark,
            resultPath=self._result_path / 'trace',
        )
        self._bench.cleanup(0)
        # self.pbscaler =

    def init_config(self):
        # 创建存储收集数据的文件
        path_list = [self._result_path, self._result_path / 'conf', self._result_path / 'log',
                     self._result_path / 'trace']
        for p in path_list:
            if not p.exists():
                p.mkdir()

        self._conf = self._result_path / 'conf' / f"{self._task_id}_conf.yml"
        self.get_resource_conf_experiment(self._csv_path, self._conf)
        # = conf_path

    def get_resource_conf_experiment(self, csv_path, conf_path):
        parameters = pd.read_csv(csv_path)
        curr_conf = {}
        for index, row in parameters.iterrows():
            if pd.notnull(row['parameter']):
                parameter_name = row['microservice'].replace('-', '_') + '_' + row['parameter'].replace('-', '_')
                if 'replicas' in parameter_name:
                    curr_conf[parameter_name] = 1
                    # curr_conf[parameter_name] = int(range_list[1]) if performance == 1 else int(range_list[0])
                    if 'nginx' in parameter_name or 'frontend' in parameter_name:
                        curr_conf[parameter_name] = 1
                elif 'resource' in parameter_name:
                    curr_conf[parameter_name] = self._task_id+1
                    prefix = parameter_name.split('resource')[0]
                    curr_conf[prefix + 'cpus'] = round(curr_conf[parameter_name] * 0.1, 2)
                    curr_conf[prefix + 'memory'] = round(curr_conf[parameter_name] * 0.2, 2)

                # prefix = parameter_name.split('resource')[0]
                # curr_conf[prefix + 'cpus'] = round(int(self._task_id + 1) * 0.1, 2)
                # curr_conf[prefix + 'memory'] = round(int(self._task_id + 1) * 0.2, 2)

        conf_path.write_text(yaml.safe_dump(curr_conf))
        return conf_path

    def run(self):
        for i in range(self._steps):
            start_time = time.time()
            self._bench.run(self._conf, self._workload, 0)
            # self._bench.run_workload()
            for j in range(5):
                self._bench.run_workload(self._workload[0])

            self.data_collector.set_task_id(self._task_id)
            self.data_collector.collect_ms_data(start_time=start_time)
            ms_data = self.data_collector.get_ms_data_csv()
            for f in Path(self._output).iterdir():
                f.rename(self._result_path / 'log' / f"{self._task_id}_{f.name}")

            self._bench.cleanup(0)
            self._workload = [self._workload[0] + self._task_id *5]
            self._task_id += 1

            self._conf = self._result_path / 'conf' / f"{self._task_id}_conf.yml"
            self.get_resource_conf_experiment(self._csv_path, self._conf)


def get_top_services(causal_scores):
    spans = sorted(causal_scores.items(), key=lambda x: x[1], reverse=True)
    top_spans = [v[0] for v in spans if v[1] > 0]
    top_services = []
    for span in top_spans:
        str_list = span.split('_')
        service_name = ''
        if 'Redis' in str_list[1]:
            service_name = str_list[0].split('service')[0] + 'redis'
        elif 'Mongo' in str_list[1]:
            service_name = str_list[0].split('service')[0] + 'mongodb'
        elif 'Memcached' in str_list[1]:
            service_name = str_list[0].split('service')[0] + 'memcached'
        else:
            service_name = str_list[0]
        if service_name not in top_services:
            top_services.append(service_name)
    return top_services


# top_services = list(set(top_services))
# top_services = get_top_services(median_attribs)
# top_services


@coast_time
def compute_causal_services(ms_csv, ms_causal_graph, slo=200000):
    ms_data = pd.read_csv(ms_csv)
    # ms_data['traceLatency']
    ms_normal = ms_data[ms_data['traceLatency'] < slo].head(1000)
    ms_normal_len = len(ms_normal)
    ms_abnormal = ms_data[ms_data['traceLatency'] > slo].head(ms_normal_len)
    print(ms_normal.count())
    print(ms_abnormal.count())
    causal_graph = nx.read_gml(ms_causal_graph)
    target_node = [node for node, out_degree in causal_graph.out_degree() if out_degree == 0][0]
    causal_model = gcm.StructuralCausalModel(causal_graph)

    for node in causal_graph.nodes:
        if len(list(causal_graph.predecessors(node))) > 0:
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
        else:
            causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(halfnorm))
    median_attribs, uncertainty_attribs = gcm.confidence_intervals(
        lambda: gcm.distribution_change(causal_model,
                                        ms_normal.sample(frac=0.6),
                                        ms_abnormal.sample(frac=0.6),
                                        target_node,
                                        difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x)),
        num_bootstrap_resamples=10)
    return get_top_services(median_attribs)


if __name__ == "__main__":
    args = parser_args()
    if args.task_name == 'sn':
        benchmark = 'social_network'
        env = 'wk20-sn_cp_resource'
        agent = BenchMark(benchmark=benchmark, env=env, steps=30, task_name='collect-traces')
        agent.run()
    if args.task_name == 'mm':
        benchmark = ''
        env = 'wk30-mm_cp_resource'
        agent = BenchMark(benchmark=benchmark, env=env, steps=30, task_name='collect-traces')
        agent.run()
    if args.task_name == 'hr':
        benchmark = ''
        env = 'wk20-hr_cp_resource'
        agent = BenchMark(benchmark=benchmark, env=env, steps=30, task_name='collect-traces')
        agent.run()
