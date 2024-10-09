import os
import time
from pathlib import Path
from typing import Any, Optional
import random
import numpy as np
import torch
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace
from gymnasium.spaces import Box
import yaml
from tunning.constants import BENCHMARK_CONFIG, GLOBAL_CONFIG
from tunning.utils import get_conf_row_count, init_config, format_time, get_node_cpu_usage, \
    get_node_memory_usage, get_reward, get_container_cpu_usage, get_container_memory_usage
from tunning.utils import my_logger
from tunning.bench.microservices_benchmark import MicroservicesBenchmark
import pandas as pd
import re


@env_register
class MicroservicesENV(CMDP):
    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False
    _support_envs: list[str] = GLOBAL_CONFIG['envs_id']

    def __init__(self, env_id: str,
                 device: torch.device = DEVICE_CPU,
                 **kwargs: Any) -> None:
        # print(env_id.split('-')[1])
        super().__init__(env_id)
        self._env_id = env_id

        self.benchmark, self._namespace = self.get_bench_name()
        my_logger.info("env create: " + self._env_id)
        env_dir = self._env_id.split('-')[0]
        self._workload = [int(re.findall(r'\d+',env_dir)[0])]
        # if self._namespace == 'hr' or 'sn':
        #     env_dir = self.dynamic_tuning()
        self._num_envs = 1
        self._device = torch.device(device)
        self._csv_path = BENCHMARK_CONFIG['microservices'][self.benchmark][
                             'configs_dir'] / f"{env_dir}" / f"{self._env_id.split('-')[1]}.csv"
        self._result_path = BENCHMARK_CONFIG['microservices'][self.benchmark][
                                'results'] / f"iSafeRM-{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}-{self._env_id}"
        my_logger.info("result path: " + str(self._result_path))
        self._slo = BENCHMARK_CONFIG['microservices'][self.benchmark]['slo']
        self._output = BENCHMARK_CONFIG['microservices'][self.benchmark]['output']
        self._key_services, self._config_dict, self._conf_path = init_config(self._csv_path, self._result_path,
                                                                             performance=1)
        self._result_path_conf = self._result_path / 'conf'
        self._result_path_log = self._result_path / 'log'
        self._result_path_trace = self._result_path / 'trace'
        self._results_summary_file = self._result_path / f"_results_summary.csv"
        results_summary_file = pd.DataFrame(columns=GLOBAL_CONFIG['result_file_column'])
        results_summary_file.to_csv(self._results_summary_file, index=False)
        # self._act_dim = get_conf_row_count(self._conf_path)
        self._act_dim = len(self._config_dict)
        # cpu + memory + workload + resource + latency
        # self._workload = BENCHMARK_CONFIG['microservices'][self.benchmark]['workload']
        self._obs_dim = len(self._key_services) * 2 + len(self._workload) + 1 + 1
        self._task_id = 0
        random.seed(123)
        self._action_space = Box(low=-1.0, high=1.0, shape=(self._act_dim,), dtype=np.float32)
        self._observation_space = Box(low=0.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32)
        self._bench = self.get_benchmark()
        self._pre_resource = self.get_resource()
        self._curr_resource = 0
        self._pre_latency = 999999999
        self._curr_latency = 0
        self._lambda1 = 0.7
        self._lambda2 = 0.3
        self._cost_flag = True
        self._cpu_usage = [0 for i in range(len(self._key_services))]
        self._mem_usage = [0 for i in range(len(self._key_services))]
        self._init_obs = []
        self._bench.cleanup(self._task_id)
        self._repetition = 1
        self._threshold = 0.9999
        self._replay = False
        self._last_safe_obs = []
        self._last_safe_conf_dict = {}
        # self._init_env()

    def get_bench_name(self):
        bench = self._env_id.split('-')[1].split('_')[0]
        # print(self._namespace)
        if bench == 'sn':
            return 'social_network', bench
        if bench == 'mm':
            return 'media_microservices', bench
        if bench == 'tt':
            return 'train_ticket', bench
        if bench == 'hr':
            return 'hotel_reservation', bench

    def _init_env(self):
        # self._bench.cleanup(self._task_id)

        obs, _, _, _, _, info = self.run_config()
        self._init_obs = obs

        return obs, info


    def get_resource(self):
        resource = 0
        for k, v in self._config_dict.items():
            if 'resource' in k:
                resource += v['value'] * self._config_dict[k.split('resource')[0] + 'replicas']['value']
        return round(resource, 2)

    def get_score(self):
        resource_ratio = round(self._pre_resource / self._curr_resource, 2)
        score = resource_ratio if resource_ratio >= 1 else 0
        return score

    def get_performance(self):
        latency_ratio = round(self._pre_latency / self._curr_latency, 2)
        performance = latency_ratio if latency_ratio >= 1 else 0
        return performance

    def get_cost(self):
        # 默认预算是100，当延迟第一次超过100的时候才开始消耗安全代价
        if self._curr_latency <= self._slo - 100:
            return 0
        if self._cost_flag:
            self._cost_flag = False
            return round(self._curr_latency - ( self._slo - 100), 2)
        else:
            # if self._curr_latency < self._pre_latency:
            #     cost = 0
            # else:
            #     cost = self._curr_latency - self._pre_latency
            cost = 0 if self._curr_latency < self._pre_latency else self._curr_latency - self._pre_latency
            return round(cost, 2)

    def get_reward(self):
        reward = min(self._lambda1 * self.get_score(), 2) + min(self._lambda2 * self.get_performance(), 1)
        reward = round(reward, 2)
        return reward


    def get_key_services_cpu_and_memory_usage(self, start_time, end_time):
        key_services_cpu = []
        key_services_memory = []
        for service in self._key_services:
            # print(service)
            key_services_cpu.append(
                get_container_cpu_usage(start_time=start_time, end_time=end_time, service_name=f"{self._namespace}_{service}"))
            key_services_memory.append(
                get_container_memory_usage(start_time=start_time, end_time=end_time, service_name=f"{self._namespace}_{service}"))
        # 若获取失败，则利用以前的数据
        for i in range(len(key_services_cpu)):
            if -1 in key_services_cpu:
                index = key_services_cpu.index(-1)
                key_services_cpu[index] = self._cpu_usage[index]
            else:
                break
        for i in range(len(key_services_memory)):
            if -1 in key_services_memory:
                index = key_services_memory.index(-1)
                key_services_memory[index] = self._mem_usage[index]
            else:
                break
        self._cpu_usage = key_services_cpu
        self._mem_usage = key_services_memory
        return key_services_cpu, key_services_memory

    def run_config(self):
        my_logger.info("Task " + str(self._task_id) + " start, please wait several minutes to generate result...")
        start_time = 0
        end_time = 0
        latency, resource = 0, 0
        cost,reward = -1,-1
        terminated = False
        truncated = False
        info = {}
        for i in range(self._repetition):
            latency_flag = False
            start_time = time.time()
            self._bench.run(self._conf_path, self._workload, self._task_id)
            end_time = time.time()
            latency, resource = self._bench.parse_result(self._conf_path)
            if latency < self._slo:
                latency_flag = True
            else:
                for l in range(4):
                    self._bench.run_workload(self._workload[0])
                    latency, resource = self._bench.parse_result(self._conf_path)
                    if latency < self._slo :
                        latency_flag = True
                        break
            if latency_flag:
                break
            if i < self._repetition - 1:
                self._bench.cleanup(self._task_id)

        cpu_usage, mem_usage = self.get_key_services_cpu_and_memory_usage(start_time=end_time-5*60, end_time=end_time)

        for f in Path(self._output).iterdir():
            f.rename(self._result_path_log / f"{self._task_id}_{f.name}")
        # 获取环境状态
        observation = cpu_usage + mem_usage + [resource] + self._workload + [latency]
        my_logger.info("original_observation: " + str(observation))
        # observation = [round(x, 2) for x in min_max(observation, 10)]

        # reward
        self._curr_resource = resource
        self._curr_latency = latency
        reward = self.get_reward()
        cost = self.get_cost()
        if latency > self._slo:
            terminated = True
            # info
            self._cost_flag = True
            info = {'final_observation': observation}
        if latency < self._slo * 0.5:
            self._last_safe_obs = observation
            self._last_safe_conf_dict = self._config_dict
        result_temp = {}
        # if reward != 0:
        my_logger.info("observation: " + str(observation))
        my_logger.info("terminated/truncated: " + str(terminated))
        my_logger.info("reward: " + str(reward))
        my_logger.info("cost: " + str(cost))

        result_temp['task'] = self._task_id
        result_temp['latency'] = latency
        result_temp['resource'] = resource
        result_temp['observation'] = [';'.join([str(x) for x in observation])]
        result_temp['reward'] = reward
        result_temp['cost'] = cost
        result_temp['workload'] = [';'.join([str(x) for x in self._workload])]
        conf_vector = self.get_conf_vector()
        result_temp['conf'] = [';'.join([str(x) for x in conf_vector])]
        result_data = pd.DataFrame(result_temp)  # a需要是字典格式
        # mode='a'表示追加, index=True表示给每行数据加索引序号, header=False表示不加标题
        result_data.to_csv(self._results_summary_file, mode='a', index=False, header=False)
        # result_data.to_csv(self._replay_path, mode='a', index=False, header=False)
        self._task_id += 1
        self._pre_latency = latency
        self._pre_resource = resource
        return observation, reward, cost, terminated, truncated, info

    def get_curr_conf_2(self,action):
        i = 0
        curr_conf = {}
        for k, v in self._config_dict.items():
            a_v = (action[i]+1)/2
            if a_v <= 0:
                a_v = 0
            elif a_v >= 1:
                a_v = 1
            if v['type'] == 'categorical':
                if a_v == 1:
                    index = len(v['range_list'])-1
                else:
                    index = int(a_v * len(v['range_list']))
                curr_conf[k] = v['range_list'][index]
                self._config_dict[k]['value'] = index
            elif v['type'] == 'discrete':
                curr_conf[k] = int(int(v['min']) + a_v * (int(v['max']) - int(v['min'])))
                if 'resource' in k:
                    prefix = k.split('resource')[0]
                    curr_conf[prefix + 'cpus'] = round(curr_conf[k] * 0.1, 2)
                    curr_conf[prefix + 'memory'] = round(curr_conf[k] * 0.2, 2)
                self._config_dict[k]['value'] = curr_conf[k]
            elif v['type'] == 'float':
                curr_conf[k] = round(float(float(v['min']) + a_v * (float(v['max']) - float(v['min']))),1)
                self._config_dict[k]['value'] = curr_conf[k]
            i += 1
        return curr_conf


    def get_curr_conf_1(self,action):
        i = 0
        coefficient = 0
        curr_conf = {}
        for k, v in self._config_dict.items():
            if action[i] > 0.3:
                coefficient = 1
            elif action[i] < -0.3:
                coefficient = -1
            else:
                coefficient = 0
            if v['type'] == 'categorical':
                # index = int(action[i] / (1 / len(v['range_list'])))
                index = int(v['value'] + v['step'] * coefficient)
                if index < 0 or index >= len(v['range_list']):
                    curr_conf[k] = v['range_list'][0]
                else:
                    curr_conf[k] = v['range_list'][index]
                self._config_dict[k]['value'] = index
                # curr_config_dict[k] = curr_conf[k]
            elif v['type'] == 'discrete':
                curr_conf[k] = int(v['value'] + v['step'] * coefficient)
                if curr_conf[k] < v['min']:
                    curr_conf[k] = round(int(v['min']), 1)
                if curr_conf[k] > v['max']:
                    curr_conf[k] = round(int(v['max']), 1)
                # if 'nginx' in k and 'replicas' in k:
                #     curr_conf[k] = 1
                if 'resource' in k:
                    prefix = k.split('resource')[0]
                    if self._namespace == 'tt':
                        curr_conf[k] *= 2
                    curr_conf[prefix + 'cpus'] = round(curr_conf[k] * 0.1, 2)
                    curr_conf[prefix + 'memory'] = round(curr_conf[k] * 0.2, 2)
                self._config_dict[k]['value'] = curr_conf[k]
            elif v['type'] == 'float':
                curr_conf[k] = round(
                    float(v['value'] + v['step'] * coefficient), 1)
                if curr_conf[k] < v['min']:
                    curr_conf[k] = round(float(v['min']), 1)
                if curr_conf[k] > v['max']:
                    curr_conf[k] = round(float(v['max']), 1)
                # 更新记录的参数值
                self._config_dict[k]['value'] = curr_conf[k]
            i += 1
        return curr_conf



    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        # my_logger.info("original_action: " + str(action))
        # action = [round(y, 2) for y in min_max([x for x in action.cpu().numpy()])]
        action = [x for x in action.cpu().numpy()]
        my_logger.info("action: " + str(action))

        curr_conf = self.get_curr_conf_1(action)
        self._conf_path = self._result_path_conf / f"{self._task_id}_conf.yml"
        my_logger.info("save task conf:" + str(self._conf_path))
        self._conf_path.write_text(yaml.safe_dump(curr_conf))
        obs, reward, cost, terminated, truncated, info = self.run_config()
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )
        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> tuple[torch.Tensor, dict[str, Any]]:
        obs = []
        if self._task_id == 0:
            self._init_obs, info = self._init_env()
            obs = self._init_obs
            my_logger.warning("env init")
        else:
            obs = self._last_safe_obs
            self._config_dict = self._last_safe_conf_dict
            my_logger.warning("env reset and reload last safe conf and obs ")
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), {}

    def set_seed(self, seed: int) -> None:
        # self.reset()
        pass

    def sample_action(self) -> torch.Tensor:
        return torch.as_tensor(
            self._action_space.sample(),
            dtype=torch.float32,
            device=self._device,
        )

    def render(self) -> Any:
        pass

    def close(self) -> None:
        pass

    def get_benchmark(self):
        return MicroservicesBenchmark(self.benchmark)

    def get_conf_vector(self):
        # 得到每次配置的值，用于相似度计算
        conf_vector = []
        for k, v in self._config_dict.items():
            conf_vector.append(v['value'])
        return conf_vector