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
from tunning.utils import init_config, copy_and_rename_file
from tunning.bench.microservices_benchmark import MicroservicesBenchmark
import pandas as pd
import re
import argparse

current_file_path = os.path.abspath(__file__)
# tuning_folder = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
tuning_folder = f"{GLOBAL_CONFIG['project_root_dir']}"

@env_register
class MicroservicesENVOffline(CMDP):
    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False
    _support_envs: list[str] = GLOBAL_CONFIG['envs_id']

    def __init__(self, env_id: str,
                 device: torch.device = DEVICE_CPU,
                 **kwargs: Any) -> None:
        super().__init__(env_id)
        self._env_id = env_id
        self.benchmark, self._namespace = self.get_bench_name()
        env_dir = self._env_id.split('-')[0]
        self._workload = [int(re.findall(r'\d+',env_dir)[0])]
        self._num_envs = 1
        self._device = torch.device(device)
        self._csv_path = BENCHMARK_CONFIG['microservices'][self.benchmark][
                             'configs_dir'] / f"{env_dir}" / f"{self._env_id.split('-')[1]}.csv"
        self._replay_path = Path(tuning_folder + '/data/replay/replay.csv')
        # self._result_path = Path(tuning_folder + '/output/result/')
        self._result_path = Path(tuning_folder + '/output/results/offline/' + f"iSafeRM-{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}-{self._env_id}")

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

        # self._result_file = Path(tuning_folder + '/output/result') / f"_results_summary.csv"
        # results_summary_file.to_csv(self._result_file)

        self._act_dim = len(self._config_dict)
        self._obs_dim = len(self._key_services) * 2 + len(self._workload) + 1 + 1
        self._task_id = 0
        random.seed(123)
        self._action_space = Box(low=-1.0, high=1.0, shape=(self._act_dim,), dtype=np.float32)
        self._observation_space = Box(low=0.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32)
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
        self._repetition = 1
        self._threshold = 0.99
        self._replay = False
        self._last_safe_obs = []
        self._last_safe_conf_dict = {}
        self._best_value = float('inf')
        # self._init_env()

    def record_optimal_value(self,value):
        # latency
        if self._best_value > value:
            self._best_value = value
            copy_and_rename_file(self._conf_path, Path(tuning_folder + f'/output/results/offline/'), 'best.yml')

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

    def compute_similarity(self, curr_conf_vector):
        replay_csv = pd.read_csv(self._replay_path)
        if replay_csv.shape[0] == 0:
            return False, 0, 0, []
        # conf_vector = result_csv['conf'].tolist()
        conf_vector_list = []
        for index, row in replay_csv.iterrows():
            conf = []
            tmp = []
            obs = []
            # for i in row['observation'].split(';'):
            for i in row['conf'].split(';'):
                conf.append(float(i))
            for i in row['observation'].split(';'):
                # for i in row['conf'].split(';'):
                obs.append(float(i))
            tmp.append(conf)
            tmp.append(float(row['latency']))
            tmp.append(float(row['resource']))
            tmp.append(obs)
            conf_vector_list.append(tmp)
        # similarity = []
        max_similarity = 0
        latency = 0
        resource = self.get_resource()
        observation = None
        for vector in conf_vector_list:
            cosine_similarity = np.dot(curr_conf_vector, vector[0]) / (
                    np.linalg.norm(curr_conf_vector) * np.linalg.norm(vector[0]))
            if cosine_similarity > max_similarity:
                max_similarity = cosine_similarity
                latency = vector[1]
                # resource = vector[2]
                observation = vector[3]

        new_obs = []
        for i in range(2*len(self._key_services)):
            noise = 0
            new_obs.append(observation[i] + noise)
        noise = 0
        latency += noise
        new_obs += [resource] + self._workload + [latency]
        latency = round(latency, 2)
        new_obs = [round(i, 2) for i in new_obs]
        return True, latency, resource, new_obs

    def get_conf_vector(self):
        # 得到每次配置的值，用于相似度计算
        conf_vector = []
        for k, v in self._config_dict.items():
            conf_vector.append(v['value'])
        return conf_vector

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
        latency_ratio = round(self._pre_latency/self._curr_latency,2)
        performance = latency_ratio if latency_ratio >= 1 else 0
        return performance

    def get_cost(self):
        # 默认预算是100，当延迟第一次超过100的时候才开始消耗安全代价
        if self._curr_latency <= self._slo - 100:
            return 0
        if self._cost_flag:
            self._cost_flag = False
            return round(self._curr_latency - 100, 2)
        else:
            cost = 0 if self._curr_latency < self._pre_latency else self._curr_latency - self._pre_latency
            return round(cost, 2)

    def get_reward(self):
        reward = min(self._lambda1 * self.get_score(), 2) + min(self._lambda2 * self.get_performance(),1)
        reward = round(reward, 2)
        return reward

    def run_config(self):
        cost,reward = -1,-1
        terminated = False
        truncated = False
        info = {}
        # 相似度计算，减少开销
        curr_conf_vector = self.get_conf_vector()
        self._replay, latency, resource, observation = self.compute_similarity(curr_conf_vector)
        if latency < self._slo * 0.5:
            self._last_safe_obs = observation
            self._last_safe_conf_dict = self._config_dict
        result_temp = {}
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
        # 外部
        # result_data.to_csv(self._result_file, mode='a', index=False, header=False)

        # result_data.to_csv(self._replay_path, mode='a', index=False, header=False)
        self._task_id += 1
        self._pre_latency = latency
        self._pre_resource = resource
        self.record_optimal_value(latency)
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
        action = [x for x in action.cpu().numpy()]
        curr_conf = self.get_curr_conf_1(action)
        self._conf_path = self._result_path_conf / f"{self._task_id}_conf.yml"
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
        else:
            obs = self._last_safe_obs
            self._config_dict = self._last_safe_conf_dict
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
        print("======================结束了=======================")
        pass

    def get_benchmark(self):
        return MicroservicesBenchmark(self.benchmark)

