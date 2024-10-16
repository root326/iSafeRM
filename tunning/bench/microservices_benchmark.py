import os
import re
import subprocess
from pathlib import Path
from typing import Tuple
from tunning.utils import get_resource
from tunning.bench.abstract import AbstractBench
from tunning.bench.exceptions import BenchException
from tunning.constants import BENCHMARK_CONFIG,GLOBAL_CONFIG


class MicroservicesBenchmark:
    def __init__(self, benchmark):
        self._benchmark = benchmark
        self.playbook_deploy = BENCHMARK_CONFIG['microservices'][self._benchmark]['playbook_dir'] / "deploy.yml"
        self.playbook_deploy_key_services = BENCHMARK_CONFIG['microservices'][self._benchmark][
                                                'playbook_dir'] / "deploy_key_services.yml"
        self.playbook_clean = BENCHMARK_CONFIG['microservices'][self._benchmark]['playbook_dir'] / "clean.yml"
        self.playbook_clean_key_services = BENCHMARK_CONFIG['microservices'][self._benchmark][
                                               'playbook_dir'] / "clean_key_services.yml"
        self.entry_ip = BENCHMARK_CONFIG['microservices'][self._benchmark]['entry_ip']
        self.out_put = Path(BENCHMARK_CONFIG['microservices'][self._benchmark]['output'])
        self.bench_log = BENCHMARK_CONFIG['microservices'][self._benchmark]['bench_log']
        self.workload_sh = BENCHMARK_CONFIG['microservices'][self._benchmark]['workload_sh']
        self.p_latency = BENCHMARK_CONFIG['microservices'][self._benchmark]['p_latency']
        self.workload = []
        self.project_name = GLOBAL_CONFIG["project_name"]

    def run(self, conf_path: Path, workload, task_id):
        conf_path = str(conf_path)
        playbook_deploy = ''
        playbook_deploy = str(self.playbook_deploy)

        workload1 = int(workload[0])
        cmd = [
            "ansible-playbook",
            playbook_deploy,
            f"-e conf_path={conf_path} entry_ip={self.entry_ip} workload1={workload1} project_name={self.project_name}",
        ]
        try:
            subprocess.run(
                cmd, check=True
            )
        except subprocess.CalledProcessError as e:
            raise BenchException(" benchmark deploy failed") from e

    def cleanup(self, task_id):
        playbook_clean = str(self.playbook_clean)
        cmd = [
            "ansible-playbook",
            playbook_clean,
            f"-e project_name={self.project_name}"
        ]
        try:
            subprocess.run(
                cmd, check=True
            )
        except subprocess.CalledProcessError as e:
            raise BenchException(" benchmark clean failed") from e

    def parse_result(self, conf_path: Path):
        latency = self._parse_latency()
        resource = get_resource(conf_path)
        return latency, resource

    def _parse_latency(self) -> float:
        log = self.out_put / self.bench_log
        p95_re = re.compile(fr"(\d+.+)\s+0\.{self.p_latency}\s+(\d+.+)")
        content = log.read_text()
        match = p95_re.search(content)
        if match == None:
            return 9999
        latency = float(match.group(1).split()[0])

        return latency

    def run_workload(self, workload):

        os.system(
            f"bash {self.workload_sh} {self.entry_ip} {workload} ")