# iSafeRM
iSafeRM: An Interpretable and Safe Resource Management Framework for Microservice Systems with SLO Guarantees. 

<font style="color:rgb(31, 35, 40);">This repository contains codes for a research paper that was submitted for publication at the 2024 WWW.</font>



# Introduction
Considering the dynamic workloads and the complex interactions among different services, it is difficult to figure out the accurate relationship between the user-facing quality of service and the available computing resource of each interior microservice. As a result, ISPs usually have to assign excessive resources to each microservice to avoid the possible end-to-end service-level objective (SLO) violations, which leads to the unnecessary waste of computing resources and consequently, additional cloud service expenses.

We present an interpretable and safe resource management framework called **iSafeRM** for microservice-based systems, which is able to automatically locate the bottleneck services for current SLO violation and conduct a safe online configuration of both computing resources and performance-critical parameters. Specifically, iSafeRM consists of three major modules: PBSLocator, PCPIdenter and SafeConfigor. 



# Environment
+ Hardware requirement
    - A cluster with 3 CPU servers or more
    - CPU: Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
    - DRAM: 64 GB for each server or more
+ Software requirement
    - Ubuntu 20.04 LTS 
    - Docker 20.10.7 
    - Python 3.9
    - YCSB
    - Wrk
    - benchmark: DeathStarBench

# PBSLocator
When a SLO violation is detected for current user request, PBSLocator extracts the service call relationships from the latest tracing data and inverts the pointing relationship to obtain the corresponding request-level causal graph. After that, it leverages the functional causal model to perform efficient counterfactual inference to locate the bottleneck services.



1. You can find the code in the tuning/PBSLocator directory and run it as follows

```shell
cd iSafeRM
make run-run-exp-find-key-services-sn
```

2. Or you can start by collecting microservice traces and then use rca.py to locate the bottleneck service.

```python
from tunning.datacollector.trace_collector import TraceCollector
data_collector = TraceCollector(
            bench_name=self._benchmark,
            resultPath=self._result_path / 'trace',
        )
data_collector.collect_ms_data(start_time=start_time)
ms_data = data_collector.get_ms_data_csv()

# By parsing microservice traces, you get a latency csv file and a causal graph

from tunning.PBSLocator.rca import compute_causal_services

top_services = compute_causal_services(csv,graph)

```



# PCPIdenter
PCPIdenter is responsible for identifying the inherent performance-critical parameters of complex bottleneck services such as databases under current workload characteristics. 

1. You can find the code in the tuning/PCPIdenter directory and run with jupyter
2. In the folder we have collected some offline data for identifying the key parameters of the database. You can collect the data in your own environment in the following ways

```shell
cd iSafeRM
make run-redis
```

3. Note that collecting the data requires deploying some monitoring software, such as Prometheus, exporter. You can deploy it as follows

```shell
cd iSafeRM
make run-prometheus
```

More details are put in the Makefile and the relevant deployment files are in the deploy folder.

# SafeConfigor
SafeConfigor explores the configuration space consists of the inherent performance-critical parameters as well as the instance specification and instance number in an online manner.

1. You can find the code in the tuning/benchenvs directory and run it as follows

```shell
cd iSafeRM
make run-rl-sn
```

2. This module can be used on its own, as long as you have a csv configuration file with microservice parameters.
    - Csv file like sn_5causal_3parameters.csv in iSafeRM/configs
    - The environment code for RL is in microservices_env.py



