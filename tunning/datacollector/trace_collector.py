from datetime import datetime
import multiprocessing
import os
from time import sleep
import traceback
from typing import Dict, List, Set
import json
import re
import pandas as pd
import requests
from tunning.datacollector import traceProcessor as t_processor
import time
import pickle
import networkx as nx
from dowhy import gcm
from tunning.constants import BENCHMARK_CONFIG, GLOBAL_CONFIG
from tunning.utils import my_logger,save_dict,load_dict


class TraceCollector:
    def __init__(
            self,
            bench_name,
            resultPath,
            prometheusHost=GLOBAL_CONFIG['prometheus_host'],

            duration=120,
            max_traces=5000,
            mointorInterval=1,
            max_processes=3
    ):
        """Initilizing an offline profiling data collector

                Args:
                    namespace (str): Namespace
                    duration (int): Duration of each round of test
                    jaegerHost (str): Address to access jaeger, e.g. http://localhost:16686
                    entryPoint (str): The entry point service of the test
                    prometheusHost (str): Address to access Prometheus, similar to jaegerHost
                    mointorInterval (str): Prometheus monitor interval
                    nodes (list[str]): Nodes that will run test
                    dataPath (str): Where to store merged data
                    dataName (str): The name of the merged data
                    cpuInterCpuSize (float | int): CPU limitation for CPU interference pod
                    memoryInterMemorySize (str): Memory limiataion for memory interference pod
                """
        self.bench_name = bench_name
        self.duration = duration
        self.jaegerHost = BENCHMARK_CONFIG['microservices'][self.bench_name]['jaegerHost']
        self.entryPoint = BENCHMARK_CONFIG['microservices'][self.bench_name]['entry_point']
        self.prometheusHost = prometheusHost
        self.monitorInterval = mointorInterval
        self.max_traces = max_traces
        self.resultPath = resultPath
        self.operations = BENCHMARK_CONFIG['microservices'][self.bench_name]['operations']
        self.max_processes = max_processes
        self.task_id = 0
        self.span_data_csv = f"{self.resultPath}/{self.task_id}_span_data.csv"
        self.trace_data_csv = f"{self.resultPath}/{self.task_id}_trace_data.csv"
        self.ms_data_csv = ''

    def set_task_id(self, task_id):
        self.task_id = task_id

    def collect_trace_data(self, limit, start_time=1690439926.282558, operation=None, no_nginx=False,
                           no_frontend=False):
        time.sleep(5)
        request_data = {
            "start": int(start_time * 1000000),
            "limit": limit,
            "service": self.entryPoint,
            "tags": '{"http.status_code":"200"}',
        }
        if operation is not None:
            request_data["operation"] = operation
        req = requests.get(f"{self.jaegerHost}/api/traces", params=request_data)
        res = json.loads(req.content)["data"]
        if len(res) == 0:
            my_logger.error(f"No traces are fetched!")
            return False, None, None
        else:
            my_logger.info(f"Number of traces: {len(res)}")
        service_id_mapping = (
            pd.json_normalize(res)
            .filter(regex="serviceName|traceID|tags")
            .rename(
                columns=lambda x: re.sub(
                    r"processes\.(.*)\.serviceName|processes\.(.*)\.tags",
                    lambda match_obj: match_obj.group(1)
                    if match_obj.group(1)
                    else f"{match_obj.group(2)}Pod",
                    x,
                )
            )
            .rename(columns={"traceID": "traceId"})
        )
        service_id_mapping = (
            service_id_mapping.filter(regex=".*Pod")
            .applymap(
                lambda x: [v["value"] for v in x if v["key"] == "hostname"][0]
                if isinstance(x, list)
                else ""
            )
            .combine_first(service_id_mapping)
        )
        spans_data = pd.json_normalize(res, record_path="spans")[
            [
                "traceID",
                "spanID",
                "operationName",
                "duration",
                "processID",
                "references",
                "startTime",
            ]
        ]

        spans_with_parent = spans_data[~(spans_data["references"].astype(str) == "[]")]
        root_spans = spans_data[(spans_data["references"].astype(str) == "[]")]
        root_spans = root_spans.rename(
            columns={
                "traceID": "traceId",
                "startTime": "traceTime",
                "duration": "traceLatency"
            }
        )[["traceId", "traceTime", "traceLatency"]]
        spans_with_parent = spans_with_parent.copy()
        spans_with_parent.loc[:, "parentId"] = spans_with_parent["references"].map(
            lambda x: x[0]["spanID"]
        )
        temp_parent_spans = spans_data[
            ["traceID", "spanID", "operationName", "duration", "processID"]
        ].rename(
            columns={
                "spanID": "parentId",
                "processID": "parentProcessId",
                "operationName": "parentOperation",
                "duration": "parentDuration",
                "traceID": "traceId",
            }
        )
        temp_children_spans = spans_with_parent[
            [
                "operationName",
                "duration",
                "parentId",
                "traceID",
                "spanID",
                "processID",
                "startTime",
            ]
        ].rename(
            columns={
                "spanID": "childId",
                "processID": "childProcessId",
                "operationName": "childOperation",
                "duration": "childDuration",
                "traceID": "traceId",
            }
        )
        merged_df = pd.merge(
            temp_parent_spans, temp_children_spans, on=["parentId", "traceId"]
        )

        merged_df = merged_df[
            [
                "traceId",
                "childOperation",
                "childDuration",
                "parentOperation",
                "parentDuration",
                "parentId",
                "childId",
                "parentProcessId",
                "childProcessId",
                "startTime",
            ]
        ]

        merged_df = merged_df.merge(service_id_mapping, on="traceId")
        merged_df = merged_df.merge(root_spans, on="traceId")
        merged_df = merged_df.assign(
            childMS=merged_df.apply(lambda x: x[x["childProcessId"]], axis=1),
            childPod=merged_df.apply(lambda x: x[f"{str(x['childProcessId'])}Pod"], axis=1),
            parentMS=merged_df.apply(lambda x: x[x["parentProcessId"]], axis=1),
            parentPod=merged_df.apply(
                lambda x: x[f"{str(x['parentProcessId'])}Pod"], axis=1
            ),
            endTime=merged_df["startTime"] + merged_df["childDuration"],
        )
        merged_df = merged_df[
            [
                "traceId",
                "traceTime",
                "startTime",
                "endTime",
                "parentId",
                "childId",
                "childOperation",
                "parentOperation",
                "childMS",
                "childPod",
                "parentMS",
                "parentPod",
                "parentDuration",
                "childDuration",
            ]
        ]
        if no_nginx:
            return True, merged_df, t_processor.no_entrance_trace_duration(merged_df, "nginx")
        elif no_frontend:
            return True, merged_df, t_processor.no_entrance_trace_duration(merged_df, "frontend")
        else:
            return True, merged_df, root_spans

    def get_trace_csv(self, start_time, operation):
        done, span_data, trace_data = self.collect_trace_data(self.max_traces, start_time=start_time,
                                                              operation=operation)
        if done:
            span_data.to_csv(self.span_data_csv, index=False)
            trace_data.to_csv(self.trace_data_csv, index=False)
            return True
        else:
            return False

    def get_ms_graph(self, request_name):
        span_data = pd.read_csv(self.span_data_csv)
        trace_id = span_data[span_data['traceTime'] == span_data['traceTime'].max()]['traceId'].values[0]
        selected_trace = span_data[span_data['traceId'] == trace_id]
        ms_graph = {}
        span_id = []
        span_name = {}
        name_count = 1
        for index, row in selected_trace.iterrows():
            temp = {}
            if row['parentId'] in span_id:
                temp['source'] = span_name[row['parentId']]
            else:
                temp['source'] = f"{row['parentMS']}_{row['parentOperation']}_{name_count}"
                span_id.append(row['parentId'])
                span_name[row['parentId']] = temp['source']
                name_count += 1
            if row['childId'] in span_id:
                temp['target'] = span_name[row['childId']]
            else:
                temp['target'] = f"{row['childMS']}_{row['childOperation']}_{name_count}"
                span_id.append(row['childId'])
                span_name[row['childId']] = temp['target']
                name_count += 1
            ms_graph[row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                'childOperation']] = temp
        ms_graph['span_name'] = span_name
        save_dict(ms_graph, f"{self.resultPath}/{self.task_id}_{request_name}")



    def span2trace(self, request_name):
        span_data = pd.read_csv(self.span_data_csv)
        trace_data = pd.read_csv(self.trace_data_csv)
        trace_id = span_data['traceId'].unique()
        ms_graph = load_dict(f"{self.resultPath}/{self.task_id}_{request_name}")
        columns_name = ['traceId', 'traceLatency'] + list(ms_graph['span_name'].values())
        trace_dict = pd.DataFrame(columns=columns_name)
        for traceId in trace_id:
            selected_trace = span_data[span_data['traceId'] == traceId]
            ms_duration = {}
            ms_duration['traceId'] = traceId
            ms_duration['traceLatency'] = trace_data[trace_data['traceId'] == traceId]['traceLatency'].tolist()[0]
            flag = True
            for index, row in selected_trace.iterrows():
                if row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                    'childOperation'] not in ms_graph:
                    flag = False
                    break
                ms_duration[ms_graph[
                    row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                        'childOperation']]['source']] = row['parentDuration']
                ms_duration[ms_graph[
                    row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                        'childOperation']]['target']] = row['childDuration']
            if flag and len(ms_duration) == len(columns_name):
                trace_dict = trace_dict._append(ms_duration, ignore_index=True)
        self.ms_data_csv = f'{self.resultPath}/{self.task_id}_{request_name}_ms_data.csv'
        trace_dict.to_csv(f'{self.resultPath}/{self.task_id}_{request_name}_ms_data.csv', index=False)

    def get_ms_data_csv(self):
        filename = os.path.basename(self.ms_data_csv)
        if filename in os.listdir(self.resultPath):
            return pd.read_csv(self.ms_data_csv)
        else:
            return None

    def get_ms_causal_graph(self, request_name):
        ms_graph = load_dict(f"{self.resultPath}/{self.task_id}_{request_name}")
        source_target = []
        for key, value in ms_graph.items():
            if key != 'span_name':
                source_target.append((value['target'], value['source']))
        causal_graph = nx.DiGraph()
        causal_graph.add_nodes_from(list(ms_graph['span_name'].values()))
        causal_graph.add_edges_from(source_target)
        nx.write_gml(causal_graph, f"{self.resultPath}/{self.task_id}_{request_name}_causal_graph.gml")

    def collect_ms_data(self, start_time):
        self.span_data_csv = f"{self.resultPath}/{self.task_id}_span_data.csv"
        self.trace_data_csv = f"{self.resultPath}/{self.task_id}_trace_data.csv"
        for request_name, operation in self.operations.items():
            done = self.get_trace_csv(start_time=start_time, operation=operation)
            if done:
                self.get_ms_graph(request_name=request_name)
                self.get_ms_causal_graph(request_name=request_name)
                self.span2trace(request_name=request_name)


def process_span_data(span_data: pd.DataFrame):
    db_data = pd.DataFrame()
    span_data = t_processor.exact_parent_duration(span_data, "merge")
    p95_df = t_processor.decouple_parent_and_child(span_data, 0.95)
    p50_df = t_processor.decouple_parent_and_child(span_data, 0.5)
    return p50_df.rename(columns={"latency": "median"}).merge(p95_df, on=["microservice", "pod"]), db_data


data_collector = TraceCollector(
    bench_name='social_network',
    resultPath='/home/XXXX-1/tuning_microservice/tunning/datacollector/traces_data',
)
data_collector.get_ms_causal_graph('ComposePost')
