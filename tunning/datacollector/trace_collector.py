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
# import traceProcessor as t_processor
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
        # self.pool = multiprocessing.Pool(max_processes)
        self.task_id = 0
        self.span_data_csv = f"{self.resultPath}/{self.task_id}_span_data.csv"
        self.trace_data_csv = f"{self.resultPath}/{self.task_id}_trace_data.csv"
        self.ms_data_csv = ''

    # def close_pool(self):
    #     self.pool.close()

    def set_task_id(self, task_id):
        self.task_id = task_id

    def collect_trace_data(self, limit, start_time=1690439926.282558, operation=None, no_nginx=False,
                           no_frontend=False):
        time.sleep(5)
        # Generate fetching url
        request_data = {
            "start": int(start_time * 1000000),
            # "end": int((start_time + self.duration * 1000) * 1000000),
            "limit": limit,
            "service": self.entryPoint,
            "tags": '{"http.status_code":"200"}',
            # "tags": status_code
        }
        if operation is not None:
            request_data["operation"] = operation
        req = requests.get(f"{self.jaegerHost}/api/traces", params=request_data)
        res = json.loads(req.content)["data"]
        if len(res) == 0:
            my_logger.error(f"No traces are fetched!")
            # self.write_log(f"No traces are fetched!", "error")
            return False, None, None
        else:
            my_logger.info(f"Number of traces: {len(res)}")
            # self.write_log(f"Number of traces: {len(res)}")
        # Record process id and microservice name mapping of all traces
        # Original headers: traceID, processes.p1.serviceName, processes.p2.serviceName, ...
        # Processed headers: traceId, p1, p2, ...
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

        # 从一个名为spans_data的数据中筛选出具有非空父引用的子集 ~取反
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
        # A merged data frame that build relationship of different spans
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

        # Map each span's processId to its microservice name
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
        # print(span_data[span_data['traceTime'] == span_data['traceTime'].max()]['traceId'].values)
        # trace_id = span_data['traceId'].unique()
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
            # print(temp)
            ms_graph[row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                'childOperation']] = temp
        ms_graph['span_name'] = span_name
        save_dict(ms_graph, f"{self.resultPath}/{self.task_id}_{request_name}")



    def span2trace(self, request_name):
        span_data = pd.read_csv(self.span_data_csv)
        trace_data = pd.read_csv(self.trace_data_csv)
        trace_id = span_data['traceId'].unique()
        ms_graph = load_dict(f"{self.resultPath}/{self.task_id}_{request_name}")
        # print(ms_graph)
        columns_name = ['traceId', 'traceLatency'] + list(ms_graph['span_name'].values())
        trace_dict = pd.DataFrame(columns=columns_name)
        # print(trace_dict)
        for traceId in trace_id:
            selected_trace = span_data[span_data['traceId'] == traceId]
            ms_duration = {}
            ms_duration['traceId'] = traceId
            ms_duration['traceLatency'] = trace_data[trace_data['traceId'] == traceId]['traceLatency'].tolist()[0]
            # 判断这条trace是否符合之前选中的依赖图，不符合就过滤他的数据
            flag = True
            for index, row in selected_trace.iterrows():
                if row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                    'childOperation'] not in ms_graph:
                    flag = False
                    break
                # my_logger.info(ms_graph[
                #     row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                #         'childOperation']]['source'] + " : " + str(row['parentDuration']))
                # my_logger.info(ms_graph[
                #    row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                #        'childOperation']]['target'] + " : " + str(row['childDuration']))
                ms_duration[ms_graph[
                    row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                        'childOperation']]['source']] = row['parentDuration']
                ms_duration[ms_graph[
                    row['parentMS'] + '_' + row['parentOperation'] + '_to_' + row['childMS'] + '_' + row[
                        'childOperation']]['target']] = row['childDuration']
            if flag and len(ms_duration) == len(columns_name):
                # print(ms_duration)
                trace_dict = trace_dict._append(ms_duration, ignore_index=True)
                # ms_duration = pd.DataFrame(ms_duration)
                # trace_dict = pd.concat([trace_dict,ms_duration])
        self.ms_data_csv = f'{self.resultPath}/{self.task_id}_{request_name}_ms_data.csv'
        trace_dict.to_csv(f'{self.resultPath}/{self.task_id}_{request_name}_ms_data.csv', index=False)

    def get_ms_data_csv(self):
        # print(os.listdir(self.resultPath))
        # print(self.ms_data_csv)
        filename = os.path.basename(self.ms_data_csv)
        # print(filename)
        if filename in os.listdir(self.resultPath):
            return pd.read_csv(self.ms_data_csv)
        else:
            # print('None')
            return None

    def get_ms_causal_graph(self, request_name):
        ms_graph = load_dict(f"{self.resultPath}/{self.task_id}_{request_name}")
        source_target = []
        for key, value in ms_graph.items():
            if key != 'span_name':
                source_target.append((value['target'], value['source']))
        # print(source_target)
        causal_graph = nx.DiGraph()
        causal_graph.add_nodes_from(list(ms_graph['span_name'].values()))
        causal_graph.add_edges_from(source_target)
        nx.write_gml(causal_graph, f"{self.resultPath}/{self.task_id}_{request_name}_causal_graph.gml")
        # gcm.util.plot(causal_graph, figure_size=[20, 20])

    def collect_ms_data(self, start_time):
        self.span_data_csv = f"{self.resultPath}/{self.task_id}_span_data.csv"
        self.trace_data_csv = f"{self.resultPath}/{self.task_id}_trace_data.csv"
        for request_name, operation in self.operations.items():
            done = self.get_trace_csv(start_time=start_time, operation=operation)
            # if self.task_id == 0:
            if done:
                self.get_ms_graph(request_name=request_name)
                self.get_ms_causal_graph(request_name=request_name)
                self.span2trace(request_name=request_name)


def process_span_data(span_data: pd.DataFrame):
    db_data = pd.DataFrame()
    # for key_word in ["Mongo", "Redis", "Mmc", "Mem"]:
    #     dbs = span_data["childOperation"].str.contains(key_word)
    #     db_layer = span_data.loc[dbs]
    #     db_layer["childMS"] = key_word
    #     db_layer["childPod"] = key_word
    #     span_data = pd.concat([span_data.loc[~dbs], db_layer])
    #     db_data = pd.concat([
    #         db_data,
    #         db_layer[[
    #             "parentMS",
    #             "parentOperation",
    #             "childMS",
    #             "childOperation",
    #             "childDuration"
    #         ]]
    #     ])
    # Calculate exact parent duration
    span_data = t_processor.exact_parent_duration(span_data, "merge")
    p95_df = t_processor.decouple_parent_and_child(span_data, 0.95)
    p50_df = t_processor.decouple_parent_and_child(span_data, 0.5)
    return p50_df.rename(columns={"latency": "median"}).merge(p95_df, on=["microservice", "pod"]), db_data


# _, span_data, trace_data = collect_trace_data(1500, operation="/wrk2-api/post/compose")
# pod_latency, _ = process_span_data(span_data)
# ms_latency = pod_latency.groupby("microservice").mean().reset_index()
# span_data.to_csv("/home/dev/tuning_microservice/tunning/datacollector/data/span_data.csv", index=False)
# trace_data.to_csv("/home/dev/tuning_microservice/tunning/datacollector/data/trace_data.csv", index=False)
#
# print(ms_latency)
# print(pod_latency)

data_collector = TraceCollector(
    bench_name='social_network',
    resultPath='/home/lilong/tuning_microservice/tunning/datacollector/traces_data',
)
# data_collector.get_trace_csv(start_time=1700892462.52159,operation="/wrk2-api/post/compose")
# data_collector.get_ms_graph('ComposePost')
# data_collector.span2trace('ComposePost')
data_collector.get_ms_causal_graph('ComposePost')
