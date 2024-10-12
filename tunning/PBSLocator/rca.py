import networkx as nx
from dowhy import gcm
import pandas as pd
from scipy.stats import halfnorm
import numpy as np

def get_top_services(causal_scores):
    spans = sorted(causal_scores.items(),key=lambda x: x[1],reverse=True)
    top_spans = [ v[0] for v in spans if v[1]>0]
    top_services = []
    for span in top_spans:
        str_list = span.split('_')
        service_name = ''
        if 'Redis' in str_list[1]:
            service_name = str_list[0].split('service')[0]+'redis'
        elif 'Mongo' in str_list[1]:
            service_name = str_list[0].split('service')[0]+'mongodb'
        elif 'Memcached' in str_list[1]:
            service_name = str_list[0].split('service')[0]+'memcached'
        else:
            service_name = str_list[0]
        if service_name not in top_services:
            top_services.append(service_name)
    return top_services


def compute_causal_services(ms_csv,ms_causal_graph,slo=200000):
    ms_data = pd.read_csv(ms_csv)
    ms_data['traceLatency']
    ms_normal = ms_data[ms_data['traceLatency'] < slo]
    ms_abnormal = ms_data[ms_data['traceLatency'] > slo]
    causal_graph = nx.read_gml(ms_causal_graph)
    target_node = [node for node, out_degree in causal_graph.out_degree() if out_degree == 0][0]
    causal_model = gcm.StructuralCausalModel(causal_graph)

    for node in causal_graph.nodes:
        if len(list(causal_graph.predecessors(node))) > 0:
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
        else:
            causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(halfnorm))
    median_attribs, uncertainty_attribs = gcm.confidence_intervals(
    lambda : gcm.distribution_change(causal_model,
                                     ms_normal.sample(frac=0.6),
                                     ms_abnormal.sample(frac=0.6),
                                     target_node,
                                     difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x)),num_bootstrap_resamples = 10)
    return get_top_services(median_attribs)


def compute_causal_services_dict(ms_csv,ms_causal_graph,slo=200000):
    ms_data = pd.read_csv(ms_csv)
    ms_data['traceLatency']
    ms_normal = ms_data[ms_data['traceLatency'] < slo]
    ms_abnormal = ms_data[ms_data['traceLatency'] > slo]
    causal_graph = nx.read_gml(ms_causal_graph)
    target_node = [node for node, out_degree in causal_graph.out_degree() if out_degree == 0][0]
    causal_model = gcm.StructuralCausalModel(causal_graph)

    for node in causal_graph.nodes:
        if len(list(causal_graph.predecessors(node))) > 0:
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
        else:
            causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(halfnorm))
    median_attribs, uncertainty_attribs = gcm.confidence_intervals(
    lambda : gcm.distribution_change(causal_model,
                                     ms_normal.sample(frac=0.6),
                                     ms_abnormal.sample(frac=0.6),
                                     target_node,
                                     difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x)),num_bootstrap_resamples = 10)

    spans = sorted(median_attribs.items(), key=lambda x: x[1], reverse=True)
    services_dict = {}
    for span in spans:
        str_list = span[0].split('_')
        service_name = ''
        if 'Redis' in str_list[1] and ("Set" in str_list[1] or "Update" in str_list[1]):
            service_name = str_list[0].split('service')[0] + 'redis_write'
        elif 'Redis' in str_list[1] and "Get" in str_list[1]:
            service_name = str_list[0].split('service')[0] + 'redis_read'
        elif 'Mongo' in str_list[1] and "Insert" in str_list[1]:
            service_name = str_list[0].split('service')[0] + 'mongodb_write'
        elif 'Mongo' in str_list[1]:
            service_name = str_list[0].split('service')[0] + 'mongodb_read'
        elif 'Memcached' in str_list[1]:
            service_name = str_list[0].split('service')[0] + 'memcached_read'
        else:
            service_name = str_list[0]
        if service_name not in services_dict.keys() and span[1] > 0:
            services_dict[service_name] = span[1]


    return services_dict