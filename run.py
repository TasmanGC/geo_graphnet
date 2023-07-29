import json
import torch
import numpy as np

from geo_graphnet.methods.hom_methods import semi_supervised_pred
from geo_graphnet.data_handling.config_handlers import GeoGraphConfig
from geo_graphnet.data_handling.graph_loading import load_homo_graph

graph = load_homo_graph(node_file=r'data\homo_graphs\real_nodes.csv',edge_file=r'data\homo_graphs\real_edges_lattice.csv')

graph.ndata['train'] = torch.Tensor([True if x in np.unique(graph.ndata['Line'].numpy())[::2] else False for x in graph.ndata['Line'].numpy()])

# config for the data
with open(r'C:\Users\bogo\Desktop\code\geo_graphnet\configs\data.json', "r") as read_file:
    data = json.load(read_file)
data_config = GeoGraphConfig.from_data_config(data)

#config for the method
with open(r'C:\Users\bogo\Desktop\code\geo_graphnet\configs\method.json', "r") as read_file:
    data = json.load(read_file)
method_config = GeoGraphConfig.from_method_config(data)

#config for the model
with open(r'C:\Users\bogo\Desktop\code\geo_graphnet\configs\model.json', "r") as read_file:
    data = json.load(read_file)
model_config = GeoGraphConfig.from_model_config(data)

semi_supervised_pred(graph, data_config,model_config,method_config)