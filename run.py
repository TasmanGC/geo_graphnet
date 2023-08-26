import json
import torch
import numpy as np

from geo_graphnet.methods.hom_methods import SemiSupervisedPred, disclaimer
from geo_graphnet.data_handling.config_handlers import GeoGraphConfig
from geo_graphnet.data_handling.graph_loading import load_homo_graph

graph = load_homo_graph(
    node_file=r"data\homo_graphs\real_nodes.csv",
    edge_file=r"data\homo_graphs\real_edges_lattice.csv",
)

graph.ndata["train"] = torch.Tensor(
    [
        True if x in np.unique(graph.ndata["Line"].numpy())[::2] else False
        for x in graph.ndata["Line"].numpy()
    ]
)

# config for the data
with open("configs\data.json", "r") as read_file:
    data = json.load(read_file)
data_config = GeoGraphConfig.from_data_config(data)

# config for the method
with open("configs\method.json", "r") as read_file:
    data = json.load(read_file)
method_config = GeoGraphConfig.from_method_config(data)

# config for the model
with open("configs\model.json", "r") as read_file:
    data = json.load(read_file)
model_config = GeoGraphConfig.from_model_config(data)

print(disclaimer)

method = SemiSupervisedPred(graph, data_config, model_config, method_config)
method.run_exp()

from geo_graphnet.visualisation.visualise_3D import Scene3D

scene = Scene3D(graph, "label")
scene.interactive(plot_cons=True)
