from geo_graphnet.data_handling.config_handlers import GeoGraphConfig
from geo_graphnet.models.hom_models import initialise_model
from torch import optim
import itertools
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from dgl import graph as dgl_graph
import warnings
from core import calc_accuracy
import numpy as np

def semi_supervised_pred(graph:dgl_graph,
                         data_config:GeoGraphConfig,
                         model_config:GeoGraphConfig, 
                         method_config:GeoGraphConfig):
    
    # determine if working with real features or random embedding
    if len(data_config.value_fields) > 0 and data_config.random_embedding > 0:
        warnings.warn('''Features and random embedding configuration detected. 
                      \n Default to real features.''')
        data_config.random_embedding = 0
    
    # checking that the features that will be selected are the same size the models input shape
    if len(data_config.value_fields) != model_config.in_feats:
        if data_config.random_embedding > 0:
            model_config.in_feats = data_config.random_embedding
        elif len(data_config.value_fields) > 0:
            model_config.in_feats = len(model_config.value_fields)
            warnings.warn(f'''Warning: Number of selected features[{len(data_config.value_fields)}]
                      != model input setting [{model_config.in_feats}]!
                      \n Setting to Number of selected features[{len(data_config.value_fields)}].''')
        
    # ensure the features selected are part of the graph object
    if not set(data_config.value_fields).issubset(list(graph.ndata.keys())):
        raise KeyError("Graph does not contain expected keys.")
    
    # define inputs and trainingselection
    node_embed = nn.Embedding(dgl_graph.num_nodes, model_config.in_feats)
    
    # if we use a real feature set
    if len(data_config.value_fields) > 0:
        real_feats = list(zip(*[graph.ndata[feature] for feature in data_config.value_fields]))
        node_embed.weight = nn.Parameter(tensor(real_feats),requires_grad=True)
        
    train_selection = graph.ndata[data_config.train_role].bool()
    true_labels = graph.ndata[data_config.label_field]
    
    # initialise the model
    model = initialise_model(model_config)

    # loading the ml/learnable params
    optim_obj = getattr(optim, method_config.optimiser)
    parameters = itertools.chain(model.parameters(), node_embed.paramters())
    optimiser = optim_obj(parameters, method_config.lr)
    loss_obj = getattr(F, method_config.loss)
    
    metrics = {}
    metrics['loss'] = []
    metrics['f1'] = []
    metrics['prec'] = []
    metrics['recall'] = []
    
    for _ in range(method_config.epochs):
        
        predictions = model(graph, node_embed.weight)
        loss_val = loss_obj(predictions[train_selection], true_labels[train_selection])
        optimiser.zero_grad()
        loss_val.backward()
        optimiser.step()
        metrics['loss'].append(loss_val.item())
        accuracy_dict = calc_accuracy(np.array(predictions).flatten(), np.array(true_labels).values)
        
        for k, v in accuracy_dict.append():
            metrics[k].append(v)
        
    return(metrics)