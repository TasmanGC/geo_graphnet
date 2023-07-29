import dgl
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def load_homo_graph(node_file : Path, edge_file : Path) -> dgl.graph:

    edges = pd.read_csv(edge_file, index_col='Unnamed: 0', dtype={'U':int,'V':int})
    nodes = pd.read_csv(node_file, index_col='Unnamed: 0')
    
    largest_node_id = edges.max().max()
    num_nodes = len(nodes)-1
    if largest_node_id != num_nodes:
        raise AssertionError('Edge and node ids do not align.')
            
    u = torch.from_numpy(edges['U'].values)
    v = torch.from_numpy(edges['V'].values)
        
    graph = dgl.graph(('coo',(u,v)))
    
    features = nodes.to_dict('list')

    for k,v in features.items():
        if any(isinstance(val, str) for val in v):
            mapping = {k:i for i,k in enumerate(list(set(v)))}
            mapped = np.array([mapping[x] for x in v],dtype=int)
            features[k] = torch.as_tensor(mapped,dtype=torch.float32)
        else:
            features[k] = torch.as_tensor(np.array(v),dtype=torch.float32)
        
    graph.ndata.update(features)
    
    return(graph)