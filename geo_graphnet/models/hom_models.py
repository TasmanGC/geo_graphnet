from dgl.nn.pytorch import GATConv, GraphConv
from custom_layers import WGraphConv
import torch.nn.functional as F
import torch.nn as nn
import dgl
from geo_graphnet.data_handling.config_handlers import GeoGraphConfig

def initialise_model(input:GeoGraphConfig):
    if input.type != 'model':
        raise ValueError('Received non model config.')
    else:
        mod_type = GeoGraphConfig.model_type
        
        if mod_type=='gcn':
            in_feats = GeoGraphConfig.in_feats
            hid_feats = GeoGraphConfig.hid_feats
            num_classes = GeoGraphConfig.num_classes
            return(GCN(in_feats,hid_feats,num_classes))    
        
        if mod_type=='gat':
            in_feats = GeoGraphConfig.in_feats
            hid_feats = GeoGraphConfig.hid_feats
            num_classes = GeoGraphConfig.num_classes
            num_heads = GeoGraphConfig.num_heads
            return(GAT(in_feats, hid_feats, num_classes, num_heads))
        
        if mod_type=='wgcn':
            in_feats = GeoGraphConfig.in_feats
            hid_feats = GeoGraphConfig.hid_feats
            num_classes = GeoGraphConfig.num_classes
            return(WGCN(in_feats, hid_feats, num_classes))
        
        else:
            raise ValueError(f'Behaviour Undefined for model type {mod_type} was expecting one of [gcn,gat,wgcn]')
    
class GCN(nn.Module):
    def __init__(self, in_feats:int, hid_feats:int, num_classes:int):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hid_feats)
        self.conv2 = GraphConv(hid_feats, num_classes)

    def forward(self, g:dgl.graph, in_feat):# TODO these forward methods should always just take a graph in
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.softmax(h,dim=1)
        h = F.log_softmax(h, dim=1) # NOTE this is the currently implemnted behaviour but is dumb might need to fix
        return h

class WGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, num_classes):
        super(WGCN, self).__init__()
        self.conv1 = WGraphConv(in_feats, hid_feats)
        self.conv2 = WGraphConv(hid_feats, num_classes)

    def forward(self, g, in_feat): # TODO these forward methods should always just take a graph in
        edge_weight = 
        # Apply graph convolution and activation.
        h = self.conv1(g, in_feat, edge_weight)
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight)
        h = F.softmax(h,dim=1)
        h = F.log_softmax(h, dim=1) # NOTE this is the currently implemnted behaviour but is dumb might need to fix
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, hid_feat, num_classes, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_feats, hid_feat, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hid_feat*num_heads, num_classes, 1)

    def forward(self, g, in_feat): # TODO these forward methods should always just take a graph in
        h, a_1 = self.layer1(g, in_feat, get_attention=True)
        # Concat last 2 dim (num_heads * out_dim)
        h = h.view(-1, h.size(1) * h.size(2)) # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        h = F.elu(h)
        h,a_2 = self.layer2(g,h,get_attention=True)
        # Sueeze the head dim as it's = 1 
        h = h.squeeze() # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        h = F.softmax(h,dim=1)
        h = F.log_softmax(h, dim=1) # NOTE this is the currently implemnted behaviour but is dumb might need to fix
        # NOTE disabling return of attention weights for now
        return(h)#,[a_1,a_2])