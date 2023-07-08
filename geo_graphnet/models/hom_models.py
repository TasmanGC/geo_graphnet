from dgl.nn.pytorch import GATConv, GraphConv
from custom_layers import WGraphConv
import torch.nn.functional as F
import torch.nn as nn
import dgl

class GCN(nn.Module):
    def __init__(self, in_feats:int, h_feats:int, num_classes:int=2):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g:dgl.graph, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.softmax(h,dim=1)
        return h

class WGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(WGCN, self).__init__()
        self.conv1 = WGraphConv(in_feats, h_feats)
        self.conv2 = WGraphConv(h_feats, num_classes)

    def forward(self, g, in_feat, edge_weight):
        # Apply graph convolution and activation.
        h = self.conv1(g, in_feat,   edge_weight)
        h = F.relu(h)
        h = self.conv2(g, h,         edge_weight)
        h = F.softmax(h,dim=1)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, hid_feat , out_feats, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_feats, hid_feat, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hid_feat*num_heads, out_feats, 1)

    def forward(self, g, h):
        h,a_1 = self.layer1(g, h, get_attention=True)
        # Concat last 2 dim (num_heads * out_dim)
        h = h.view(-1, h.size(1) * h.size(2)) # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        h = F.elu(h)
        h,a_2 = self.layer2(g,h,get_attention=True)
        # Sueeze the head dim as it's = 1 
        h = h.squeeze() # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        h = F.softmax(h,dim=1)
        return(h,[a_1,a_2])