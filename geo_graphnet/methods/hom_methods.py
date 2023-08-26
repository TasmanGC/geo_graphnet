import warnings
warnings.simplefilter("ignore", UserWarning)
from geo_graphnet.data_handling.config_handlers import GeoGraphConfig
from geo_graphnet.models.hom_models import initialise_model
from torch import optim
import itertools
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from dgl import graph as dgl_graph
import warnings
from .core import calc_accuracy
import numpy as np


disclaimer =    """ geo_graphnet performs semi-supervised prediction using GNN on geoscience
                    data. Copyright (C) 2023  Tasman Gillfeather-Clark

                    This program is free software: you can redistribute it and/or modify
                    it under the terms of the GNU General Public License as published by
                    the Free Software Foundation version 3 of the License

                    This program is distributed in the hope that it will be useful,
                    but WITHOUT ANY WARRANTY; without even the implied warranty of
                    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
                    GNU General Public License for more details.

                    You should have received a copy of the GNU General Public License
                    along with this program.  If not, see <https://www.gnu.org/licenses/>.\n"""

class SemiSupervisedPred:
    def __init__(self,
                graph : dgl_graph,
                data_config : GeoGraphConfig,
                model_config : GeoGraphConfig, 
                method_config : GeoGraphConfig
                ):
        self.graph = graph
        self.data_config = data_config
        self.model_config = model_config
        self.method_config = method_config
        self.set_up_experiment()
        self.has_run = False
        
    def set_up_experiment(self):
        # determine if working with real features or random embedding
        if len(self.data_config.value_fields) > 0 and self.data_config.random_embedding > 0:
            warnings.warn('''Features and random embedding configuration detected. 
                        \n Default to real features.''')
            self.data_config.random_embedding = 0
        
        # checking that the features that will be selected are the same size the models input shape
        if len(self.data_config.value_fields) != self.model_config.in_feats:
            if self.data_config.random_embedding > 0:
                self.model_config.in_feats = self.data_config.random_embedding
            elif len(self.data_config.value_fields) > 0:
               self.model_config.in_feats = len(self.model_config.value_fields)
               warnings.warn(f'''Warning: Number of selected features[{len(self.data_config.value_fields)}]
                                != model input setting [{self.model_config.in_feats}]!
                                \n Setting to Number of selected features[{len(self.data_config.value_fields)}].''')
            
        # ensure the features selected are part of the graph object
        if not set(self.data_config.value_fields).issubset(list(self.graph.ndata.keys())):
            raise KeyError("Graph does not contain expected keys.")
        
        # define inputs and trainingselection
        self.node_embed = nn.Embedding(self.graph.num_nodes(), self.model_config.in_feats)
        
        # if we use a real feature set
        if len(self.data_config.value_fields) > 0:
            self.real_feats = list(zip(*[self.graph.ndata[feature] for feature in self.data_config.value_fields]))
            self.node_embed.weight = nn.Parameter(tensor(self.real_feats),requires_grad=True)
            
        self.train_selection = self.graph.ndata[self.data_config.train_role].bool()
        self.true_labels = self.graph.ndata[self.data_config.label_field]
        
        # initialise the model
        self.model = initialise_model(self.model_config)

        # loading the ml/learnable params
        self.optim_obj = getattr(optim, self.method_config.optimiser)
        self.parameters = itertools.chain(self.model.parameters(), self.node_embed.parameters())
        self.optimiser = self.optim_obj(self.parameters, self.method_config.lr)
        self.loss_obj = getattr(F, self.method_config.loss)
        
        self.metrics = {}
        self.metrics['loss'] = []
        self.metrics['f1'] = []
        self.metrics['prec'] = []
        self.metrics['recall'] = []
        
    def run_exp(self):
        if self.has_run:
            raise AttributeError('This experiment has already been run.')
        
        self.has_run = True
        for i in range(self.method_config.epochs):
            
            predictions = self.model(self.graph, self.node_embed.weight)
            self.loss_val = self.loss_obj(predictions[self.train_selection], self.true_labels[self.train_selection].long())
            self.optimiser.zero_grad()
            self.loss_val.backward()
            self.optimiser.step()
            self.metrics['loss'].append(self.loss_val.item())
            pre_val = np.argmax(np.array(predictions.detach()), axis=1)
            tru_val =  np.array(self.true_labels.detach())
            accuracy_dict = calc_accuracy(pre_val, tru_val)
            
            for k, v in accuracy_dict.items():
                self.metrics[k].append(v)
            
            self.graph.ndata[f'Prediction_Epoch_{str(i).zfill(4)}'] = predictions
            
        return(self.metrics, self.graph, [self.model_config,self.data_config,self.method_config])

    def run_iter(self,i):
        
        if self.has_run:
            self.method_config.epochs = self.method_config.epochs + 1
        else:
            self.method_config.epochs = 1
        
        self.has_run = True
            
        predictions = self.model(self.graph, self.node_embed.weight)
        self.loss_val = self.loss_obj(predictions[self.train_selection], self.true_labels[self.train_selection].long())
        self.optimiser.zero_grad()
        self.loss_val.backward()
        self.optimiser.step()
        self.metrics['loss'].append(self.loss_val.item())
        pre_val = np.argmax(np.array(predictions.detach()), axis=1)
        tru_val =  np.array(self.true_labels.detach())
        accuracy_dict = calc_accuracy(pre_val, tru_val)        
        self.graph.ndata[f'Prediction_Epoch_{str(i).zfill(4)}'] = predictions
        for k, v in accuracy_dict.items():
                self.metrics[k].append(v)
                
        return(self.metrics['loss'][-1])
