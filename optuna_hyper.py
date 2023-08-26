import  optuna
from    optuna.trial import TrialState
from datetime import datetime
import json
import os
import torch
import numpy as np
from geo_graphnet.data_handling.graph_loading import load_homo_graph
import itertools
from dgl.data.utils import save_graphs

from geo_graphnet.methods.hom_methods import SemiSupervisedPred, disclaimer
from geo_graphnet.data_handling.config_handlers import GeoGraphConfig

class Objective(object):
    def  __init__(self, node_file, edge_file, experiment_dir, metric='loss'):
        self.experiment_dir = experiment_dir
        self.selected_metric = metric
        self.starting_graph = load_homo_graph(node_file=node_file, edge_file=edge_file)
        self.starting_graph.ndata['train'] = torch.Tensor([True if x in np.unique(self.starting_graph.ndata['Line'].numpy()) else False 
                                                     for x in self.starting_graph.ndata['Line'].numpy()])
        self.features = [x for x in self.starting_graph.ndata.keys() if x not in ['Line', 'Base', 'label', 'train']] 
        self.feature_combinations = []
        
        for r in range(1, len(self.features) + 1):
            self.feature_combinations.extend([list(x) for x in itertools.combinations(iterable=self.features, r=r)])
            
        self.feature_combinations.extend([]) # random embedding test
        self.best_value  = 100 
        print(disclaimer)
    
    def __call__(self, trial):
        # for now we don't mess with the data config
        with open(r'C:\Users\bogo\Desktop\code\geo_graphnet\configs\data.json', "r") as read_file:
            data = json.load(read_file)
        data_config = GeoGraphConfig.from_data_config(data)
        data_config.value_fields    = trial.suggest_categorical('feats', self.feature_combinations)  
        
        method_config = GeoGraphConfig.from_method_config({})
        method_config.method_type   = 'semi'
        method_config.epochs        = trial.suggest_int('epochs',1,100)
        method_config.loss          = trial.suggest_categorical('loss',["nll_loss"])
        method_config.optimiser     = trial.suggest_categorical('optim',[
                                                                        "Adam",
                                                                        "Adamax"
                                                                        ])

        method_config.lr            = trial.suggest_float('lr',0.0001,1)
        
        model_config = GeoGraphConfig.from_model_config({})
        model_config.model_type  = trial.suggest_categorical('model',['gcn'])
        model_config.in_feats    = trial.suggest_int('rand_feat',0,100) if len(data_config.value_fields)==0 else len(data_config.value_fields)
        model_config.hid_feats   = trial.suggest_int('hidden_feats',0,1000)
        model_config.num_classes = 2
        model_config.num_heads   = 0 if model_config.model_type == 'gcn' else trial.suggest_int('num_heads',0,7)  
              
        # initialise the instnace
        experiment = SemiSupervisedPred(self.starting_graph, data_config, model_config, method_config)
        
        for i in range(method_config.epochs):
            experiment.run_iter(i)
            metric = experiment.metrics[self.selected_metric][-1]
            trial.report(metric,i)
            
            if i == 20:
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        if metric  < self.best_value: 
            self.best_value = metric
            # save behaviour 
            save_graphs(f"{self.experiment_dir}/best_model_graph.bin", experiment.graph)
            
            with open(f'{self.experiment_dir}/best_model_metrics.json', 'w') as fp:
                json.dump(experiment.metrics,fp)
                
            configs = [experiment.data_config,experiment.model_config,experiment.method_config]
            
            for conf in configs:
                conf_dict = conf.config__2json()
                with open(f"{self.experiment_dir}/best_model_{conf_dict['type']}_config.json",'w') as fp:
                    json.dump(conf_dict,fp)
                
        return metric

if __name__=="__main__":
    out_dir = "out"
    edges = r"C:\Users\bogo\Desktop\code\geo_graphnet\data\homo_graphs\real_edges_lattice.csv"
    nodes = r"C:\Users\bogo\Desktop\code\geo_graphnet\data\homo_graphs\real_nodes.csv"
    mode = "semi"
    gnn_type = "GCN"
    
    now = datetime.now()
    date = now.strftime('%d%m%y_%H%M%S')
    study_event = f'{date}_{mode}_production_test'
    
    experiment_dir = f"{out_dir}\{study_event}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    study = optuna.create_study(
                                direction="minimize", 
                                study_name=study_event,
                                storage=f"sqlite:///{experiment_dir}/{mode}_{gnn_type}_lattice.db",
                                load_if_exists=True
                                )
    
    study.optimize(Objective(nodes,edges,experiment_dir), n_trials=1000)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))