import warnings

class GeoGraphConfig:
    def __init__(self):
        self.type = None
        
    @classmethod
    def config2json(cls):
        return({x:y for x,y in vars(cls).items() if '__' not in x})
    
    @classmethod
    def from_data_config(cls, config_object:dict):
        required_keys = [
            'feature_list',
            'value_fields',
            'label_field',
            'train_role',
            'node_type',
            'graph_con_method',
            'graph_con_variables',
            'random_embedding'     
        ]
        
        if set(required_keys).issubset(set(config_object.keys())):
            for k,v in config_object.items():
                setattr(cls, k, v)
                cls.type = 'data'
            return(cls)
        
        elif config_object=={}:
            warnings.warn(f'''Warning: Data config blank''')
            for k in required_keys:
                setattr(cls, k, None)
            cls.type = 'data'
            return(cls)
        
        else:
            raise KeyError('Config missing keys for data instance.')

    @classmethod
    def from_model_config(cls, config_object:dict):
        required_keys = [
            'model_type',
            'in_feats',
            'hid_feats',
            'num_classes',  
            'num_heads'
        ]
        
        if set(required_keys).issubset(set(config_object.keys())):
            for k,v in config_object.items():
                setattr(cls, k, v)
                cls.type = 'model'
            return(cls)
        
        elif config_object=={}:
            warnings.warn(f'''Warning: Model config blank''')
            for k in required_keys:
                setattr(cls, k, None)
            cls.type = 'model'
            return(cls)
        
        else:
            raise KeyError('Config missing keys for model instance.')
        
    @classmethod
    def from_method_config(cls, config_object:dict):
        required_keys = [
            'method_type',
            'epochs',
            'loss',
            'optimiser',
            'lr',
            'repeats'
        ]
        
        if set(required_keys).issubset(set(config_object.keys())):
            for k,v in config_object.items():
                setattr(cls, k, v)
                cls.type = 'method'
            return(cls)
        
        elif config_object=={}:
            warnings.warn(f'''Warning: Method config blank''')
            for k in required_keys:
                setattr(cls, k, None)
            cls.type = 'method'
            return(cls)
        
        else:
            raise KeyError('Config missing keys for method instance.')