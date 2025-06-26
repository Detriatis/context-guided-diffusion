import torch
import torch.nn as nn 
import yaml
import numpy as np 
from swiss_roll import DATA_DIR

def save_model(model: nn.Module, path):
    with open(path, 'wb') as f:
        torch.save(model.state_dict(), f)

def load_model(modelclass, path):
    with open(path, 'rb') as f:
        state = torch.load(f)
        model = modelclass()
        model.load_state_dict(state)
    return model  

def save_metrics(metrics: dict, path):
    with open(path, 'w') as f:
        yaml.dump(metrics, f)

def load_metrics(path):
    with open(path, 'r') as f:
        metrics = yaml.load(f, yaml.FullLoader)
    return metrics

def save_config(config: dict, path):
    with open(path, 'w') as f:
        yaml.dump(config, f) 

def load_config(path):
    with open(path, 'r') as f:
        conf = yaml.load(f, yaml.FullLoader) 
    return conf 

def load_swissroll(path=None, split=True, split_value=1):
    if not path: 
        path = DATA_DIR / 'swiss_roll.npz'
       
    with open(path, 'rb') as f: 
        data = np.load(f)
        xyz_points = data['xyz_points']
        manifold_points = data['manifold_points']
    X = torch.tensor(xyz_points, dtype=torch.float32) 
    Y = torch.tensor(manifold_points, dtype=torch.float32).view(-1, 1)
    
    if split: 
        split = Y.squeeze() < split_value
    
        return X[split], Y[split], X[~split], Y[~split]
    else: 
        return X, Y, None, None 