import torch
import torch.nn as nn 
import yaml

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