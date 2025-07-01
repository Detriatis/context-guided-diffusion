import torch
import torch.nn as nn 
import yaml
import numpy as np 
import matplotlib.pyplot as plt
from swiss_roll import DATA_DIR
from torch.distributions import MultivariateNormal 

def save_model(model: nn.Module, path):
    with open(path, 'wb') as f:
        torch.save(model.state_dict(), f)

def load_model(modelclass, model_config, path):
    with open(path, 'rb') as f:
        state = torch.load(f)
        model = modelclass(**model_config)
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

def sample_uniform(dims, low, high, device=None) -> torch.Tensor:
    return (high - low) * torch.rand(size=dims, dtype=torch.float32, device=device) + (low)

def sample_gaussian_context(dims, mean=(0, 0), cov_scale=10.0, device=None) -> torch.Tensor:
    cov = torch.eye(dims[-1], device=device) * cov_scale
    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    dist = MultivariateNormal(mean, cov)
    return dist.sample((dims[0], ))

def mixture_of_gaussians(dims, means, cov_scales, weights, device):
    if sum(weights) != 1.0: 
        weights /= sum(weights)
    
    counts = [int(dims[0] * weight) for weight in weights] 
    counts[0] += dims[0] - sum(counts) 

    parts = [] 
    for mean, cov_scale, count in zip(means, cov_scales, counts):
        if count > 0:
            batch = sample_gaussian_context((count, dims[-1]), mean=mean, cov_scale=cov_scale, device=device)
            parts.append(batch)
    samples = torch.cat(parts, dim=0) 
    idx = torch.randperm(dims[0], device=device)
    return samples[idx]

if __name__ == '__main__':
    batch_size = 10_000
    means = [(0.0, 0.0), (3.0, 0.0), (-2.0, 2.0)]
    cov_scales = [2 , 0.4, 0.6]
    weights = [0.5, 0.3, 0.2]
    samples = mixture_of_gaussians(dims=(batch_size, 2), means=means, cov_scales=cov_scales, weights=weights, device='cpu')
    plt.scatter(samples[:,0], samples[:, 1])
    plt.savefig('./testplot.png')
    X, Y, _, _ = load_swissroll(split=False)
    print(Y.min())