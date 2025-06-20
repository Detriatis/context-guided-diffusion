import torch
import torch.nn as nn
import torch.optim as optim 

from swiss_roll import DATA_DIR
from swiss_roll.utils import load_model

class DiffusionBlock(nn.Module):
    def __init__(self, nunits):
        super(DiffusionBlock, self).__init__()
        self.linear = nn.Linear(nunits, nunits)

    def forward(self, x):
        x = self.linear(x)
        return nn.functional.relu(x)

class DiffusionModel(nn.Module):
    def __init__(self, n_features=2, n_blocks=4, n_units=64): 
        super(DiffusionModel, self).__init__()

        self.inblock = nn.Linear(n_features + 1, n_units)
        self.blocks = nn.ModuleList([
            DiffusionBlock(n_units) for _ in range(n_blocks)
        ])
        self.outblock = nn.Linear(n_units, n_features)

    def forward(self, x, t):
        x = torch.hstack((x, t))
        x = self.inblock(x)
        for block in self.blocks:
            x = block(x)
        return self.outblock(x)

def noise(x, t, bar_alphas):
    eps = torch.randn_like(x)
    mu = (bar_alphas[t] ** 0.5).repeat(1, x.shape[1]) * x
    var = ((1 - bar_alphas[t]) ** 0.5).repeat(1, x.shape[1]) * eps
    return mu + var, eps

def cosine_schedule(T:int, s:float) -> torch.Tensor:
    t = torch.arange(0, T, 1, dtype=torch.float32)
    schedule = torch.cos((t/T+ s) / (1 + s) * torch.pi / 2) ** 2
    return schedule 

def get_bar_alphas(schedule):
    return schedule / schedule[0] 

def get_alphas(bar_alphas):
    return  bar_alphas / torch.concat([bar_alphas[0:1], bar_alphas[0:-1]])

def get_betas(bar_alphas): 
    return 1 - (bar_alphas / torch.concat([bar_alphas[0:1], bar_alphas[:-1]])) 


def train_diffusion_model(X, device, retrain, gen_model_path): 
    
    schedule = cosine_schedule(40, s=0.008)
    baralphas = get_bar_alphas(schedule) 
    alphas = get_alphas(baralphas)
    betas = get_betas(baralphas)
    T = 40

    if gen_model_path.exists() and not retrain:
        model = load_model(DiffusionModel, gen_model_path)
        return model
    else: 
        model = DiffusionModel(
            n_features=2, 
            n_blocks=4, 
            n_units=64
            )
    
    model.train()
    model.to(device)
    batch_size = 2048
    n_epochs = 100
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    #scheduler = optim.lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.01, total_iters=n_epochs)

    for epoch in range(n_epochs):
        epoch_loss = steps = 0
        for i in range(0, len(X), batch_size):
            XBatch = X[i:i+batch_size]
            timesteps = torch.randint(0, T, size=[len(XBatch), 1])
            
            Xnoise, eps = noise(XBatch, timesteps, baralphas)
            pred_eps = model(Xnoise.to(device), timesteps.to(device))
            
            loss = loss_fn(pred_eps, eps.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            steps += 1 
            epoch_loss += loss
        print(f"Epoch {epoch} loss = {epoch_loss / steps}") 