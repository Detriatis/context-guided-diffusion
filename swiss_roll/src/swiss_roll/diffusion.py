import torch
import torch.nn as nn
import torch.optim as optim 

from swiss_roll import DATA_DIR
from swiss_roll.utils import load_model, load_swissroll, save_model
from swiss_roll.scheduling import Schedular

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

def train_diffusion_model(X, device='cpu', retrain=True): 
    schedular = Schedular(T=40, s=0.008) 
    schedule = schedular.schedule
    baralphas = schedular.bar_alphas
    alphas = schedular.alphas
    betas = schedular.betas
    
    gen_model_path = DATA_DIR / 'generator.pth'

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
                timesteps = schedular.timesteps(XBatch)
                
                Xnoise, eps = schedular.noise(XBatch, timesteps, baralphas)
                pred_eps = model(Xnoise.to(device), timesteps.to(device))
                
                loss = loss_fn(pred_eps, eps.to(device))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                
                steps += 1 
                epoch_loss += loss
            print(f"Epoch {epoch} loss = {epoch_loss / steps}") 
        
        save_model(model, gen_model_path)

if __name__ == '__main__':
    XYZ_points, Y = load_swissroll() 
    X = XYZ_points[:, [0, 2]]
    train_diffusion_model(X, 'cpu', True)