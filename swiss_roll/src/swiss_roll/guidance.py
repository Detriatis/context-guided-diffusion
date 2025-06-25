import torch
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim
from torch.nn import GaussianNLLLoss

import psutil 
import yaml
import argparse

from swiss_roll.utils import save_model, save_metrics, load_swissroll
from swiss_roll import DATA_DIR, RUNS_DIR, CONF_DIR
from sklearn.datasets import make_swiss_roll
from torch.utils.data import Dataset, DataLoader
from swiss_roll.scheduling import Schedular
from itertools import product

torch.random.manual_seed(42) 

class Data(Dataset):
    def __init__(self, x, y):
        super(Dataset, self).__init__()
        self.x = x
        self.y = y
        
        try: 
            assert len(self.x) == len(self.y)
        except:
            raise AssertionError('Len of data and labels must match')

    def __len__(self):
        return len(self.y) 

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class GuidanceBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GuidanceBlock, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.dropout =  nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.layer(x)
        x = torch.sin(x) 
        x = self.dropout(x)
        return x
    
class GuidanceModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2, n_hidden_layers=2):
        super(GuidanceModel, self).__init__()

        self.inlayer = GuidanceBlock(input_dim, hidden_dim)
        self.midlayer = nn.ModuleList([GuidanceBlock(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.outlayer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.inlayer(x)
        for layer in self.midlayer: 
            x = layer(x)

        x = self.outlayer(x) 
        return x
    
class ContextEmbedding(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

def cgd_regularization_term(
    model_predictions: torch.Tensor,
    context_embeddings: torch.Tensor,
    covariance_scale_hyper: float,
    diagonal_offset_hyper: float,
    target_logvar_hyper: float,
    target_mean_hyper: float,
):
    from torch.distributions import MultivariateNormal

    """Computes the Context-Guided Diffusion regularization term."""
    K = torch.matmul(context_embeddings, context_embeddings.T)
    K = K * covariance_scale_hyper
    K = K + torch.eye(K.shape[0], device=K.device) * diagonal_offset_hyper
    
    num_output_dims = model_predictions.shape[-1]
    
    mean_preds = model_predictions[:, :(num_output_dims // 2):]
    var_preds = model_predictions[:, (num_output_dims // 2):]

    mean_target = torch.ones_like(mean_preds) * target_mean_hyper
    var_target = torch.ones_like(var_preds) * target_logvar_hyper 
    
    means_likelihood = MultivariateNormal(mean_target.T, K)
    vars_likelihood = MultivariateNormal(var_target.T, K)

    mean_log_p = means_likelihood.log_prob(mean_preds.T)
    var_log_p =  vars_likelihood.log_prob(var_preds.T)
    
    log_ps = torch.cat([mean_log_p, var_log_p], dim=0)
    return -log_ps.sum()

def sample_uniform(dims, low, high):
    return (high - low) * torch.rand(size=dims) + (low)

def train_guidance_model(
    guidance_model, 
    context_encoder, 
    guidance_optimizer, 
    schedular: Schedular,
    dataloader,  
    l2_lambda,
    sigma_t, 
    tau_t,
    reg_lambda,
    target_mean_val, 
    target_logvar_val,
    ctx_size,
    use_ctx,
    ctx_set,
    n_epochs, 
    device,
    run_id = None,
    **kwargs
):
    from torch.nn.functional import softplus

    guidance_model.train()
    for epoch in range(n_epochs):
        for x, y in dataloader:
            t = schedular.randtimesteps(x)
            x, y = x.to(device), y.to(device)
            x_perturbed, eps = schedular.noise(x, t) 
            preds = guidance_model(x_perturbed)
            mu = preds[:, :1]
            var = torch.exp(preds[:, 1:])
            
            # NLL Loss
            loss_fn = GaussianNLLLoss()
            nll = loss_fn(mu, y, var)
            # CGD Regularization
            if use_ctx: 
                idx = torch.randperm(ctx_set.size(0))[:ctx_size]
                x_ctx = ctx_set[idx]

                t = schedular.randtimesteps(x_ctx)
                x_ctx_perturbed, ctx_eps = schedular.noise(x_ctx, t)


                with torch.no_grad():
                    ctx_embeds = context_encoder(x_ctx_perturbed)

                
                ctx_preds = guidance_model(x_ctx_perturbed)
                
                reg = cgd_regularization_term(
                    model_predictions = ctx_preds,
                    context_embeddings = ctx_embeds,
                    covariance_scale_hyper = sigma_t,
                    diagonal_offset_hyper = tau_t,
                    target_logvar_hyper = target_logvar_val,
                    target_mean_hyper = target_mean_val,
                )
                reg_scale = batch_size * len(dataloader)
                reg /= reg_scale
            else: 
                reg = 0 
            
            # L2 Regularization
            l2 = sum((p ** 2).sum() for p in guidance_model.parameters())
            loss = nll + l2_lambda * l2 + reg_lambda * reg 
            
            guidance_optimizer.zero_grad()
            loss.backward()
            guidance_optimizer.step()
        print(f"NLL: {nll.mean().item():.4f}, L2: {l2_lambda * l2:.4f}, Reg: {reg:.4f}")
        print(f"Total loss {loss:.4f}")
        if use_ctx: 
            print(f"Predicted mean: {ctx_preds[:, 0].mean().item():.4f}", 
                  f"Predicted uncertainity {ctx_preds[:, 1].mean().item():.4f}")
        print("Epoch", epoch)

def evaluate_model(model, dataloader, device='cpu', coverage_alpha=0.05):
    model.eval()
    total_nll = 0.0
    total_mse = 0.0
    total_in_interval = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            mu = preds[:, :1]
            var = torch.exp(preds[:, 1:])
            std = var.sqrt()

            # NLL
            dist = torch.distributions.Normal(mu, std)
            nll = -dist.log_prob(y)
            total_nll += nll.sum().item()

            # MSE
            mse = ((mu - y) ** 2)
            total_mse += mse.sum().item()

            # Coverage: fraction of points inside predicted interval
            z = torch.distributions.Normal(0, 1).icdf(
                torch.tensor(1 - coverage_alpha / 2)
            )
            lower = mu - z * std
            upper = mu + z * std
            in_interval = ((y >= lower) & (y <= upper)).float().sum().item()
            total_in_interval += in_interval

            total_samples += y.numel()

    return {
        "nll": total_nll / total_samples,
        "mse": total_mse / total_samples,
        f"{int((1 - coverage_alpha) * 100)}%_coverage": total_in_interval / total_samples
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index') 
    args = parser.parse_args()
    index = args.index

    device='cpu'
    conf = CONF_DIR / 'guidance_conf' / f'{index}.yaml'
    
    with open(conf, 'r') as f:
        conf = yaml.load(f, yaml.FullLoader) 
      
    X, Y, _, _  = load_swissroll(split=True, split_value=1) 
    X = X[:, [0, 2]]
    data = Data(X, Y)
    batch_size = 128
    dataloader = DataLoader(data, batch_size=batch_size) 

    guidance_model = GuidanceModel(2, 32 , 2, 2)
    embedding_generator = GuidanceModel(2, 32, 2, 2)
   
    def context_encoder(x):
        h=embedding_generator.inlayer(x)
        for layer in embedding_generator.midlayer:
            h=layer(h)
        return h
    
    schedular = Schedular()
    
    guidance_optimiser = optim.Adam(guidance_model.parameters(), 1e-2)
    device='cpu'
    
    target_meanval = Y.mean().to(device)
    print(target_meanval)
    target_logvar = torch.tensor([0.7], dtype=torch.float32)

    ctx_set = sample_uniform((10000, 2), -2.5, 2.5)

    train_guidance_model(guidance_model=guidance_model, 
                         context_encoder=context_encoder, 
                         guidance_optimizer=guidance_optimiser, 
                         dataloader=dataloader, 
                         target_mean_val=target_meanval,
                         target_logvar_val=target_logvar,
                         ctx_set=ctx_set,
                         schedular=schedular,
                         n_epochs = 100,
                         **conf,
                         device=device)
   
    results = evaluate_model(guidance_model, dataloader, device)

    path = RUNS_DIR / 'guidance_models' / f'run_{conf["run_id"]}'
   
    if not path.exists():
        path.mkdir(parents=True)
    
    save_metrics(results, path / 'eval_metrics.yaml') 
    save_model(guidance_model, path / 'guidance_model.pth')

