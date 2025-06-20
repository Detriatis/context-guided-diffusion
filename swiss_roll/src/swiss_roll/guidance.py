import torch
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim
import psutil 
import yaml
import argparse

from swiss_roll.utils import save_model, save_metrics, load_swissroll
from swiss_roll import DATA_DIR, RUNS_DIR, CONF_DIR
from sklearn.datasets import make_swiss_roll
from torch.utils.data import Dataset, DataLoader
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
        mu = x[:, :1]
        var = x[:, 1:]
        return mu, var  
    
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
    mean_preds: torch.Tensor, 
    var_preds: torch.Tensor,
    context_embeddings: torch.Tensor,
    covariance_scale_hyper: float,
    diagonal_offset_hyper: float,
    target_variance_hyper: float,
    target_mean_hyper: float, 
):
    """
    Compute the context-guided diffusion regularization term.

    Args:
        model_predictions: Predictions of the guidance model on a noised 
            context batch sampled from a problem-informed context set.
        context_embeddings: The embeddings of the noised context points, derived 
            either from a pre-trained or randomly initialized model.
        covariance_scale_hyper: The covariance scale hyperparameter, used to        
            determine the strength of the smoothness constraints in K(x).
            Optionally scaled with the noising schedule of the forward process.
        diagonal_offset_hyper: The diagonal offset hyperparameter, used to
            determine how closely the predictions have to match m(x).
            Optionally scaled with the noising schedule of the forward process
        target_variance_hyper: The target variance hyperparameter, used to
            determine the level of predictive uncertainty on the context set.

    Returns:
        The context-guided diffusion regularization term.
    """

    from torch.distributions import MultivariateNormal

    # construct the covariance matrix and multiply it with 
    # the covariance scale hyperparameter
    K = torch.matmul(context_embeddings, context_embeddings.T)
    K = K * covariance_scale_hyper

    # add the diagonal offset hyperparameter to the diagonal of K
    K = K + torch.eye(K.shape[0]) * diagonal_offset_hyper

    # specify mean functions that encode the desired behavior of
    # reverting to the context set mean and variance hyperparameter
    mean_target = torch.ones_like(mean_preds) * target_mean_hyper # assuming standardized labels
    var_target = torch.ones_like(var_preds) * target_variance_hyper
    # compute the Mahalanobis distance between the predictions and the
    # mean functions defined above through their log-likelihood under
    # a multivariate Gaussian distribution with covariance K
    means_likelihood = MultivariateNormal(mean_target.T, K)
    vars_likelihood = MultivariateNormal(var_target.T, K)
    mean_log_p = means_likelihood.log_prob(mean_preds.T)
    var_log_p = vars_likelihood.log_prob(var_preds.T)
    log_ps = torch.cat([mean_log_p, var_log_p], dim=0)

    return -log_ps.sum()



def sample_uniform(dims, low, high):
    return (high - low) * torch.rand(size=dims) + (low)

def train_guidance_model(
    model, context_encoder, optimizer, 
    dataloader,  
    l2_lambda,
    sigma_t, 
    tau_t,
    target_mean_val, 
    target_logvar_val,
    ctx_size,
    use_ctx, 
    device,
    run_id = None,
    **kwargs
):
    from torch.nn.functional import softplus

    model.train()
    for epoch in range(4):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            mu, r = model(x)

            var = torch.exp(softplus(r))
            # NLL Loss
            dist = torch.distributions.Normal(mu, var ** 1/2)
            nll = -dist.log_prob(y)
            
            
            # CGD Regularization
            if use_ctx: 
                idx = torch.randperm(ctx_set.size(0))[:ctx_size]
                
                with torch.no_grad():
                    ctx_embeds = context_encoder(ctx_set[idx])

                
                mu_ctx, r_ctx = model(ctx_set[idx])
                logvar_ctx = softplus(r_ctx)
                
                reg = cgd_regularization_term(
                    mu_ctx,
                    logvar_ctx,
                    ctx_embeds,
                    covariance_scale_hyper=sigma_t,
                    diagonal_offset_hyper=tau_t,
                    target_mean_hyper=target_mean_val,
                    target_variance_hyper=target_logvar_val
                )
            
                # 1. Scale reg by dataset size (per epoch)
                reg_scale = len(dataloader.dataset)

                # 2. Normalize CGD reg
                reg = reg / reg_scale

                # 3. Rescale based on context size
                reg = reg * ctx_size
            else: 
                reg = 0  
            # L2 Regularization
            l2 = sum((p ** 2).sum() for p in model.parameters())
            loss = nll.mean() + l2_lambda * l2 + reg * 10
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"NLL: {nll.mean().item():.4f}, L2: {l2_lambda * l2:.4f}, Reg: {reg:.4f}")
        print("The mean predicted context", mu_ctx.mean())
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
            mu, r = model(x)
            var = torch.exp(F.softplus(r))
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
    
    XYZ_points, Y = load_swissroll() 
    X = XYZ_points[:, [0, 2]]
    split = Y.squeeze() < 1
    
    data = Data(X[split], Y[split])
    dataloader = DataLoader(data, batch_size=128) 

    guidance_model = GuidanceModel(2, 32 , 2)
    context_encoder = ContextEmbedding(2, 32)
    guidance_optimiser = optim.Adam(guidance_model.parameters(), lr=1e-2)
    device='cpu'
    target_meanval = Y[split].mean().to(device)
    target_logvar = torch.tensor([0.7], dtype=torch.float32)

    ctx_set = sample_uniform((10000, 2), -2.5, 2.5)
    
    train_guidance_model(guidance_model, context_encoder, guidance_optimiser, dataloader, 
                target_mean_val=target_meanval,
                target_logvar_val=target_logvar,
                ctx_set=ctx_set,
                **conf,
                device=device)
   
    results = evaluate_model(guidance_model, dataloader, device)

    path = RUNS_DIR / 'guidance_models' / f'run_{conf["run_id"]}'
   
    if not path.exists():
        path.mkdir(parents=True)
    
    save_metrics(results, path / 'eval_metrics.yaml') 
    save_model(guidance_model, path / 'guidance_model.pth')

