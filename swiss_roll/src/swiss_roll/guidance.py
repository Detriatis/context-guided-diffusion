import torch
import math
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim
from torch.nn import GaussianNLLLoss

import copy 
import yaml
import logging
import argparse
import time
from pathlib import Path 

from swiss_roll.utils import save_model, save_metrics, load_swissroll, save_config 
from swiss_roll import DATA_DIR, RUNS_DIR, CONF_DIR, PROJECT_ROOT
from swiss_roll.config import load_config, Config, Metrics, TrainValMetrics
from sklearn.datasets import make_swiss_roll
from torch.utils.data import Dataset, DataLoader
from swiss_roll.scheduling import Schedular
from itertools import product
from dataclasses import asdict
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
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=2, n_hidden_layers=2):
        super(GuidanceModel, self).__init__()

        self.inlayer = GuidanceBlock(input_dim, hidden_dim)
        self.midlayer = nn.ModuleList([GuidanceBlock(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.outlayer = nn.Linear(hidden_dim, output_dim)
    
    def embed(self, x):
        x = self.inlayer(x) 
        for layer in self.midlayer:
            x = layer(x) 
        return x 
    
    def forward(self, x):
        h = self.embed(x) 
        x = self.outlayer(h) 
        return x, h
    
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
    logvar_preds = model_predictions[:, (num_output_dims // 2):]
    
    mean_target = torch.ones_like(mean_preds) * target_mean_hyper
    logvar_target = torch.ones_like(logvar_preds) * target_logvar_hyper
    try: 
        means_likelihood = MultivariateNormal(loc=mean_target.T, covariance_matrix=K)
        logvars_likelihood = MultivariateNormal(loc=logvar_target.T, covariance_matrix=K)
    except ValueError as em:
        return torch.tensor(float("inf"), device=K.device)
    mean_log_p = means_likelihood.log_prob(mean_preds.T)
    logvar_log_p =  logvars_likelihood.log_prob(logvar_preds.T)
    
    log_ps = torch.cat([mean_log_p, logvar_log_p], dim=0)

    return -log_ps.sum()

def sample_uniform(dims, low, high, device=None) -> torch.Tensor:
    return (high - low) * torch.rand(size=dims, dtype=torch.float32, device=device) + (low)

def toy_cgd_term(preds, target_logvar, target_mean):
    logvar = preds[:, 1]
    mean = preds[:, 0]
    return ((logvar- target_logvar)**2).mean() + ((mean - target_mean)**2).mean()
import torch

def train_guidance_model(
    guidance_model: GuidanceModel, 
    context_encoder: GuidanceModel, 
    guidance_optimizer, 
    schedular: Schedular,
    dataloader: DataLoader,
    ctx_set: torch.Tensor,
    cfg: Config, 
    logger, 
    device,
):
    
    sigma_t = torch.tensor(float(cfg.cgd.sigma_t), dtype=torch.float32, device=device, requires_grad=False)
    tau_t = torch.tensor(float(cfg.cgd.tau_t), dtype=torch.float32, device=device, requires_grad=False)
    guidance_model.train()
    
    for epoch in range(cfg.train.n_epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
 
            t = schedular.randtimesteps(x)
            x_perturbed, eps = schedular.noise(x, t) 
            if cfg.run.expose_time: 
                x_perturbed = torch.hstack([x_perturbed, t]) 
            
            preds, embed = guidance_model(x_perturbed)
            mu, logvar = preds.chunk(2, dim=-1)
            var = torch.exp(logvar)
            loss_fn = GaussianNLLLoss(full=True)
            y_perturbed, _ = schedular.noise(y, t)

            nll = loss_fn(mu, y_perturbed, var)
            
            # CGD Regularization
            if cfg.run.reg_type != 'None':
                X_ctx = ctx_set[torch.randperm(ctx_set.shape[0])[:cfg.cgd.ctx_size]]
                t = schedular.randtimesteps(X_ctx)
                x_ctx_perturbed, ctx_eps = schedular.noise(X_ctx, t)
                if cfg.run.expose_time:
                    x_ctx_perturbed = torch.hstack([x_ctx_perturbed, t]) 
                
                ctx_preds, ctx_embeds = guidance_model(x_ctx_perturbed) 
                
                with torch.no_grad():
                    if not cfg.run.update_embeds:
                        _, ctx_embeds = context_encoder(x_ctx_perturbed)
                
                if cfg.run.reg_type == "cgd":
                    raw_reg = cgd_regularization_term(
                        model_predictions = ctx_preds,
                        context_embeddings = ctx_embeds.detach(),
                        covariance_scale_hyper = sigma_t,
                        diagonal_offset_hyper = tau_t,
                        target_logvar_hyper = cfg.cgd.target_logvar,
                        target_mean_hyper = cfg.cgd.target_mean,
                    )
            
                if cfg.run.reg_type == "mse":
                    raw_reg = toy_cgd_term(ctx_preds, 
                                        target_logvar=cfg.cgd.target_logvar,
                                        target_mean=cfg.cgd.target_mean) 
           
            if cfg.run.reg_type == "None":
                raw_reg = 0


            if cfg.cgd.reg_scale_by== 'dataset_size': 
                reg_scale = cfg.train.batch_size * len(dataloader)
           
            if cfg.cgd.reg_scale_by== 'ctx_size':
                reg_scale = cfg.cgd.ctx_size


            reg = (raw_reg / reg_scale) * cfg.cgd.reg_lambda


            # L2 Regularization
            l2 = sum((p ** 2).sum() for p in guidance_model.parameters())
            l2 = ((1 / (2 * cfg.opt.l2_lambda)) * l2) / reg_scale

            loss = nll.mean() + l2 + reg
            
            guidance_optimizer.zero_grad()
            loss.backward()
            guidance_optimizer.step()
        
        logger.info(f"NLL: {nll.mean().item():.4f}, L2: {l2:.4f}, Reg: {reg}")
        logger.info(f"Raw Reg: {raw_reg}")
        logger.info(f"Total loss {loss:.4f}")
        if cfg.run.reg_type != 'None': 
            logger.info(f"Predicted context mean: {ctx_preds[:, 0].mean().item():.4f}") 
            logger.info(f"Predicted context std: {ctx_preds[:, 0].std().item():.4f}") 
            logger.info(f"Predicted context uncertainity {ctx_preds[:, 1].mean().item():.4f}")
            logger.info(f"Predicted context uncertainity std {ctx_preds[:, 1].std().item():.4f}")
        logger.info(f"Completed epoch {epoch+1} of {cfg.train.n_epochs}\n")


def evaluate_model(model, dataloader, cfg: Config, device='cpu', coverage_alpha=0.05):

    if cfg.cgd.reg_scale_by== 'dataset_size': 
        reg_scale = cfg.train.batch_size * len(dataloader)
    
    if cfg.cgd.reg_scale_by== 'ctx_size':
        reg_scale = cfg.cgd.ctx_size

    model.eval()
    total_nll = 0.0
    total_mse = 0.0
    total_supervised_loss = 0.0
    total_in_interval = 0
    total_samples = 0
    l2 = sum((p ** 2).sum() for p in model.parameters())
    l2 = ((1 / (2 * cfg.opt.l2_lambda)) * l2) / reg_scale

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            t = torch.zeros(size=(x.size(0), 1))
            if cfg.run.expose_time: 
                x = torch.hstack([x, t])
            preds, _ = model(x)
            mu = preds[:, :1]
            var = torch.exp(preds[:, 1:])
            std = var.sqrt()

            # NLL
            dist = torch.distributions.Normal(mu, std)
            nll = -dist.log_prob(y)
            total_nll += nll.sum().item()
            total_supervised_loss += (nll + l2).sum() 
            # MSE
            mse = ((mu - y) ** 2)
            total_mse += mse.sum().item()

            # Coverage: fraction of points inside predicted interval
            z = torch.distributions.Normal(0, 1).icdf(
                torch.tensor(1 - coverage_alpha / 2, device=device)
            )
            lower = mu - z * std
            upper = mu + z * std
            in_interval = ((y >= lower) & (y <= upper)).float().sum().item()
            total_in_interval += in_interval

            total_samples += y.numel()

    results = Metrics(**{
        "nll": total_nll / total_samples,
        "mse": total_mse / total_samples,
        "loss": total_supervised_loss / total_samples,
        "l2": l2, 
        "cov": total_in_interval / total_samples
    })
    
    return results


def get_dataloaders(batch_size):
    X, Y, high_X, high_Y  = load_swissroll(split=True, split_value=1) 
    X = X[:, [0, 2]]
    data = Data(X, Y)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True) 

    high_X = high_X[:, [0, 2]]
    high_data = Data(high_X, high_Y)
    high_dataloader = DataLoader(high_data, batch_size=batch_size)
    return dataloader, high_dataloader

def run_once(cfg: Config, logger) -> tuple[Metrics, Metrics, GuidanceModel]:
    '''
    returns 
    training_metrics: dict
    validation_metrics: dict
    guidance_model: GuidanceModel
    '''
    logger.info(f'Starting run with following parameters {cfg}') 
    cfg = copy.deepcopy(cfg)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train = cfg.train 
    dataloader, high_dataloader = get_dataloaders(train.batch_size)

    run = cfg.run 
    if run.expose_time:
        input_dim = 3 
    else: 
        input_dim = 2 

    guidance_model = GuidanceModel(input_dim, 32, 2, 2).to(device) 
    context_encoder = GuidanceModel(input_dim, 32, 2, 2).to(device).eval()

    opt = cfg.opt 
    guidance_optimiser = optim.Adam(
        guidance_model.parameters(),
        opt.lr 
    )

    schedular = Schedular(device=device) 

    cgd = cfg.cgd 
    cgd.target_logvar = torch.tensor([cgd.target_logvar], dtype=torch.float32).to(device)
    cgd.target_mean = torch.tensor([cgd.target_mean], dtype=torch.float32).to(device)

    samp = cfg.samp
    ctx_X = samp.sampler_func(samp.sample_shape, **samp.sampler_args)

    train_guidance_model(guidance_model=guidance_model, 
                         context_encoder=context_encoder, 
                         guidance_optimizer=guidance_optimiser, 
                         dataloader=dataloader, 
                         schedular=schedular,
                         ctx_set=ctx_X,
                         logger=logger,
                         cfg=cfg,
                         device=device)
    

    validation_metrics = evaluate_model(guidance_model, high_dataloader, cfg, device) 
    training_metrics = evaluate_model(guidance_model, dataloader, cfg, device)
    
    return training_metrics, validation_metrics, guidance_model 

def writeout_results(run_metrics, guidance_model, cfg: Config, writeout=None): 
    if index: 
        writeout = RUNS_DIR / 'guidance_models' / f'run_{cfg.run.run_id}'
    
    if not writeout.exists():
        writeout.mkdir(parents=True)
    
    save_metrics(run_metrics, writeout / 'eval_metrics.yaml') 
    save_model(guidance_model, writeout / 'guidance_model.pth')
    save_config(conf, writeout / 'conf.yaml')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=False, default=None) 
    parser.add_argument('--conf', required=False, default=None, help='Full conf path')
    parser.add_argument('--logger', required=False, default=None, help='Full logging path')
    parser.add_argument('--writeout', required=False, default=None, help='Full writeout path') 
    
    args = parser.parse_args()
    start = time.perf_counter()
    
    if args.logger is None:
        args.logger = Path("/dev/null")
    else: 
        args.logger = Path(args.logger).with_suffix('.log')
        args.logger.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(filename=args.logger, encoding='utf-8', level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")
    logger = logging.getLogger(__name__) 
    
    index = args.index
    conf = args.conf
    writeout = args.writeout

    assert (index is not None) or (conf is not None and writeout is not None), "Must provide either --index or both --conf and --writeout"
    if conf is not None: 
        conf = Path(conf)
    if writeout is not None: 
        writeout = Path(writeout) 
    
    if index:
        conf = CONF_DIR / 'guidance_conf' / f'{index}.yaml'
    
    cfg = load_config(conf) 
    training_metrics, validation_metrics, guidance_model = run_once(cfg, logger) 
    
    run_metrics =TrainValMetrics(**{
        'train': training_metrics,
        'val': validation_metrics 
    })

    writeout_results(asdict(run_metrics), guidance_model, cfg, writeout)
   
    logger.info(f"{time.perf_counter() - start:.6f} s")