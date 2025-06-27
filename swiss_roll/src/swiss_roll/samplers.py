import torch
from torch.distributions import Normal

from swiss_roll import DATA_DIR, RUNS_DIR
from swiss_roll.utils import load_model
from swiss_roll.diffusion import DiffusionModel
from swiss_roll.guidance import GuidanceModel
from swiss_roll.scheduling import Schedular 


def sample_ddpm(model, 
                nsamples, 
                nfeatures, 
                diffusion_steps, 
                device, 
                bar_alphas, 
                alphas, 
                betas
                ):
    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
    with torch.no_grad():
        x = torch.randn(size=(nsamples, nfeatures)).to(device)
        xt = [x]
        for t in range(diffusion_steps-1, 0, -1):
            predicted_noise = model(x, torch.full([nsamples, 1], t).to(device))
            # See DDPM paper between equations 11 and 12
            x = 1 / (alphas[t] ** 0.5) * (x - (1 - alphas[t]) / ((1-bar_alphas[t]) ** 0.5) * predicted_noise)
            if t > 1:
                # See DDPM paper section 3.2.
                # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                variance = betas[t]
                std = variance ** (0.5)
                x += std * torch.randn(size=(nsamples, nfeatures)).to(device)
            xt += [x]
        return x, xt
    
def sample_ddpm_x0(
                model, 
                nsamples, 
                nfeatures, 
                diffusion_steps, 
                device, 
                bar_alphas, 
                alphas, 
                betas
                   ):
    """Sampler that uses the equations in DDPM paper to predict x0, then use that to predict x_{t-1}
    
    This is how DDPM is implemented in HuggingFace Diffusers, to allow working with models that predict
    x0 instead of the noise. It is also how we explain it in the Mixture of Diffusers paper.
    """
    with torch.no_grad():
        x = torch.randn(size=(nsamples, nfeatures)).to(device)
        for t in range(diffusion_steps-1, 0, -1):
            predicted_noise = model(x, torch.full([nsamples, 1], t).to(device))
            # Predict original sample using DDPM Eq. 15
            x0 = (x - (1 - bar_alphas[t]) ** (0.5) * predicted_noise) / bar_alphas[t] ** (0.5)
            # Predict previous sample using DDPM Eq. 7
            c0 = (bar_alphas[t-1] ** (0.5) * betas[t]) / (1 - bar_alphas[t])
            ct = alphas[t] ** (0.5) * (1 - bar_alphas[t-1]) / (1 - bar_alphas[t])
            x = c0 * x0 + ct * x
            # Add noise
            if t > 1:
                # Instead of variance = betas[t] the Stable Diffusion implementation uses this expression
                variance = (1 - bar_alphas[t-1]) / (1 - bar_alphas[t]) * betas[t]
                variance = torch.clamp(variance, min=1e-20)
                std = variance ** (0.5)
                x += std * torch.randn(size=(nsamples, nfeatures)).to(device)
        return x

def predicted_mu(eps: torch.Tensor, x_t: torch.Tensor, t: int,
                 alphas_bar: torch.Tensor) -> torch.Tensor:
    
    sqrt_ab  = torch.sqrt(alphas_bar[t])
    sqrt_one = torch.sqrt(1.0 - alphas_bar[t])
    return (x_t - sqrt_one * eps) / sqrt_ab

def normalised_grad(logp: torch.Tensor, 
                    x_t: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    
    (g,) = torch.autograd.grad(logp, x_t, create_graph=False)
    return g / (g.norm(dim=1, keepdim=True) + eps)

def sample_ddpm_guided(diffusion_model,
                      guidance_model, 
                      *, 
                      betas: torch.Tensor, 
                      alphas_bar: torch.Tensor, 
                      y: torch.Tensor, 
                      nsamples: int,
                      guidance_scale: float = 3.0, 
                      device: torch.device = "cuda"):


    T = len(betas) - 1
    nfeatures = 2
    x = torch.randn(nsamples, nfeatures, device=device) 
    trajectory = [x.clone()]
    sqrt_beta = betas.sqrt()

    for t in range(T, 0, -1):
        t_tensor = torch.full((nsamples, 1), t, device=device, dtype=torch.long)
        eps_t = diffusion_model(x, t_tensor)
        mu_t = predicted_mu(eps_t, x, t, alphas_bar)
        sigma_t = sqrt_beta[t]

        x = x.detach().requires_grad_(True)
        preds_g = guidance_model(x) 
        mu_g, logvar_g = torch.chunk(preds_g, 2, dim=-1)
        var_g = torch.exp(logvar_g).clamp_min(1e-6)
        logp = Normal(mu_g, var_g.sqrt()).log_prob(y).sum()
        g = normalised_grad(logp, x)

        if t > 1: 
            noise = torch.randn_like(x)
            x = mu_t + sigma_t * noise + guidance_scale * betas[t] * g 
        else: 
            x = mu_t 

        trajectory.append(x.clone())

    return x, trajectory


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    diffusion_model_path = DATA_DIR / 'generator.pth' 
    diffusion_model = load_model(DiffusionModel, diffusion_model_path)

    schedule = Schedular(device=device)

    guidance_model_path = RUNS_DIR / 'guidance_models' / 'run_0000' / 'guidance_model.pth'
    guidance_model = load_model(GuidanceModel, guidance_model_path)
    y_cond = torch.tensor([[1.0]], device=device, dtype=torch.float32)

    samples, traj = sample_ddpm_guided(
        diffusion_model=diffusion_model, 
        guidance_model=guidance_model,
        betas=schedule.betas,
        alphas_bar=schedule.bar_alphas,
        y=y_cond,
        nsamples=8,
        guidance_scale=3.0,
        device=device 
    ) 