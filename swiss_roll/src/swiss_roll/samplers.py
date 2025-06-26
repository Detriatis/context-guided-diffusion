import torch

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
    
    g = torch.autograd.grd(logp, x_t, create_graph=False)
    return g / g.norm(dim=1, keemdim=True) + eps

@torch.no_grad()
def sample_ddpm_guided(diffusion_model,
                      guidance_model, 
                      *, 
                      betas: torch.Tensor, 
                      alphas_bar: torch.Tensor, 
                      y: torch.Tensor, 
                      nsamples: int,
                      guidance_scale: float = 3.0, 
                      device: torch.device = "cuda"):

    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
    T = len(betas) - 1

    
    x = torch.randn(size=(nsamples, nfeatures)).to(device)
    xt = [x]
    for t in range(diffusion_steps-1, 0, -1):
        with torch.no_grad():
            predicted_noise = model(x, torch.full([nsamples, 1], t).to(device))
            # See DDPM paper between equations 11 and 12
            variance = betas[t]
            std = variance ** (0.5)
            mu = predicted_mu(predicted_noise, x, t)
        if t > 1:
            x.requires_grad_(True) 
            mu_g, r = guidance_model(x)            
            var_g = torch.exp(r).clamp_min(1e-6)
            
            # NLL Loss
            dist = torch.distributions.Normal(mu_g, var_g ** 0.5)
            nll = dist.log_prob(target_y * torch.ones_like(mu_g)).sum()
            nll.backward()
            g = x.grad

            # See DDPM paper section 3.2.
            # Choosing the variance through beta_t is optimal for x_0 a normal distribution
            x = mu + std * torch.randn(size=(nsamples, nfeatures)).to(device) + guidance_scale * variance * g
        else: 
            x = mu 
        xt += [x]
    return x, xt