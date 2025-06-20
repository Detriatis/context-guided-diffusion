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