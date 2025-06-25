import torch

class Schedular():
    def __init__(self, T=40, s=0.008):
        self.T = T 
        self.s = s 
        self.schedule = self.cosine_schedule(T, s) 
        self.bar_alphas = self.get_bar_alphas(self.schedule)
        self.alphas = self.get_alphas(self.bar_alphas)
        self.betas = self.get_betas(self.bar_alphas)

    def noise(self, x, t):
        eps = torch.randn_like(x)
        mu = (self.bar_alphas[t] ** 0.5).repeat(1, x.shape[1]) * x
        var = ((1 - self.bar_alphas[t]) ** 0.5).repeat(1, x.shape[1]) * eps
        return mu + var, eps

    def cosine_schedule(self, T:int, s:float) -> torch.Tensor:
        t = torch.arange(0, T, 1, dtype=torch.float32)
        schedule = torch.cos((t/T+ s) / (1 + s) * torch.pi / 2) ** 2
        return schedule 

    def get_bar_alphas(self, schedule):
        return schedule / schedule[0] 

    def get_alphas(self, bar_alphas):
        return  bar_alphas / torch.concat([bar_alphas[0:1], bar_alphas[0:-1]])

    def get_betas(self, bar_alphas): 
        return 1 - (bar_alphas / torch.concat([bar_alphas[0:1], bar_alphas[:-1]])) 

    def randtimesteps(self, X):
        return torch.randint(0, self.T, size=[len(X), 1])