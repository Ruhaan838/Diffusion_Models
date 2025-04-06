import torch
from torch import nn

class GaussianDiffusion:
    def __init__(self, noise_steps=1000, st_beta=1e-4, end_beta=0.02, image_size=64, device='cpu'):
        
        self.noise_steps = noise_steps
        self.betas = torch.linspace(st_beta, end_beta, noise_steps)
        self.alpha = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.image_size = image_size
        self.device = device
        
    def noise_sample(self, x_0, t):
        alpha_t = torch.sqrt(self.alpha_hat[t])
        one_min_alpha = torch.sqrt(1 - self.alpha_hat[t])
        noise = torch.randn_like(x_0)
        return alpha_t * x_0 + one_min_alpha * noise, noise
    
    def sample_t(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    #this fllow the paper's algorithm 2
    @torch.no_grad()
    def sampling(self, model: nn.Module, n, labels=None, cfg_scale=3):
        model.eval()
        x = torch.randn(n, 3, self.image_size, self.image_size).to(self.device)
        for i in reversed(range(1, self.noise_steps)):
            t = (torch.ones(n) * i).long().to(self.device)
            if cfg_scale > 0 and labels is not None:
                uncondi = model(x, t, None)
                pred = model(x, t, labels)
                pred = torch.lerp(uncondi, pred, cfg_scale)
            else:
                pred = model(x, t, labels)
                
            alpha_t = self.alpha[t]
            alpha_t_hat = self.alpha_hat[t]
            beta_t = self.betas[t]
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_t_hat) * pred) + torch.sqrt(beta_t) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).to(torch.uint8)
        return x

