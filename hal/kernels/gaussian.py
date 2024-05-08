# gaussian.py

import math
import torch

__all__ = ['Gaussian', 'RFFGaussian']


class Gaussian:

    def __init__(self, sigma=None):
        self.sigma = sigma

    def __call__(self, x, y=None):
        if y is None:
            if self.sigma is None:
                dist = torch.cdist(x, x, p=2) ** 2
                dist_u = torch.triu(dist, diagonal=1)
                if (dist_u > 0).sum():
                    sigma = torch.sqrt(0.5 * torch.median(dist_u[dist_u > 0]))
                else:
                    sigma = 1.0
            else:
                dist = torch.cdist(x, x, p=2) ** 2
                sigma = self.sigma
        else:
            dist = torch.cdist(x, y, p=2) ** 2
            if self.sigma is None:
                sigma = torch.sqrt(0.5 * torch.median(dist[dist > 0]))
            else:
                sigma = self.sigma

        kernel = torch.exp(-dist / (2*sigma**2))
        return kernel


class RFFGaussian:

    def __init__(self, sigma=None, rff_dim=200, sigma_numel_max=9000):
        self.sigma = sigma
        self.rff_dim = rff_dim
        self.numel_max = sigma_numel_max
        self.w = None
        self.b = None

    def _calc_w_b(self, x):
        dim_x = x.shape[1]
        if self.sigma is None:
            n = min(self.numel_max, x.shape[0])
            rand = torch.randperm(n)
            x_samp = x[rand, :]
            x_samp = x_samp[0: n, :]
            dist = torch.cdist(x_samp, x_samp, p=2) ** 2
            dist = torch.triu(dist, diagonal=1)
            if (dist > 0).sum(): 
                sigma = torch.sqrt(0.5 * torch.median(dist[dist > 0]))
            else:
                sigma = 1

        else:
            sigma = self.sigma
        mu_x = torch.zeros(dim_x, device=x.device, dtype=x.dtype)
        sigma_x = torch.eye(dim_x, device=x.device, dtype=x.dtype) / (sigma ** 2)
        px = torch.distributions.MultivariateNormal(mu_x, sigma_x)
        self.w = px.sample((self.rff_dim,))
        p = torch.distributions.uniform.Uniform(torch.tensor([0.0], device=x.device, dtype=x.dtype), 2 * torch.tensor([math.pi], device=x.device, dtype=x.dtype))
        self.b = p.sample((self.rff_dim,)).squeeze(1)
        

    def __call__(self, x):
        if self.w is None or self.b is None:
            self._calc_w_b(x)

        x = x.to(dtype=self.w.dtype)
        device = x.device

        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        try:
            phi_x = math.sqrt(2 / self.rff_dim) * torch.cos(torch.mm(x, self.w.to(device=device).t()) + self.b.to(device=device))
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

        return phi_x.to(dtype=self.w.dtype)



