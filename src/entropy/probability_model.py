import torch
import torch.nn as nn

class GaussianEntropyModel(nn.Module):
    """Estimates symbol probability distribution using Gaussian prior."""
    def __init__(self, channels=128):
        super().__init__()
        self.location = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, y):
        sigma = torch.clamp(self.scale.abs(), min=1e-6)
        cdf_upper = 0.5 * (1.0 + torch.erf((y + 0.5 - self.location) / (sigma * 1.41421356)))
        cdf_lower = 0.5 * (1.0 + torch.erf((y - 0.5 - self.location) / (sigma * 1.41421356)))
        pmf = torch.clamp(cdf_upper - cdf_lower, min=1e-9)
        return pmf
