import torch
import torch.nn as nn

class RateDistortionLoss(nn.Module):
    """Loss function: Loss = Distortion + (lambda * Rate)."""
    def __init__(self, lmbda=0.01):
        super().__init__()
        self.lmbda = lmbda
        self.mse = nn.MSELoss()

    def forward(self, original, reconstructed, pmf):
        distortion = self.mse(reconstructed, original)
        # BPP estimation using Shannon entropy calculation
        rate = -torch.log2(pmf).sum() / (original.size(0) * original.size(2) * original.size(3))
        total_loss = distortion + self.lmbda * rate
        return total_loss, distortion.item(), rate.item()
