import torch
import torch.nn as nn

class GDN(nn.Module):
    """Generalized Divisive Normalization (GDN) layer for neural image compression."""
    def __init__(self, channels: int, inverse: bool = False, beta_min: float = 1e-6, gamma_init: float = 0.1):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.beta = nn.Parameter(torch.ones(channels) * 0.5)
        self.gamma = nn.Parameter(torch.eye(channels) * gamma_init)

    def forward(self, x):
        _, c, _, _ = x.shape
        beta = torch.clamp(self.beta, min=self.beta_min)
        gamma = torch.abs(self.gamma)
        
        # Compute local normalization denominator
        norm = torch.einsum('bchw,cd->bdhw', x ** 2, gamma) + beta.view(1, -1, 1, 1)
        if self.inverse:
            return x * torch.sqrt(norm)
        else:
            return x / torch.sqrt(norm)

class ConvAnalysisTransform(nn.Module):
    """Encoder: Transforms input data into continuous latent representation."""
    def __init__(self, in_channels=3, latent_channels=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            GDN(latent_channels),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            GDN(latent_channels),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.network(x)

class ConvSynthesisTransform(nn.Module):
    """Decoder: Reconstructs raw data from compressed latent representation."""
    def __init__(self, out_channels=3, latent_channels=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(latent_channels, inverse=True),
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(latent_channels, inverse=True),
            nn.ConvTranspose2d(latent_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, y_hat):
        return self.network(y_hat)
