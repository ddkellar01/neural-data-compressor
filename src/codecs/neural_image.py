import torch
import torch.nn as nn

class ConvAnalysisTransform(nn.Module):
    """Encoder: Transforms input data into continuous latent representation."""
    def __init__(self, in_channels=3, latent_channels=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.GDN(latent_channels),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.GDN(latent_channels),
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
            nn.GDN(latent_channels, inverse=True),
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.GDN(latent_channels, inverse=True),
            nn.ConvTranspose2d(latent_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, y_hat):
        return self.network(y_hat)
