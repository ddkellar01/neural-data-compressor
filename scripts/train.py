import torch
from torch.optim import Adam
from src.codecs.neural_image import ConvAnalysisTransform, ConvSynthesisTransform
from src.entropy.probability_model import GaussianEntropyModel
from src.losses.rate_distortion import RateDistortionLoss

def train_one_epoch(encoder, decoder, entropy_model, dataloader, optimizer, criterion, device):
    encoder.train()
    decoder.train()
    for batch in dataloader:
        x = batch.to(device)
        optimizer.zero_grad()
        y = encoder(x)
        y_hat = torch.round(y)  # Continuous-to-discrete latent
        x_hat = decoder(y_hat)
        pmf = entropy_model(y_hat)
        loss, dist, rate = criterion(x, x_hat, pmf)
        loss.backward()
        optimizer.step()
    print(f"Epoch Loss: {loss.item():.4f} | Distortion: {dist:.4f} | Rate: {rate:.4f}")
