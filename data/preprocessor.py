import torch
from torchvision import transforms

def get_image_transforms(img_size=256):
    """Builds preprocessing pipeline for image tensor inputs."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def add_quantization_noise(tensor: torch.Tensor, training=True) -> torch.Tensor:
    """Applies uniform noise during training to simulate discretization."""
    if training:
        noise = torch.rand_like(tensor) - 0.5
        return tensor + noise
    return torch.round(tensor)
