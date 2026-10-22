import torch

class UniformQuantizer:
    """Quantizes model weights into low-bit integers (e.g., 4-bit or 8-bit)."""
    def __init__(self, num_bits=4):
        self.num_bits = num_bits
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = (2 ** (num_bits - 1)) - 1

    def quantize(self, tensor: torch.Tensor):
        scale = tensor.abs().max() / float(self.qmax)
        scale = torch.clamp(scale, min=1e-8)
        quantized = torch.round(tensor / scale).clamp(self.qmin, self.qmax)
        return quantized.to(torch.int8), scale

    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor):
        return quantized.float() * scale
