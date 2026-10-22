import math
import torch

def calculate_psnr(mse_value: float, max_val: float = 1.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    if mse_value == 0:
        return float('inf')
    return 20 * math.log10(max_val) - 10 * math.log10(mse_value)

def calculate_bpp(num_bits: int, num_pixels: int) -> float:
    """Computes Bits-Per-Pixel (BPP) metric."""
    return num_bits / float(num_pixels)
