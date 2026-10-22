import torch
import torch.nn as nn
from typing import Dict, List

class WeightCalibrationEngine:
    """Calculates activation scales and optimal quantization parameters across transformer channels."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_stats: Dict[str, torch.Tensor] = {}

    def register_calibration_hooks(self) -> List:
        hooks = []
        def get_activation_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    max_val = output.detach().abs().reshape(-1, output.shape[-1]).max(dim=0)[0]
                    if name not in self.activation_stats:
                        self.activation_stats[name] = max_val
                    else:
                        self.activation_stats[name] = torch.maximum(self.activation_stats[name], max_val)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(get_activation_hook(name)))
        return hooks

    def compute_optimal_scales(self, num_bits: int = 4) -> Dict[str, torch.Tensor]:
        scales = {}
        qmax = (2 ** (num_bits - 1)) - 1
        for name, stat in self.activation_stats.items():
            scales[name] = stat / float(qmax)
        return scales
