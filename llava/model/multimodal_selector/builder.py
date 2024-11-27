import torch
import torch.nn as nn
import re

from .selector_mlp import SelectorProjector
from .selector_attn import SelectorAttn

import warnings
warnings.filterwarnings("ignore")

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_selector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_selector(config, delay_load=False, **kwargs):
    selector_type = getattr(config, "mm_selector_type", "linear")

    if selector_type == "linear":
        return nn.Linear(config.mm_selector_size, 1)
    if selector_type == "selector_mlp":
        mlp_type = "mlp2x_gelu"
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mlp_type)
        return SelectorProjector(mlp_gelu_match, config)
    if selector_type == "selector_attn":
        return SelectorAttn(config)
    # if mlp_gelu_match:
    #     mlp_depth = int(mlp_gelu_match.group(1))
    #     modules = [nn.Linear(config.mm_selector_size, config.hidden_size), nn.GELU()]
    #     for _ in range(1, mlp_depth - 1):
    #         modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    #         modules.append(nn.GELU())
        
    #     modules.append(nn.Linear(config.hidden_size, 1))

    #     return nn.Sequential(*modules)  
    
    # TODO: have to add resnet block
    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", selector_type)

    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_selector_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if selector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown sampler type: {selector_type}")
