import torch
import torch.nn as nn
import re
from .resampler import Resampler
from .ldp import LDPNetV2Projector
from .abstract import CAbstractor


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if 'abs' in projector_type:
        config.encoder_hidden_size = config.hidden_size // 4
        config.num_eos_tokens = 0
        config.output_hidden_size = 4096
        config.prenorm = False
        config.pos_emb = True
        config.depth = 3
        config.mlp_depth = 2
        config.num_query_tokens = 144
        config.initializer_range = 0.02
        return CAbstractor(config=config,num_input_tokens=config.num_input_tokens)
    
    if 'resampler' in projector_type:
        embed_dim = config.hidden_size
        return Resampler(12,embed_dim,embed_dim//128,kv_dim=config.vision_embed_dim)
    
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'ldp':
        return LDPNetV2Projector(config)
    
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
