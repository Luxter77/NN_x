from typing import List
from itertools import pairwise

import torch
import torch.nn as nn

import numpy as np

from .MoE import MoE
from ..extra import SwiGLU

class MoEFF(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_ff: int, num_experts: int):
        super().__init__()

        self.ff = nn.ModuleList()

        # Input
        self.ff.append(nn.Linear(in_features, out_features))
        self.ff.append(SwiGLU(out_features))

        # MoEs
        self.ff.append(MoE(
            dim=out_features,
            routed_scaling_factor=1,       # Start simple
            topk_method="greedy",          # Start simple
            n_group=4,                     # Not used with greedy, but provide a value
            topk_group=2,                  # Not used with greedy
            hidden_dim=None,               # Use default calculation based on dim
            n_routed_experts=num_experts,
            num_experts_per_tok=4,         # Common value for k
            n_shared_experts=2,            # Include two shared expert block initially
            expert="swiglu",
        ))

        # FFs
        for _ in range(max(num_ff - 1, 0)):
            self.ff.append(nn.Linear(out_features, out_features))
            self.ff.append(SwiGLU(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.ff: x = layer(x)
        return x

class LinFF(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_ff: int, num_experts: int):
        super().__init__()

        self.ff = nn.ModuleList()

        # Input
        self.ff.append(nn.Linear(in_features, out_features))
        self.ff.append(SwiGLU(out_features))

        # Linears
        for _ in range(num_experts):
            self.ff.append(nn.Linear(out_features, out_features))
            self.ff.append(SwiGLU(out_features))
        
        # FFs
        for _ in range(max(num_ff - 1, 0)):
            self.ff.append(nn.Linear(out_features, out_features))
            self.ff.append(SwiGLU(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.ff: x = layer(x)
        return x

class UNetTFF(nn.Module):
    def __init__(self, num_features: int = 768, num_funnel: int = 7, num_ff: int = 3, num_experts: int = 16, bottleneck_dim: int = 27, moe: bool = True):
        super().__init__()
        self.num_features   = num_features
        self.num_funnel     = num_funnel + 1
        self.num_ff         = num_ff
        self.num_experts    = num_experts
        self.bottleneck_dim = bottleneck_dim
        self.moe            = moe

        tff_block_type = MoEFF if self.moe else LinFF

        encoder_dimentions = np.linspace(self.num_features, self.bottleneck_dim, self.num_funnel, dtype=int)
        # default -> Encoder dimensions: [768 662 556 450 344 238 132  27]

        decoder_dimentions = np.linspace(self.bottleneck_dim, self.num_features, self.num_funnel, dtype=int) * 2
        # default -> Decoder dimensions: [ 54 264 476 688 900 1112 1324 1536]

        self.encoder = nn.ModuleList()
        for in_features, out_features in pairwise(encoder_dimentions):
            self.encoder.append(tff_block_type(in_features, out_features, self.num_ff, self.num_experts))

        self.bottleneck = tff_block_type(self.bottleneck_dim, self.bottleneck_dim, self.num_ff, self.num_experts)

        self.decoder = nn.ModuleList()
        for in_features, out_features in pairwise(decoder_dimentions):
            self.decoder.append(tff_block_type(in_features, out_features, self.num_ff, self.num_experts))
        
        # Add the final layer to match the original number of features
        self.decoder.append(nn.Linear(decoder_dimentions[-1], self.num_features))
        self.decoder.append(SwiGLU(self.num_features))

        self.bos = nn.Parameter(torch.randn(1, self.num_features))
        self.eos = nn.Parameter(torch.randn(1, self.num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        x = self.encoder_forward(x, skip_connections)

        x = self.bottleneck(x)

        x = self.decoder_forward(x, skip_connections)

        return x

    def encoder_forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]):
        for encoder_layer in self.encoder:
            try:
                x = encoder_layer(x)
                skip_connections.append(x)
            except Exception as e:
                print(f"Error in encoder {encoder_layer} layer: {e}")
                print(f"x shape: {x.shape}")
                raise e
        return x

    def decoder_forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]):
        for decoder_layer, skip_connection in zip(self.decoder, skip_connections[::-1]):
            x = torch.cat([x, skip_connection], dim=1)
            try:
                x = decoder_layer(x)
            except Exception as e:
                print(f"Error in decoder {decoder_layer} layer: {e}")
                print(f"skip_connection shape: {skip_connection.shape}")
                print(f"x shape: {x.shape}")
                raise e
        return x
