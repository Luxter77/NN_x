import torch
import torch.nn as nn
import torch.nn.functional as F

from ..extra import SwiGLU

DEFAULT_LATENT_SIZE   = 2000
DEFAULT_INPUT_SIZE    = 768
DEFAULT_NETWORK_DEPTH = 10

class SDVAEEncoder(nn.Module):
    def __init__(self, input_features: int = DEFAULT_INPUT_SIZE, latent_shape: int = DEFAULT_LATENT_SIZE, network_depth: int = 10, dropout: float = None):
        super().__init__()
        
        self.input_features = input_features
        self.latent_shape   = latent_shape
        self.network_depth  = network_depth
        self.dropout        = dropout

        input_proj  = [nn.Linear(self.input_features, self.latent_shape), nn.Mish()]
        if self.dropout: input_proj.append(nn.Dropout(self.dropout / 2.0))
        self.input_proj = nn.Sequential(*input_proj)

        self.ff_stack = nn.ModuleList()
        for _ in range(self.network_depth):
            stack = [nn.LayerNorm(self.latent_shape), nn.Linear(self.latent_shape, self.latent_shape), SwiGLU(self.latent_shape)]
            if self.dropout:
                stack.append(nn.Dropout(self.dropout))
            self.ff_stack.append(nn.Sequential(*stack))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected x.shape = (batch, features)
        x = self.input_proj(x)
        for forward_stack in self.ff_stack:
            x = (x * 0.9) + forward_stack(x)
        return x

class SDVAEBottleneck(nn.Module):
    def __init__(self, latent_shape: int = DEFAULT_LATENT_SIZE):
        super().__init__()
        self.latent_shape = latent_shape

        self.fc_mu     = nn.Linear(self.latent_shape, self.latent_shape)
        self.fc_logvar = nn.Linear(self.latent_shape, self.latent_shape)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Expected x.shape = (batch, latent_shape)
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class SDVAEDecoder(nn.Module):
    def __init__(self, latent_shape: int = DEFAULT_LATENT_SIZE, output_features: int = DEFAULT_INPUT_SIZE, network_depth: int = DEFAULT_NETWORK_DEPTH, dropout: float = None):
        super().__init__()
        
        self.latent_shape    = latent_shape
        self.output_features = output_features
        self.network_depth   = network_depth
        self.dropout         = dropout

        self.ff_stack = nn.ModuleList()
        for _ in range(self.network_depth):
            stack = [nn.LayerNorm(self.latent_shape), nn.Linear(self.latent_shape, self.latent_shape), SwiGLU(self.latent_shape)]
            if self.dropout: stack.append(nn.Dropout(self.dropout * 1.5))
            self.ff_stack.append(nn.Sequential(*stack))

        output_proj = [nn.Linear(self.latent_shape, self.output_features), nn.Mish()]
        if self.dropout: output_proj.append(nn.Dropout(self.dropout / 2.0))
        self.output_proj = nn.Sequential(*output_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected x.shape = (batch, features)
        for forward_stack in self.ff_stack:
            x = (x * 0.9) + forward_stack(x)
        return self.output_proj(x)

class SkipDepthVAE(nn.Module):
    def __init__(self, features: int = DEFAULT_INPUT_SIZE, latent_shape: int = DEFAULT_LATENT_SIZE, network_depth: int = DEFAULT_NETWORK_DEPTH, dropout: float = None):
        super().__init__()

        self.features      = features
        self.latent_shape  = latent_shape
        self.network_depth = network_depth
        self.dropout       = dropout
        
        self.encoder    = SDVAEEncoder(features, latent_shape, network_depth, dropout)
        self.bottleneck = SDVAEBottleneck(latent_shape)
        self.decoder    = SDVAEDecoder(latent_shape, features, network_depth, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Expected x.shape = (batch, features)
        h          = self.encoder(x)
        mu, logvar = self.bottleneck(h)

        std = torch.exp(0.5 * logvar)
        z   = mu + torch.randn_like(std) * std

        recon_x = self.decoder(z)

        return recon_x, h, mu, logvar