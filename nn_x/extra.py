from typing import Optional

import torch
import torch.nn as nn

def count_parameters(model: nn.Module, dtype: Optional[torch.dtype] = None) -> int:
    # Count the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print(f"Total number of parameters (in millions): {total_params / 1e6} M")
    # Estimate the size of the model in RAM (approximate)
    # If dtype is provided, use its element size for more accurate estimation
    # otherwise each parameter is typically a 32-bit float (4 bytes)
    estimated_size_gb = total_params * (dtype if dtype else 4) / (1024 ** 3)
    print(f"Estimated size of the model in RAM: {estimated_size_gb:.2f} GB")
    return total_params

def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SwiGLU(nn.Module):
    """
    Implements the SwiGLU activation function as proposed in the paper
    'GLU Variants Improve Transformer' (https://arxiv.org/abs/2002.05202v1).

    The SwiGLU activation is defined as: SwiGLU(x, W, V) = Swish(xW) * (xV)
    where Swish(x) = x * sigmoid(x), implemented here using nn.SiLU.

    This module uses a single linear layer to compute both projections (xW and xV)
    by projecting the input to 2 * dim, then splitting the result.
    """
    def __init__(self, dim: int, bias: bool = True):
        """
        Initializes the SwiGLU module.

        Args:
            dim (int): The input/output dimension.
            bias (bool): Whether to include bias terms in the linear projection.
                         Default: True.
        """
        super().__init__()

        self.dim  = dim
        self.bias = bias

        self.linear = nn.Linear(dim, 2 * dim, bias=bias)

        # Swish activation function (x * sigmoid(x))
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SwiGLU activation.

        Args:
            x (torch.Tensor): Input tensor of shape (*, dim), where * means
                              any number of leading dimensions.

        Returns:
            torch.Tensor: Output tensor of shape (*, dim).
        """
        # Project the input using the linear layer
        # Output shape: (*, 2 * dim)
        projected_x = self.linear(x)

        # Split the projected tensor into two halves along the last dimension (features)
        # xW and xV will each have shape (*, dim)
        xW, xV = projected_x.chunk(2, dim=-1)

        # Apply the Swish activation to the first half (the gate)
        gate = self.swish(xW)

        # Element-wise multiply the activated gate with the second half
        # Output shape: (*, dim)
        output = gate * xV

        return output
