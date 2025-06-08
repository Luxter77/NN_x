import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..extra import SwiGLU

# The current implementation of the MoE forward pass has a potential bottleneck due to the loop iterating through experts and applying a mask.
# This can lead to non-contiguous memory access and loop overhead.
# Solution:  Replace the loop with `torch.scatter_add` or similar operations for significantly improved performance.
# This involves pre-allocating the output tensor and then using scatter operations to accumulate the results from experts based on the gate's output indices.

class MoEGate(nn.Module):
    """
    A Mixture-of-Experts (MoE) Gate module.

    This module determines which experts should process each token based on the
    input hidden states. It calculates routing scores for each expert and selects
    the top-k experts using either a greedy or a group-limited greedy strategy.

    Args:
        num_experts_per_input (int): The number of experts to select for each input token (top_k).
        n_routed_experts (int): The total number of experts available for routing.
        routed_scaling_factor (int): A scaling factor often applied to the routed inputs
                                      (Note: Not directly used in this gate's forward pass,
                                      might be used in the main MoE layer).
        topk_method (str): The method for selecting top-k experts.
                           Supported: "greedy", "group_limited_greedy".
        n_group (int): The number of expert groups (used only if topk_method="group_limited_greedy").
        topk_group (int): The number of groups to select from (used only if topk_method="group_limited_greedy").
        hidden_size (int): The dimensionality of the input hidden states.
    """
    def __init__(self, num_experts_per_input: int, n_routed_experts: int, routed_scaling_factor: int, topk_method: str, n_group: int, topk_group: int, hidden_size: int):
        super().__init__()

        # --- Configuration ---
        self.top_k                 = num_experts_per_input
        self.n_routed_experts      = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor # Note: Not used in forward, potentially used elsewhere
        self.topk_method           = topk_method
        self.n_group               = n_group
        self.topk_group            = topk_group
        self.hidden_size           = hidden_size

        # --- Validation ---
        if self.topk_method == "group_limited_greedy":
            if self.n_routed_experts % self.n_group != 0:
                raise ValueError("n_routed_experts must be divisible by n_group for group_limited_greedy")
            if self.topk_group > self.n_group:
                raise ValueError("topk_group cannot be larger than n_group")

        # --- Parameters ---
        # Learnable weight matrix for projecting hidden states to expert logits.
        # Shape: (n_routed_experts, hidden_size)
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))

        # --- Initialization ---
        # Initialize weights using Kaiming uniform initialization for better training dynamics.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the top-k expert indices and their corresponding routing weights (scores).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - topk_idx (torch.Tensor): Indices of the selected top-k experts for each token.
                                           Shape: (batch_size * sequence_length, top_k).
                - topk_weight (torch.Tensor): Routing weights (softmax scores) for the selected experts.
                                              Shape: (batch_size * sequence_length, top_k).
        """
        _batch_size, _seq_len, hidden_dim = x.shape

        # Flatten the batch and sequence dimensions for independent processing of each token.
        # Shape: (batch_size * sequence_length, hidden_size)
        hidden_states = x.view(-1, hidden_dim)
        num_tokens = hidden_states.shape[0] # Same as batch_size * seq_len

        # --- Calculate Routing Logits and Scores ---
        # Project hidden states to get logits for each expert.
        # Using float32 for stability, especially in mixed-precision training.
        # Shape: (num_tokens, n_routed_experts)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32)) # pylint: disable=not-callable

        # Apply softmax to convert logits into probability scores (routing weights).
        # Shape: (num_tokens, n_routed_experts)
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        # --- Select Top-k Experts ---
        if self.topk_method == "greedy":
            # Simple greedy selection: Choose the 'top_k' experts with the highest scores.
            # sorted=False can be slightly faster if order doesn't matter downstream.
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        elif self.topk_method == "group_limited_greedy":
            # Group-limited greedy selection:
            # 1. Divide experts into 'n_group' groups.
            # 2. Find the highest score *within* each group for each token.
            # 3. Select the top 'topk_group' groups based on these max scores.
            # 4. Select the top 'top_k' experts *only from the selected groups*.

            experts_per_group = self.n_routed_experts // self.n_group

            # Reshape scores to (num_tokens, n_group, experts_per_group)
            scores_grouped = scores.view(num_tokens, self.n_group, experts_per_group)

            # Find the maximum score within each group for each token.
            # Shape: (num_tokens, n_group)
            group_max_scores = scores_grouped.max(dim=-1).values

            # Select the top 'topk_group' groups based on their max scores.
            # group_topk_indices shape: (num_tokens, topk_group)
            _, group_topk_indices = torch.topk(group_max_scores, k=self.topk_group, dim=-1, sorted=False)

            # Create a mask to identify the selected groups.
            # Shape: (num_tokens, n_group)
            group_mask = torch.zeros_like(group_max_scores, device=scores.device) # Use same device
            # Scatter '1's into the mask at the indices of the selected groups.
            group_mask.scatter_(1, group_topk_indices, 1)

            # Expand the group mask to match the original score dimensions,
            # effectively creating a mask for individual experts based on group selection.
            # Shape: (num_tokens, n_routed_experts)
            expert_mask = (
                group_mask.unsqueeze(-1) # (num_tokens, n_group, 1)
                .expand(num_tokens, self.n_group, experts_per_group) # (num_tokens, n_group, experts_per_group)
                .reshape(num_tokens, -1) # (num_tokens, n_routed_experts)
            )

            # Apply the mask: Zero out scores of experts in non-selected groups.
            # We use `masked_fill` with the *inverted* boolean mask.
            # Shape: (num_tokens, n_routed_experts)
            masked_scores = scores.masked_fill(~expert_mask.bool(), 0.0)

            # Select the top 'top_k' experts from the masked scores.
            # These will necessarily come from the initially selected groups.
            topk_weight, topk_idx = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)

        else:
            raise ValueError(f"Unsupported topk_method: {self.topk_method}")

        # Return the indices and weights of the chosen experts for each token.
        return topk_idx, topk_weight

class MoE(nn.Module):
    """
    Mixture-of-Experts layer.

    Routes input tokens to a subset of 'routed' experts based on a learned gate
    and combines their outputs. Optionally includes 'shared' experts whose output
    is added to all tokens.

    Args:
        dim (int): Input and output dimension of the layer.
        routed_scaling_factor (int): Scaling factor potentially used by the gate
                                      (passed to MoEGate).
        topk_method (str): Method used by the gate for selecting experts
                           ("greedy", "group_limited_greedy").
        n_group (int): Number of expert groups for group-limited gating.
        topk_group (int): Number of groups to select from for group-limited gating.
        hidden_dim (int | None): Intermediate hidden dimension for the MLP experts.
                                  If None, defaults to a calculated value (e.g., 4*dim).
        n_routed_experts (int): Total number of 'routed' experts. Defaults to 12.
        num_experts_per_tok (int): Number of experts to route each token to (k).
                                   Defaults to 4.
        n_shared_experts (int): Scales the hidden dimension of the *single* shared
                                expert MLP block. If > 1, it creates one larger
                                shared MLP, not multiple separate ones. Defaults to 2.
        expert (str): Type of expert block to use (currently hardcoded to use SwiGLU
                   via `expert_block`). Defaults to "swiglu".
    """
    def __init__(self, dim: int, routed_scaling_factor: int, topk_method: str, n_group: int, topk_group: int,
                 hidden_dim: Optional[int] = None, n_routed_experts: int = 12, num_experts_per_tok: int = 4,
                 n_shared_experts: int = 2, expert: Union[str, nn.Module] = "swiglu"
    ):
        super().__init__()

        # --- Configuration ---
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Note: This scales hidden_dim of the shared expert
        self.n_shared_experts = n_shared_experts # Controls shared MLP hidden size

        # --- Determine MLP Block Type ---
        # TODO: Add the rest :3c
        if isinstance(expert, nn.Module):
            expert_block = expert
        elif expert == "swiglu":
            expert_block = SwiGLU
        elif expert == "linear":
            expert_block = nn.Linear
        else:
            raise ValueError(f"Unsupported MLP type: {expert}. Currently only 'swiglu' is supported.")

        # --- Calculate Hidden Dimension ---
        # Use provided hidden_dim or calculate a default
        # (Using SwiGLU's default calculation here as an example)
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            # Ensure hidden_dim is a multiple of a value (common practice)
            multiple_of = 256 # Example value, adjust as needed
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # --- Module Instantiation ---
        # Create the bank of 'routed' experts
        self.experts = nn.ModuleList([expert_block(dim, hidden_dim) for _ in range(n_routed_experts)])

        # Create the gating mechanism
        self.gate = MoEGate(
            num_experts_per_input=num_experts_per_tok,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            topk_method=topk_method,
            n_group=n_group,
            topk_group=topk_group,
            hidden_size=dim,
        )

        # Create the 'shared' expert block(s)
        # Note: The current implementation uses n_shared_experts to scale the
        # hidden dimension of a SINGLE shared MLP block.
        # If n_shared_experts > 0, create the shared block.
        if self.n_shared_experts > 0:
            shared_hidden_dim = hidden_dim * self.n_shared_experts
            self.shared_experts = expert_block(dim, shared_hidden_dim)
            # print(f"Initialized shared expert with dim={dim}, hidden_dim={shared_hidden_dim}") # Debug print
        else:
            self.shared_experts = None
            # print("No shared experts initialized.") # Debug print


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Mixture-of-Experts layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        # Store input for residual connection and original shape
        identity = x
        batch_size, seq_len, dim = x.shape
        num_tokens = batch_size * seq_len

        # --- Gating ---
        # Get top-k expert indices and their weights (scores) for each token
        # topk_idx shape: (num_tokens, num_experts_per_tok)
        # topk_weight shape: (num_tokens, num_experts_per_tok)
        topk_idx, topk_weight = self.gate(x)

        # --- Routed Expert Processing ---
        # Reshape input for expert processing: (num_tokens, dim)
        x_flat = x.view(num_tokens, dim)

        # Flatten the expert indices: (num_tokens * num_experts_per_tok)
        flat_topk_idx = topk_idx.view(-1)

        # Repeat input tokens based on how many experts they are routed to (k times)
        # Shape: (num_tokens * num_experts_per_tok, dim)
        # This prepares the input for indexed processing by experts.
        x_repeated = x_flat.repeat_interleave(self.num_experts_per_tok, dim=0)

        # Pre-allocate output tensor for routed experts
        y = torch.empty_like(x_repeated)

        # Process tokens with their assigned experts
        # This loop iterates through each expert and processes the tokens assigned to it.
        # Note: This can be a bottleneck due to potential non-contiguous memory access
        # and loop overhead. Optimized implementations often use scatter operations.
        for i, expert in enumerate(self.experts):
            # Create a mask for tokens assigned to the current expert 'i'
            mask = flat_topk_idx == i
            if mask.any(): # Only process if there are tokens for this expert
                # Select input tokens for expert 'i', process them, and store in 'y'
                y[mask] = expert(x_repeated[mask]).to(dtype=x.dtype) # Ensure output dtype matches input

        # --- Combine Routed Expert Outputs ---
        # Reshape y back to (num_tokens, num_experts_per_tok, dim)
        y = y.view(num_tokens, self.num_experts_per_tok, dim)

        # Weight the expert outputs by the gate scores and sum them
        # topk_weight shape: (num_tokens, num_experts_per_tok) -> unsqueeze adds dim -> (num_tokens, num_experts_per_tok, 1)
        # Weighted sum results in shape: (num_tokens, dim)
        y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)

        # Reshape combined output back to the original input shape
        # Shape: (batch_size, sequence_length, dim)
        y = y.view(batch_size, seq_len, dim)

        # --- Shared Expert Processing & Final Output ---
        # Process original input through shared experts (if they exist)
        if self.shared_experts is not None:
            shared_output = self.shared_experts(identity)
            # Add shared expert output to the combined routed expert output
            output = y + shared_output
        else:
            # If no shared experts, the output is just the combined routed output
            output = y

        # The original code added shared output to y (routed output),
        # implying no residual connection from the *original* input `identity`
        # to the final output *unless* shared_experts are used.
        # If a final residual `identity + output` is desired regardless of shared experts,
        # it should be added here. Assuming the original logic is intended:
        return output
