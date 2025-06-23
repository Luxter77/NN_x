from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, tensor

def vocab_aware_embedding_loss(predicted_embedding: Tensor, true_token_indexes: Tensor,
                               embedding_layer: nn.Embedding, vocab_penalty_weight: float = 1.0,
                               distance_function: str = "cosine") -> Tensor:
    """
    Calculates a loss that penalizes predictions far from the true token
    and far from the vocabulary embedding space.
    Args:
        - predicted_embedding (Tensor): Predicted embedding (batch_size, embedding_dim).
        - true_token_indexes (Tensor): Index of the true token (batch_size,).
        - embedding_layer (torch.nn.Embedding): The embedding layer itself.
        - vocab_penalty_weight (float): Weight for the vocabulary penalty term.
    Returns: Tensor: The calculated loss (scalar).
    """
    true_embedding = embedding_layer(true_token_indexes).detach() # (batch_size, embedding_dim)

    # 1. Distance to Correct Token Embedding
    if   distance_function == "cosine":
        # cosine distance
        distance_to_true = 1 - ((predicted_embedding / predicted_embedding.norm(dim=1, keepdim=True)) * (true_embedding / true_embedding.norm(dim=1, keepdim=True))).sum(dim=1).mean()
    elif distance_function == "l2":
        # L2 Distance
        distance_to_true = torch.cdist(predicted_embedding, true_embedding).mean() # Calculate mean over batch

    # 2. Minimum Distance to Any Vocabulary Embedding
    # Calculate distances from predicted_embedding to ALL embeddings in emb.weight
    distances_to_vocab = torch.cdist(predicted_embedding, embedding_layer.weight.detach()) # (batch_size, num_embeddings)
    # Find the minimum distance for each predicted embedding (along dimension 1 - vocab embeddings)
    min_dist_to_vocab = distances_to_vocab.min(dim=1).values  # (batch_size,)
    vocab_penalty = min_dist_to_vocab.mean() # Mean over batch

    # Total Loss
    loss = distance_to_true + vocab_penalty_weight * vocab_penalty
    return loss

def decorrelation_loss(z: Tensor) -> Tensor:
    b, d = z.shape
    if b <= 1: return tensor(0.0, device=z.device)
    scale = (d + 1 / d) # I know its - 1 but the difference is minimal and i dont like it when / 0
    corref = z.T.corrcoef().nan_to_num()
    mask   = ~torch.eye(d, device=z.device).bool()
    loss   = corref[mask].pow(2).mean()
    return scale * loss

def sampled_decorrelation_loss(z: Tensor, k: int = 36, p: float = 1.5) -> Tensor:
    b, d = z.shape
    if b <= 1: return tensor(0.0, device=z.device)
    if d <= k: return decorrelation_loss(z)

    m = int(d)

    perm = torch.randperm(m, device=z.device)

    if perm.shape[0] % k != 0:
        missing = (k - perm.shape[0] % k) % k
        perm = torch.cat([perm, torch.randperm(m, device=z.device)[:missing]])

    correfs = []
    for ws in range(0, m, k):
        corref = z[:, perm[ws:ws+k]].T.corrcoef().nan_to_num()
        mask   = ~torch.eye(corref.shape[1], device=z.device).bool()
        correfs.append(corref[mask])

    loss = torch.cat(correfs).pow(2).mean()
    return loss

def deco_vae_cosine_loss(recon_x: Tensor, x: Tensor, h: Tensor, mu: Tensor, logvar: Tensor,
                         alpha: float = 1, beta: float = 1e-2, gamma: float = 5e-2, decorrelation_k: int = None) -> Tuple[Tensor, Tensor, Tensor]:
    recon_loss  = alpha * (1 - F.cosine_similarity(recon_x, x, dim=1)).mean()
    kld         = beta  * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean())
    correlation = gamma
    
    if decorrelation_k: correlation *= sampled_decorrelation_loss(h, decorrelation_k)
    else:               correlation *= decorrelation_loss(h)
    
    return recon_loss, kld, correlation

def deco_vae_mse_loss(recon_x: Tensor, x: Tensor, h: Tensor, mu: Tensor, logvar: Tensor,
                      alpha: float = 100, beta: float = 1e-2, gamma: float = 5e-2) -> Tuple[Tensor, Tensor, Tensor]:
    recon_loss  = alpha * (F.mse_loss(recon_x, x, reduction='mean'))
    kld         = beta  * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean())
    correlation = gamma * decorrelation_loss(h)
    return recon_loss, kld, correlation

def deco_vae_bce_loss(recon_x: Tensor, x: Tensor, h: Tensor, mu: Tensor, logvar: Tensor,
                      alpha: float = 100, beta: float = 1e-2, gamma: float = 5e-2) -> Tuple[Tensor, Tensor, Tensor]:
    recon_loss  = alpha * F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
    kld         = beta  * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean())
    correlation = gamma * decorrelation_loss(h)
    return recon_loss, kld, correlation
