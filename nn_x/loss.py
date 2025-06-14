import torch
import torch.nn as nn
import torch.nn.functional as F

def vocab_aware_embedding_loss(predicted_embedding: torch.Tensor, true_token_indexes: torch.Tensor,
                               embedding_layer: nn.Embedding, vocab_penalty_weight: float = 1.0,
                               distance_function: str = "cosine") -> torch.Tensor:
    """
    Calculates a loss that penalizes predictions far from the true token
    and far from the vocabulary embedding space.
    Args:
        - predicted_embedding (torch.Tensor): Predicted embedding (batch_size, embedding_dim).
        - true_token_indexes (torch.Tensor): Index of the true token (batch_size,).
        - embedding_layer (torch.nn.Embedding): The embedding layer itself.
        - vocab_penalty_weight (float): Weight for the vocabulary penalty term.
    Returns: torch.Tensor: The calculated loss (scalar).
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

def decorrelation_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] <= 1: return torch.tensor(0.0, device=z.device)
    return torch.sum((torch.nan_to_num(torch.corrcoef(z.T), nan=0.0) * (1 - torch.eye(z.shape[1], device=z.device))) ** 2)

def deco_vae_cosine_loss(recon_x: torch.Tensor, x: torch.Tensor, h: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                         alpha: float = 1, beta: float = 1e-2, gamma: float = 5e-2) -> torch.Tensor:
    recon_loss  = alpha * (1 - F.cosine_similarity(recon_x, x, dim=1)).mean()
    kld         = beta  * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    correlation = gamma * decorrelation_loss(h) / h.shape[0]
    return recon_loss, kld, correlation

def deco_vae_mse_loss(recon_x: torch.Tensor, x: torch.Tensor, h: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                      alpha: float = 100, beta: float = 1e-2, gamma: float = 5e-2) -> torch.Tensor:
    recon_loss  = alpha * (F.mse_loss(recon_x, x, reduction='mean'))
    kld         = beta  * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    correlation = gamma * decorrelation_loss(h) / h.shape[0]
    return recon_loss, kld, correlation

def deco_vae_bce_loss(recon_x: torch.Tensor, x: torch.Tensor, h: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                      alpha: float = 100, beta: float = 1e-2, gamma: float = 5e-2) -> torch.Tensor:
    recon_loss  = alpha * F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
    kld         = beta  * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    correlation = gamma * decorrelation_loss(h) / h.shape[0]
    return recon_loss, kld, correlation
