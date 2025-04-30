import torch
import torch.nn as nn

def vocab_aware_embedding_loss(predicted_embedding: torch.Tensor, true_token_indexes: torch.Tensor, embedding_layer: nn.Embedding, vocab_penalty_weight: float = 1.0):
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

    # 1. Distance to Correct Token Embedding (L2 Distance)
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
