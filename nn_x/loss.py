import torch
import torch.nn as nn

def vocab_aware_embedding_loss(predicted_embedding: torch.Tensor, true_token_indexes: torch.Tensor, embedding_layer: nn.Embedding, vocab_penalty_weight: float = 1.0, distance_function: str = "cosine") -> torch.Tensor:
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
    """
    Computes the decorrelation loss for a batch of vectors.

    This loss penalizes the linear correlation between the different dimensions of the vectors. It does so by calculating the covariance matrix of
    the batch and summing the squares of the off-diagonal elements. A lower loss means the dimensions are less correlated.

    Args:
        z (torch.Tensor): The batch of vectors from the encoder of shape: (batch_size, d_model).

    Returns:
        torch.Tensor: A scalar tensor representing the decorrelation loss.
    """
    # The covariance matrix is not defined for a batch size of 1.
    if z.shape[0] <= 1:
        return torch.tensor(0.0, device=z.device)

    # 1. Center the data
    # Subtract the mean of each dimension
    z_centered = z - z.mean(dim=0)

    # 2. Compute the covariance matrix
    # The formula is (Z^T * Z) / (n - 1)
    cov_matrix = (z_centered.T @ z_centered) / (z.shape[0] - 1)

    # 3. Calculate the loss
    # We want to push the off-diagonal elements to zero.
    # We take the squared Frobenius norm of the off-diagonal elements.
    identity_mask = torch.eye(z.shape[1], device=z.device)
    
    # Zero out the diagonal elements to only consider off-diagonal correlations
    off_diagonal_cov = cov_matrix * (1 - identity_mask)
    
    # The loss is the sum of the squares of these off-diagonal elements
    loss = torch.sum(off_diagonal_cov**2)

    return loss