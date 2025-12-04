"""
Utility functions for embedding operations and caching.
"""

from typing import Dict, List, Optional, Union
import os
import pickle
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def compute_similarity(
    embed1: torch.Tensor,
    embed2: torch.Tensor,
    metric: str = "cosine"
) -> torch.Tensor:
    """
    Compute similarity between embeddings.

    Args:
        embed1: First embedding(s)
        embed2: Second embedding(s)
        metric: Similarity metric ('cosine', 'dot_product', 'euclidean')

    Returns:
        Similarity scores
    """
    if metric == "cosine":
        # Normalize and compute dot product
        embed1_norm = torch.nn.functional.normalize(embed1, p=2, dim=-1)
        embed2_norm = torch.nn.functional.normalize(embed2, p=2, dim=-1)
        return torch.sum(embed1_norm * embed2_norm, dim=-1)

    elif metric == "dot_product":
        return torch.sum(embed1 * embed2, dim=-1)

    elif metric == "euclidean":
        # Return negative distance (higher is more similar)
        return -torch.norm(embed1 - embed2, p=2, dim=-1)

    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def compute_pairwise_similarity(
    embeddings: torch.Tensor,
    metric: str = "cosine"
) -> torch.Tensor:
    """
    Compute pairwise similarity matrix.

    Args:
        embeddings: Tensor of shape (N, embedding_dim)
        metric: Similarity metric

    Returns:
        Similarity matrix of shape (N, N)
    """
    if metric == "cosine":
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return torch.matmul(embeddings_norm, embeddings_norm.T)

    elif metric == "dot_product":
        return torch.matmul(embeddings, embeddings.T)

    elif metric == "euclidean":
        # Compute pairwise distances
        dist = torch.cdist(embeddings, embeddings, p=2)
        return -dist

    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def aggregate_similarities(
    similarities: torch.Tensor,
    method: str = "mean"
) -> torch.Tensor:
    """
    Aggregate similarity scores.

    Args:
        similarities: Tensor of similarity scores
        method: Aggregation method ('mean', 'min', 'max', 'median')

    Returns:
        Aggregated score
    """
    if method == "mean":
        return torch.mean(similarities)
    elif method == "min":
        return torch.min(similarities)
    elif method == "max":
        return torch.max(similarities)
    elif method == "median":
        return torch.median(similarities)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings to avoid recomputation.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_cache: Dict[str, torch.Tensor] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieve embedding from cache.

        Args:
            key: Cache key

        Returns:
            Cached embedding or None if not found
        """
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pt"
            if cache_file.exists():
                embedding = torch.load(cache_file)
                # Store in memory for faster access
                self.memory_cache[key] = embedding
                return embedding

        return None

    def set(self, key: str, embedding: torch.Tensor, save_to_disk: bool = True):
        """
        Store embedding in cache.

        Args:
            key: Cache key
            embedding: Embedding to cache
            save_to_disk: Whether to persist to disk
        """
        # Store in memory
        self.memory_cache[key] = embedding

        # Optionally save to disk
        if save_to_disk and self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pt"
            torch.save(embedding, cache_file)

    def clear(self, clear_disk: bool = False):
        """
        Clear cache.

        Args:
            clear_disk: Whether to also clear disk cache
        """
        self.memory_cache.clear()

        if clear_disk and self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pt"):
                cache_file.unlink()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        if key in self.memory_cache:
            return True

        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pt"
            return cache_file.exists()

        return False


def batch_encode(
    encoder,
    items: List,
    batch_size: int = 32,
    show_progress: bool = True,
    **encode_kwargs
) -> torch.Tensor:
    """
    Encode items in batches.

    Args:
        encoder: Encoder with forward/encode method
        items: List of items to encode
        batch_size: Batch size
        show_progress: Show progress bar
        **encode_kwargs: Additional arguments for encoder

    Returns:
        Tensor of all embeddings
    """
    from tqdm import tqdm

    all_embeddings = []

    iterator = range(0, len(items), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding")

    for i in iterator:
        batch = items[i:i + batch_size]
        embeddings = encoder(batch, **encode_kwargs)
        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)


def interpolate_embeddings(
    embed1: torch.Tensor,
    embed2: torch.Tensor,
    alpha: float = 0.5,
    normalize: bool = True
) -> torch.Tensor:
    """
    Interpolate between two embeddings.

    Args:
        embed1: First embedding
        embed2: Second embedding
        alpha: Interpolation weight (0 = embed1, 1 = embed2)
        normalize: Normalize result

    Returns:
        Interpolated embedding
    """
    result = (1 - alpha) * embed1 + alpha * embed2

    if normalize:
        result = torch.nn.functional.normalize(result, p=2, dim=-1)

    return result
