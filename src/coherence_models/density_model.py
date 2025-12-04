"""
Density models for detecting rare/anomalous reasoning chains.
"""

from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


class DensityModel(nn.Module):
    """
    Base class for density models.

    Used to assess if a chain's embeddings are in high-density regions
    of the training distribution (typical) or low-density (anomalous).
    """

    def __init__(self):
        super().__init__()

    def fit(self, embeddings: torch.Tensor):
        """
        Fit the density model to training embeddings.

        Args:
            embeddings: Training embeddings (N, embed_dim)
        """
        raise NotImplementedError

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute density score for embeddings.

        Args:
            embeddings: Embeddings to score (batch_size, embed_dim) or (embed_dim,)

        Returns:
            Density scores (higher = more typical)
        """
        raise NotImplementedError


class KDEDensityModel(DensityModel):
    """
    Kernel Density Estimation for computing embedding density.
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = 'gaussian',
        aggregation: str = 'mean'
    ):
        """
        Initialize KDE model.

        Args:
            bandwidth: KDE bandwidth
            kernel: Kernel type
            aggregation: How to aggregate densities across chain steps
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.aggregation = aggregation
        self.kde = None

    def fit(self, embeddings: torch.Tensor):
        """
        Fit KDE to training embeddings.

        Args:
            embeddings: Training embeddings (N, embed_dim)
        """
        # Convert to numpy for sklearn
        X = embeddings.detach().cpu().numpy()

        # Fit KDE
        self.kde = KernelDensity(
            bandwidth=self.bandwidth,
            kernel=self.kernel
        )
        self.kde.fit(X)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute density score.

        Args:
            embeddings: Query embeddings

        Returns:
            Density score
        """
        if self.kde is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle single embedding
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Convert to numpy
        X = embeddings.detach().cpu().numpy()

        # Compute log density
        log_densities = self.kde.score_samples(X)

        # Convert to tensor
        densities = torch.tensor(
            np.exp(log_densities),
            dtype=torch.float32,
            device=embeddings.device
        )

        # Aggregate if multiple steps
        if self.aggregation == 'mean':
            score = densities.mean()
        elif self.aggregation == 'min':
            score = densities.min()
        elif self.aggregation == 'median':
            score = densities.median()
        else:
            score = densities.mean()

        if squeeze:
            score = score.squeeze()

        return score


class GMMDensityModel(DensityModel):
    """
    Gaussian Mixture Model for density estimation.
    """

    def __init__(
        self,
        n_components: int = 8,
        covariance_type: str = 'full',
        aggregation: str = 'mean'
    ):
        """
        Initialize GMM model.

        Args:
            n_components: Number of Gaussian components
            covariance_type: Type of covariance
            aggregation: How to aggregate densities
        """
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.aggregation = aggregation
        self.gmm = None

    def fit(self, embeddings: torch.Tensor):
        """
        Fit GMM to training embeddings.

        Args:
            embeddings: Training embeddings (N, embed_dim)
        """
        # Convert to numpy
        X = embeddings.detach().cpu().numpy()

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=42
        )
        self.gmm.fit(X)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute density score.

        Args:
            embeddings: Query embeddings

        Returns:
            Density score
        """
        if self.gmm is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle single embedding
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Convert to numpy
        X = embeddings.detach().cpu().numpy()

        # Compute log probability
        log_probs = self.gmm.score_samples(X)

        # Convert to probability
        probs = torch.tensor(
            np.exp(log_probs),
            dtype=torch.float32,
            device=embeddings.device
        )

        # Aggregate
        if self.aggregation == 'mean':
            score = probs.mean()
        elif self.aggregation == 'min':
            score = probs.min()
        else:
            score = probs.mean()

        if squeeze:
            score = score.squeeze()

        return score


class LearnedDensityModel(DensityModel):
    """
    Neural density model that learns to predict typicality.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256
    ):
        """
        Initialize learned density model.

        Args:
            embedding_dim: Embedding dimensionality
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Neural network to predict density
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def fit(self, embeddings: torch.Tensor):
        """
        For learned model, fitting is done via standard training.
        This method can be used for self-supervised pre-training.
        """
        pass

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute learned density score.

        Args:
            embeddings: Query embeddings

        Returns:
            Density score
        """
        # Handle single embedding
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Forward pass
        scores = self.network(embeddings).squeeze(-1)

        # Aggregate across steps
        score = scores.mean()

        if squeeze:
            score = score.squeeze()

        return score
