"""
Confidence prediction head for learning to predict answer correctness.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceHead(nn.Module):
    """
    MLP head that learns to predict confidence/correctness from coherence features.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        activation: str = "relu"
    ):
        """
        Initialize confidence head.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Input features (batch_size, input_dim) or (input_dim,)

        Returns:
            Confidence scores (batch_size,) or scalar
        """
        # Handle single sample
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Forward pass
        logits = self.mlp(features)

        # Apply sigmoid to get probability
        confidence = torch.sigmoid(logits).squeeze(-1)

        if squeeze:
            confidence = confidence.squeeze(0)

        return confidence


class EnsembleConfidenceHead(nn.Module):
    """
    Ensemble of confidence heads for improved robustness.
    """

    def __init__(
        self,
        num_heads: int = 5,
        input_dim: int = 10,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        """
        Initialize ensemble.

        Args:
            num_heads: Number of heads in ensemble
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.heads = nn.ModuleList([
            ConfidenceHead(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
            for _ in range(num_heads)
        ])

    def forward(
        self,
        features: torch.Tensor,
        return_individual: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Args:
            features: Input features
            return_individual: Return individual head predictions

        Returns:
            Ensemble confidence score(s)
        """
        predictions = [head(features) for head in self.heads]
        predictions = torch.stack(predictions)

        if return_individual:
            return predictions

        # Average ensemble
        ensemble_confidence = predictions.mean(dim=0)

        return ensemble_confidence


class CalibratedConfidenceHead(nn.Module):
    """
    Confidence head with temperature scaling for calibration.
    """

    def __init__(
        self,
        base_head: ConfidenceHead,
        initial_temperature: float = 1.0
    ):
        """
        Initialize calibrated head.

        Args:
            base_head: Base confidence head
            initial_temperature: Initial temperature value
        """
        super().__init__()

        self.base_head = base_head

        # Learnable temperature parameter
        self.temperature = nn.Parameter(
            torch.tensor(initial_temperature)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temperature scaling.

        Args:
            features: Input features

        Returns:
            Calibrated confidence scores
        """
        # Get base logits (before sigmoid)
        # We need to modify base_head to expose logits
        # For now, use inverse sigmoid
        base_confidence = self.base_head(features)

        # Convert to logits
        epsilon = 1e-7
        base_confidence = torch.clamp(base_confidence, epsilon, 1 - epsilon)
        logits = torch.log(base_confidence / (1 - base_confidence))

        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Convert back to probability
        calibrated_confidence = torch.sigmoid(scaled_logits)

        return calibrated_confidence


class MultiTaskConfidenceHead(nn.Module):
    """
    Multi-task head that predicts both confidence and reasoning quality.
    """

    def __init__(
        self,
        input_dim: int = 10,
        shared_dims: List[int] = [512, 256],
        task_dims: List[int] = [128],
        dropout: float = 0.3
    ):
        """
        Initialize multi-task head.

        Args:
            input_dim: Input feature dimension
            shared_dims: Shared layer dimensions
            task_dims: Task-specific layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim

        self.shared = nn.Sequential(*shared_layers)

        # Task-specific heads
        # Task 1: Confidence (binary)
        confidence_layers = []
        task_prev_dim = prev_dim
        for dim in task_dims:
            confidence_layers.extend([
                nn.Linear(task_prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            task_prev_dim = dim
        confidence_layers.append(nn.Linear(task_prev_dim, 1))
        self.confidence_head = nn.Sequential(*confidence_layers)

        # Task 2: Reasoning quality score (regression)
        quality_layers = []
        task_prev_dim = prev_dim
        for dim in task_dims:
            quality_layers.extend([
                nn.Linear(task_prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            task_prev_dim = dim
        quality_layers.append(nn.Linear(task_prev_dim, 1))
        self.quality_head = nn.Sequential(*quality_layers)

    def forward(
        self,
        features: torch.Tensor
    ) -> dict:
        """
        Forward pass for multi-task prediction.

        Args:
            features: Input features

        Returns:
            Dictionary with confidence and quality predictions
        """
        # Shared representation
        shared_repr = self.shared(features)

        # Confidence prediction
        confidence_logits = self.confidence_head(shared_repr)
        confidence = torch.sigmoid(confidence_logits).squeeze(-1)

        # Quality prediction (0-1 scale)
        quality_logits = self.quality_head(shared_repr)
        quality = torch.sigmoid(quality_logits).squeeze(-1)

        return {
            'confidence': confidence,
            'quality': quality
        }
