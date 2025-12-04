"""
Chain-level confidence scoring combining internal and cross-modal coherence.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .internal_coherence import InternalCoherenceMetric
from .cross_modal_coherence import CrossModalCoherenceMetric
from ..coherence_models.density_model import DensityModel


class ChainConfidenceScorer(nn.Module):
    """
    Compute overall confidence score for a CoT chain.

    Combines:
    1. Internal textual coherence
    2. Cross-modal coherence
    3. Optional density-based rarity penalty
    """

    def __init__(
        self,
        internal_metric: Optional[InternalCoherenceMetric] = None,
        cross_modal_metric: Optional[CrossModalCoherenceMetric] = None,
        density_model: Optional[DensityModel] = None,
        internal_weight: float = 0.5,
        cross_modal_weight: float = 0.4,
        density_weight: float = 0.1,
        use_learned_weights: bool = False,
        feature_dim: int = 10
    ):
        """
        Initialize chain confidence scorer.

        Args:
            internal_metric: Internal coherence metric
            cross_modal_metric: Cross-modal coherence metric
            density_model: Optional density model for rarity detection
            internal_weight: Weight for internal coherence
            cross_modal_weight: Weight for cross-modal coherence
            density_weight: Weight for density score
            use_learned_weights: Whether to learn weights
            feature_dim: Dimension of combined features
        """
        super().__init__()

        # Coherence metrics
        self.internal_metric = internal_metric or InternalCoherenceMetric()
        self.cross_modal_metric = cross_modal_metric or CrossModalCoherenceMetric()
        self.density_model = density_model

        # Weights
        if use_learned_weights:
            # Learn weights via a small MLP
            self.weight_network = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Softmax(dim=-1)
            )
        else:
            # Fixed weights
            self.register_buffer(
                'weights',
                torch.tensor([internal_weight, cross_modal_weight, density_weight])
            )
            self.weight_network = None

    def compute_features(
        self,
        internal_scores: Dict[str, torch.Tensor],
        cross_modal_scores: Dict[str, torch.Tensor],
        density_score: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract feature vector from component scores.

        Args:
            internal_scores: Internal coherence scores
            cross_modal_scores: Cross-modal scores
            density_score: Optional density score

        Returns:
            Feature vector
        """
        features = [
            internal_scores['overall'],
            internal_scores['smoothness'],
            internal_scores['goal_directedness'],
            internal_scores['semantic_density'],
            cross_modal_scores['overall'],
            cross_modal_scores['alignment'],
        ]

        # Add contrastive if available
        if 'contrastive_score' in cross_modal_scores:
            features.append(cross_modal_scores['contrastive_score'])

        # Add per-step stats
        if 'per_step_coherence' in cross_modal_scores:
            per_step = cross_modal_scores['per_step_coherence']
            features.extend([
                per_step.mean(),
                per_step.std(),
                per_step.min()
            ])

        # Pad to fixed size
        while len(features) < 10:
            features.append(torch.tensor(0.0))

        return torch.stack(features[:10])

    def forward(
        self,
        step_embeddings: torch.Tensor,
        modal_embeddings: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,  # Backward compatibility
        question_embedding: Optional[torch.Tensor] = None,
        answer_embedding: Optional[torch.Tensor] = None,
        negative_modal_embeddings: Optional[torch.Tensor] = None,
        negative_image_embeddings: Optional[torch.Tensor] = None  # Backward compatibility
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence score for a chain.

        Args:
            step_embeddings: (num_steps, embed_dim)
            modal_embeddings: (num_modals, embed_dim) - can be image, audio, or other modality
            image_embeddings: (DEPRECATED) Use modal_embeddings instead
            question_embedding: Optional question embedding
            answer_embedding: Optional answer embedding
            negative_modal_embeddings: Optional negative samples
            negative_image_embeddings: (DEPRECATED) Use negative_modal_embeddings instead

        Returns:
            Dictionary with confidence scores and components
        """
        # Backward compatibility: support old image_embeddings parameter
        if modal_embeddings is None and image_embeddings is not None:
            modal_embeddings = image_embeddings
        if negative_modal_embeddings is None and negative_image_embeddings is not None:
            negative_modal_embeddings = negative_image_embeddings

        if modal_embeddings is None:
            raise ValueError("Must provide either modal_embeddings or image_embeddings")

        results = {}

        # Internal coherence
        internal_scores = self.internal_metric(
            step_embeddings=step_embeddings,
            question_embedding=question_embedding,
            answer_embedding=answer_embedding
        )
        results['internal'] = internal_scores

        # Cross-modal coherence
        cross_modal_scores = self.cross_modal_metric(
            step_embeddings=step_embeddings,
            modal_embeddings=modal_embeddings,
            negative_modal_embeddings=negative_modal_embeddings
        )
        results['cross_modal'] = cross_modal_scores

        # Density score (if model available)
        density_score = None
        if self.density_model is not None:
            # Compute density over the chain
            density_score = self.density_model(step_embeddings)
            results['density'] = density_score

        # Extract features
        features = self.compute_features(
            internal_scores,
            cross_modal_scores,
            density_score
        )
        results['features'] = features

        # Compute weights
        if self.weight_network is not None:
            weights = self.weight_network(features)
        else:
            weights = self.weights

        # Combine scores
        component_scores = torch.stack([
            internal_scores['overall'],
            cross_modal_scores['overall'],
            density_score if density_score is not None else torch.tensor(0.5)
        ])

        confidence = torch.sum(weights * component_scores)

        # Ensure in [0, 1] range
        confidence = torch.clamp(confidence, 0.0, 1.0)

        results['confidence'] = confidence
        results['weights'] = weights

        return results

    def compute_for_batch(
        self,
        batch_step_embeddings: List[torch.Tensor],
        batch_modal_embeddings: Optional[List[torch.Tensor]] = None,
        batch_image_embeddings: Optional[List[torch.Tensor]] = None,  # Backward compatibility
        batch_question_embeddings: Optional[List[torch.Tensor]] = None,
        batch_answer_embeddings: Optional[List[torch.Tensor]] = None,
        batch_negative_embeddings: Optional[List[torch.Tensor]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute confidence for a batch of chains.

        Args:
            batch_step_embeddings: List of step embeddings
            batch_modal_embeddings: List of modal embeddings (image, audio, etc.)
            batch_image_embeddings: (DEPRECATED) Use batch_modal_embeddings instead
            batch_question_embeddings: Optional question embeddings
            batch_answer_embeddings: Optional answer embeddings
            batch_negative_embeddings: Optional negative samples

        Returns:
            List of result dictionaries
        """
        # Backward compatibility
        if batch_modal_embeddings is None and batch_image_embeddings is not None:
            batch_modal_embeddings = batch_image_embeddings

        if batch_modal_embeddings is None:
            raise ValueError("Must provide either batch_modal_embeddings or batch_image_embeddings")

        results = []

        for i in range(len(batch_step_embeddings)):
            question_emb = (
                batch_question_embeddings[i]
                if batch_question_embeddings
                else None
            )
            answer_emb = (
                batch_answer_embeddings[i]
                if batch_answer_embeddings
                else None
            )
            neg_emb = (
                batch_negative_embeddings[i]
                if batch_negative_embeddings
                else None
            )

            result = self.forward(
                step_embeddings=batch_step_embeddings[i],
                modal_embeddings=batch_modal_embeddings[i],
                question_embedding=question_emb,
                answer_embedding=answer_emb,
                negative_modal_embeddings=neg_emb
            )
            results.append(result)

        return results


class AdaptiveConfidenceScorer(nn.Module):
    """
    Adaptive confidence scorer that learns to weight components
    based on the type of reasoning task.
    """

    def __init__(
        self,
        base_scorer: ChainConfidenceScorer,
        num_reasoning_types: int = 5,
        embedding_dim: int = 768
    ):
        """
        Initialize adaptive scorer.

        Args:
            base_scorer: Base confidence scorer
            num_reasoning_types: Number of reasoning task types
            embedding_dim: Embedding dimensionality
        """
        super().__init__()
        self.base_scorer = base_scorer

        # Task-specific weight predictors
        self.task_embeddings = nn.Embedding(num_reasoning_types, embedding_dim)
        self.task_weight_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        step_embeddings: torch.Tensor,
        modal_embeddings: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,  # Backward compatibility
        reasoning_type_id: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task-adaptive confidence score.

        Args:
            step_embeddings: Step embeddings
            modal_embeddings: Modal embeddings (image, audio, etc.)
            image_embeddings: (DEPRECATED) Use modal_embeddings instead
            reasoning_type_id: ID of reasoning task type
            **kwargs: Additional arguments

        Returns:
            Confidence scores and components
        """
        # Backward compatibility
        if modal_embeddings is None and image_embeddings is not None:
            modal_embeddings = image_embeddings

        if modal_embeddings is None:
            raise ValueError("Must provide either modal_embeddings or image_embeddings")

        # Get base scores
        results = self.base_scorer(
            step_embeddings=step_embeddings,
            modal_embeddings=modal_embeddings,
            **kwargs
        )

        # Predict task-specific weights
        task_emb = self.task_embeddings(
            torch.tensor(reasoning_type_id, device=step_embeddings.device)
        )
        task_weights = self.task_weight_predictor(task_emb)

        # Recompute confidence with task-specific weights
        component_scores = torch.stack([
            results['internal']['overall'],
            results['cross_modal']['overall'],
            results.get('density', torch.tensor(0.5))
        ])

        adaptive_confidence = torch.sum(task_weights * component_scores)
        adaptive_confidence = torch.clamp(adaptive_confidence, 0.0, 1.0)

        results['adaptive_confidence'] = adaptive_confidence
        results['task_weights'] = task_weights

        return results
