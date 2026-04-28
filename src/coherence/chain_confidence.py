"""
Chain-level confidence scoring combining internal and cross-modal coherence.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .internal_coherence import InternalCoherenceMetric
from .cross_modal_coherence import CrossModalCoherenceMetric
from ..coherence_models.density_model import DensityModel
# Assuming the text-based metrics are saved here:
# from .text_coherence import NLICoherenceMetric, PRMCoherenceMetric


class ChainConfidenceScorer(nn.Module):
    """
    Compute overall confidence score for a CoT chain.

    Combines:
    1. Internal textual coherence (Embedding-based)
    2. Cross-modal coherence
    3. Optional density-based rarity penalty
    4. Optional NLI-based logical entailment (Text-based)
    5. Optional PRM-based step correctness (Text-based)
    """

    def __init__(
        self,
        internal_metric: Optional[InternalCoherenceMetric] = None,
        cross_modal_metric: Optional[CrossModalCoherenceMetric] = None,
        density_model: Optional[DensityModel] = None,
        nli_metric = None,  # Optional[NLICoherenceMetric]
        prm_metric = None,  # Optional[PRMCoherenceMetric]
        internal_weight: float = 0.3,
        cross_modal_weight: float = 0.3,
        density_weight: float = 0.1,
        nli_weight: float = 0.15,
        prm_weight: float = 0.15,
        use_learned_weights: bool = False,
        feature_dim: int = 16  # 16 to include weighted_alignment feature
    ):
        super().__init__()

        # Coherence metrics (Embedding-based)
        self.internal_metric = internal_metric or InternalCoherenceMetric()
        self.cross_modal_metric = cross_modal_metric or CrossModalCoherenceMetric()
        self.density_model = density_model
        
        # Coherence metrics (Text-based)
        self.nli_metric = nli_metric
        self.prm_metric = prm_metric

        # 5 Component weights: Internal, CrossModal, Density, NLI, PRM
        if use_learned_weights:
            self.weight_network = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 5), # Output 5 weights instead of 3
                nn.Softmax(dim=-1)
            )
        else:
            self.register_buffer(
                'weights',
                torch.tensor([
                    internal_weight, 
                    cross_modal_weight, 
                    density_weight, 
                    nli_weight if nli_metric else 0.0, 
                    prm_weight if prm_metric else 0.0
                ])
            )
            # Normalize static weights just in case some are disabled
            self.weights = self.weights / (self.weights.sum() + 1e-8)
            self.weight_network = None

    def compute_features(
        self,
        internal_scores: Dict[str, torch.Tensor],
        cross_modal_scores: Dict[str, torch.Tensor],
        density_score: Optional[torch.Tensor] = None,
        nli_scores: Optional[Dict[str, torch.Tensor]] = None,
        prm_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        
        features = [
            internal_scores['overall'],
            internal_scores['smoothness'],
            internal_scores['goal_directedness'],
            internal_scores['semantic_density'],
            cross_modal_scores['overall'],
            cross_modal_scores.get('alignment', torch.tensor(0.0)),
            cross_modal_scores.get('weighted_alignment', torch.tensor(0.0)),
            cross_modal_scores.get('eb_variance_penalised', torch.tensor(0.0)),
        ]

        if 'contrastive_score' in cross_modal_scores:
            features.append(cross_modal_scores['contrastive_score'])

        if 'per_step_coherence' in cross_modal_scores:
            per_step = cross_modal_scores['per_step_coherence']
            features.extend([per_step.mean(), per_step.std(), per_step.min()])
        
        # Add NLI Features
        if nli_scores:
            features.extend([
                nli_scores['overall'],
                nli_scores['cumulative_step_nli'],
                nli_scores['goal_nli']
            ])
            
        # Add PRM Features
        if prm_scores:
            features.extend([
                prm_scores['overall'],
                prm_scores['min_step_reward']
            ])

        # Pad to fixed size
        while len(features) < 16:
            features.append(torch.tensor(0.0).to(features[0].device if isinstance(features[0], torch.Tensor) else 'cpu'))

        return torch.stack(features[:16])

    def forward(
        self,
        step_embeddings: torch.Tensor,
        modal_embeddings: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        question_embedding: Optional[torch.Tensor] = None,
        answer_embedding: Optional[torch.Tensor] = None,
        negative_modal_embeddings: Optional[torch.Tensor] = None,
        negative_image_embeddings: Optional[torch.Tensor] = None,
        # --- NEW TEXT PARAMETERS ---
        text_steps: Optional[List[str]] = None,
        text_query: Optional[str] = None,
        text_final_answer: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:

        if modal_embeddings is None and image_embeddings is not None:
            modal_embeddings = image_embeddings
        if negative_modal_embeddings is None and negative_image_embeddings is not None:
            negative_modal_embeddings = negative_image_embeddings

        if modal_embeddings is None:
            raise ValueError("Must provide either modal_embeddings or image_embeddings")

        results = {}

        # 1. Internal coherence (Embeddings)
        internal_scores = self.internal_metric(
            step_embeddings=step_embeddings,
            question_embedding=question_embedding,
            answer_embedding=answer_embedding
        )
        results['internal'] = internal_scores

        # 2. Cross-modal coherence
        cross_modal_scores = self.cross_modal_metric(
            step_embeddings=step_embeddings,
            modal_embeddings=modal_embeddings,
            negative_modal_embeddings=negative_modal_embeddings
        )
        results['cross_modal'] = cross_modal_scores

        # 3. Density score
        density_score = None
        if self.density_model is not None:
            density_score = self.density_model(step_embeddings)
            results['density'] = density_score

        # 4. Text-based Logical Entailment (NLI)
        nli_scores = None
        if self.nli_metric is not None and text_steps is not None:
            nli_scores = self.nli_metric(steps=text_steps, final_answer=text_final_answer)
            results['nli'] = nli_scores

        # 5. Text-based PRM Reasoning
        prm_scores = None
        if self.prm_metric is not None and text_steps is not None and text_query is not None:
            prm_scores = self.prm_metric(query=text_query, steps=text_steps)
            results['prm'] = prm_scores

        # Extract features
        features = self.compute_features(
            internal_scores, cross_modal_scores, density_score, nli_scores, prm_scores
        )
        results['features'] = features

        # Compute weights
        if self.weight_network is not None:
            weights = self.weight_network(features)
        else:
            weights = self.weights

        # Combine scores (Fallback to neutral 0.5 if component is missing)
        component_scores = torch.stack([
            internal_scores['overall'].to(weights.device),
            cross_modal_scores['overall'].to(weights.device),
            density_score.to(weights.device) if density_score is not None else torch.tensor(0.5, device=weights.device),
            nli_scores['overall'].to(weights.device) if nli_scores is not None else torch.tensor(0.5, device=weights.device),
            prm_scores['overall'].to(weights.device) if prm_scores is not None else torch.tensor(0.5, device=weights.device)
        ])

        confidence = torch.sum(weights * component_scores)
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
        batch_negative_embeddings: Optional[List[torch.Tensor]] = None,
        # --- NEW TEXT PARAMETERS ---
        batch_steps: Optional[List[List[str]]] = None,
        batch_query: Optional[List[str]] = None,
        batch_final_answer: Optional[List[str]] = None
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
                negative_modal_embeddings=neg_emb,
                text_steps=batch_steps[i],
                text_query=batch_query[i],
                text_final_answer=batch_final_answer[i]
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
        raise NotImplementedError()
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
