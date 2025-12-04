"""
Cross-modal coherence metrics between text reasoning steps and other modalities.

Measures how well CoT steps align with visual (images), auditory (audio),
or other non-textual information.
"""

from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embeddings.embedding_utils import compute_similarity


class CrossModalCoherenceMetric(nn.Module):
    """
    Compute cross-modal coherence between reasoning steps and other modalities.

    Supports alignment with images, audio, video, or any other modality embeddings.

    Measures:
    1. Step-to-modal alignment: Do steps refer to relevant modal information?
    2. Contrastive coherence: Is alignment stronger to true modality than negatives?
    3. Attention-weighted alignment: Do steps attend to relevant modal features?
    """

    def __init__(
        self,
        similarity_metric: str = "cosine",
        contrastive_margin: float = 0.2,
        temperature: float = 0.07,
        use_attention: bool = False
    ):
        """
        Initialize cross-modal coherence metric.

        Args:
            similarity_metric: Method to compute similarity
            contrastive_margin: Margin for contrastive loss
            temperature: Temperature for contrastive similarity
            use_attention: Whether to use attention weighting
        """
        super().__init__()
        self.similarity_metric = similarity_metric
        self.contrastive_margin = contrastive_margin
        self.temperature = temperature
        self.use_attention = use_attention

    def compute_step_modal_alignment(
        self,
        step_embeddings: torch.Tensor,
        modal_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment between steps and modal embeddings (images, audio, etc.).

        Args:
            step_embeddings: (num_steps, embed_dim)
            modal_embeddings: (num_modals, embed_dim) or (embed_dim,) - can be image, audio, etc.

        Returns:
            Alignment score
        """
        if modal_embeddings.dim() == 1:
            modal_embeddings = modal_embeddings.unsqueeze(0)

        # Compute similarity matrix: (num_steps, num_modals)
        similarities = []
        if self.similarity_metric == "clap":
            for modal_emb in modal_embeddings:
                similarities.append(torch.matmul(modal_emb, step_embeddings.T))
        else:
            for step_emb in step_embeddings:
                step_sims = []
                for modal_emb in modal_embeddings:
                    sim = compute_similarity(
                            step_emb,
                            modal_emb,
                            metric=self.similarity_metric
                        )
                    step_sims.append(sim)
                similarities.append(torch.stack(step_sims))

        similarity_matrix = torch.stack(similarities)  # (num_steps, num_modals)

        # Aggregate: average max similarity per step
        max_sim_per_step = similarity_matrix.max(dim=1)[0]
        alignment_score = max_sim_per_step.mean()

        return alignment_score

    def compute_contrastive_coherence(
        self,
        step_embeddings: torch.Tensor,
        positive_modal_embeddings: torch.Tensor,
        negative_modal_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive coherence: alignment to true vs. distractor modals.

        Args:
            step_embeddings: (num_steps, embed_dim)
            positive_modal_embeddings: (num_pos_modals, embed_dim) - true modality (image/audio)
            negative_modal_embeddings: (num_neg_modals, embed_dim) - distractor modality

        Returns:
            Dictionary with contrastive scores
        """
        # Positive alignment
        pos_alignment = self.compute_step_modal_alignment(
            step_embeddings,
            positive_modal_embeddings
        )

        # Negative alignment
        neg_alignment = self.compute_step_modal_alignment(
            step_embeddings,
            negative_modal_embeddings
        )

        # Contrastive score: positive should be higher than negative
        contrastive_score = pos_alignment - neg_alignment

        # Apply margin
        margin_score = F.relu(contrastive_score - self.contrastive_margin)

        return {
            'contrastive_score': contrastive_score,
            'margin_score': margin_score,
            'positive_alignment': pos_alignment,
            'negative_alignment': neg_alignment
        }

    def compute_attention_weighted_alignment(
        self,
        step_embeddings: torch.Tensor,
        modal_embeddings: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention-weighted alignment.

        Each step attends to modal features, weighted by relevance.

        Args:
            step_embeddings: (num_steps, embed_dim)
            modal_embeddings: (num_modals, embed_dim) - image, audio, or other modality
            return_attention: Whether to return attention weights

        Returns:
            Weighted alignment score and optionally attention weights
        """
        if modal_embeddings.dim() == 1:
            modal_embeddings = modal_embeddings.unsqueeze(0)

        # Compute attention scores: (num_steps, num_modals)
        attention_logits = torch.matmul(
            step_embeddings,
            modal_embeddings.T
        ) / self.temperature

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_logits, dim=1)

        # Compute similarity matrix
        similarity_matrix = []
        for step_emb in step_embeddings:
            step_sims = []
            for modal_emb in modal_embeddings:
                sim = compute_similarity(
                    step_emb,
                    modal_emb,
                    metric=self.similarity_metric
                )
                step_sims.append(sim)
            similarity_matrix.append(torch.stack(step_sims))

        similarity_matrix = torch.stack(similarity_matrix)  # (num_steps, num_modals)

        # Weight similarities by attention
        weighted_similarities = similarity_matrix * attention_weights

        # Aggregate
        alignment_score = weighted_similarities.sum(dim=1).mean()

        if return_attention:
            return alignment_score, attention_weights
        else:
            return alignment_score, None

    def compute_per_step_coherence(
        self,
        step_embeddings: torch.Tensor,
        modal_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coherence score for each step individually.

        Args:
            step_embeddings: (num_steps, embed_dim)
            modal_embeddings: (num_modals, embed_dim) - image, audio, or other modality

        Returns:
            Per-step coherence scores (num_steps,)
        """
        if modal_embeddings.dim() == 1:
            modal_embeddings = modal_embeddings.unsqueeze(0)

        per_step_scores = []

        for step_emb in step_embeddings:
            # Max similarity to any modal embedding
            step_sims = []
            for modal_emb in modal_embeddings:
                sim = compute_similarity(
                    step_emb,
                    modal_emb,
                    metric=self.similarity_metric
                )
                step_sims.append(sim)

            max_sim = torch.stack(step_sims).max()
            per_step_scores.append(max_sim)

        return torch.stack(per_step_scores)

    def forward(
        self,
        step_embeddings: torch.Tensor,
        modal_embeddings: torch.Tensor,
        negative_modal_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute overall cross-modal coherence.

        Args:
            step_embeddings: (num_steps, embed_dim)
            modal_embeddings: (num_modals, embed_dim) - can be image, audio, or other modality
            negative_modal_embeddings: Optional negative samples

        Returns:
            Dictionary of coherence scores
        """
        results = {}

        # Basic alignment
        alignment = self.compute_step_modal_alignment(
            step_embeddings,
            modal_embeddings
        )
        results['alignment'] = alignment

        # Attention-weighted alignment
        if self.use_attention:
            weighted_alignment, attention = self.compute_attention_weighted_alignment(
                step_embeddings,
                modal_embeddings,
                return_attention=True
            )
            results['weighted_alignment'] = weighted_alignment
            results['attention_weights'] = attention

        # Contrastive coherence
        if negative_modal_embeddings is not None:
            contrastive_scores = self.compute_contrastive_coherence(
                step_embeddings,
                modal_embeddings,
                negative_modal_embeddings
            )
            results.update(contrastive_scores)

        # Per-step scores
        per_step_scores = self.compute_per_step_coherence(
            step_embeddings,
            modal_embeddings
        )
        results['per_step_coherence'] = per_step_scores
        results['min_step_coherence'] = per_step_scores.min()

        # Overall score (weighted combination)
        if self.use_attention and 'weighted_alignment' in results:
            overall = results['weighted_alignment']
        else:
            overall = alignment

        # Add contrastive component if available
        if 'contrastive_score' in results:
            overall = 0.7 * overall + 0.3 * results['contrastive_score']

        results['overall'] = overall

        return results

    def compute_for_batch(
        self,
        batch_step_embeddings: List[torch.Tensor],
        batch_modal_embeddings: List[torch.Tensor],
        batch_negative_embeddings: Optional[List[torch.Tensor]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute cross-modal coherence for a batch.

        Args:
            batch_step_embeddings: List of step embeddings
            batch_modal_embeddings: List of modal embeddings (image, audio, etc.)
            batch_negative_embeddings: Optional negative samples

        Returns:
            List of score dictionaries
        """
        results = []

        for i, (step_embeds, modal_embeds) in enumerate(
            zip(batch_step_embeddings, batch_modal_embeddings)
        ):
            neg_embeds = (
                batch_negative_embeddings[i]
                if batch_negative_embeddings
                else None
            )

            scores = self.forward(
                step_embeddings=step_embeds,
                modal_embeddings=modal_embeds,
                negative_modal_embeddings=neg_embeds
            )
            results.append(scores)

        return results

    # Backward compatibility aliases for image-specific methods
    def compute_step_image_alignment(self, step_embeddings: torch.Tensor,
                                    image_embeddings: torch.Tensor) -> torch.Tensor:
        """Backward compatibility: Use compute_step_modal_alignment instead."""
        return self.compute_step_modal_alignment(step_embeddings, image_embeddings)


class RegionAlignmentMetric(nn.Module):
    """
    Fine-grained region-level alignment between steps and image regions.

    Requires access to spatial features (e.g., from vision transformer patches).
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize region alignment metric.

        Args:
            temperature: Temperature for attention computation
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        step_embeddings: torch.Tensor,
        region_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute region-level alignment.

        Args:
            step_embeddings: (num_steps, embed_dim)
            region_features: (num_regions, embed_dim) - e.g., ViT patches

        Returns:
            Dictionary with alignment scores and attention maps
        """
        # Compute step-to-region attention
        attention_logits = torch.matmul(
            step_embeddings,
            region_features.T
        ) / self.temperature  # (num_steps, num_regions)

        attention_weights = F.softmax(attention_logits, dim=1)

        # Compute region-weighted alignment
        similarities = torch.matmul(
            step_embeddings,
            region_features.T
        )  # (num_steps, num_regions)

        weighted_sim = (similarities * attention_weights).sum(dim=1)
        alignment_score = weighted_sim.mean()

        # Attention entropy (lower = more focused attention)
        attention_entropy = -(
            attention_weights * torch.log(attention_weights + 1e-8)
        ).sum(dim=1).mean()

        return {
            'alignment': alignment_score,
            'attention_entropy': attention_entropy,
            'attention_weights': attention_weights
        }
