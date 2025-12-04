"""
Internal textual coherence metrics for CoT chains.

Measures how well reasoning steps connect to each other and progress
toward the goal.
"""

from typing import List, Optional, Dict
import torch
import torch.nn as nn

from ..embeddings.embedding_utils import compute_similarity, compute_pairwise_similarity


class InternalCoherenceMetric(nn.Module):
    """
    Compute internal coherence of a Chain-of-Thought.

    Measures:
    1. Local smoothness: Do consecutive steps stay semantically close?
    2. Goal-directedness: Do steps progress toward the answer?
    3. Overall chain density: Is the chain compact in embedding space?
    """

    def __init__(
        self,
        similarity_metric: str = "cosine",
        aggregation: str = "mean",
        goal_directedness_weight: float = 0.3,
        smoothness_weight: float = 0.7
    ):
        """
        Initialize internal coherence metric.

        Args:
            similarity_metric: Method to compute similarity
            aggregation: How to aggregate step similarities
            goal_directedness_weight: Weight for goal-directedness component
            smoothness_weight: Weight for smoothness component
        """
        super().__init__()
        self.similarity_metric = similarity_metric
        self.aggregation = aggregation
        self.goal_weight = goal_directedness_weight
        self.smoothness_weight = smoothness_weight

    def compute_local_smoothness(
        self,
        step_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute local smoothness: similarity between consecutive steps.

        Args:
            step_embeddings: Tensor of shape (num_steps, embed_dim)

        Returns:
            Smoothness score
        """
        if len(step_embeddings) < 2:
            return torch.tensor(1.0)

        # Compute similarity between consecutive steps
        similarities = []
        for i in range(len(step_embeddings) - 1):
            sim = compute_similarity(
                step_embeddings[i],
                step_embeddings[i + 1],
                metric=self.similarity_metric
            )
            similarities.append(sim)

        similarities = torch.stack(similarities)

        # Aggregate
        if self.aggregation == "mean":
            return torch.mean(similarities)
        elif self.aggregation == "min":
            return torch.min(similarities)
        elif self.aggregation == "harmonic_mean":
            # Harmonic mean is more sensitive to low values
            return len(similarities) / torch.sum(1.0 / (similarities + 1e-8))
        else:
            return torch.mean(similarities)

    def compute_goal_directedness(
        self,
        step_embeddings: torch.Tensor,
        goal_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute goal-directedness: do steps move toward the goal?

        Args:
            step_embeddings: Tensor of shape (num_steps, embed_dim)
            goal_embedding: Target embedding (answer or final step)

        Returns:
            Goal-directedness score
        """
        if len(step_embeddings) < 2:
            return torch.tensor(1.0)

        # Compute similarity of each step to goal
        similarities_to_goal = []
        for step_emb in step_embeddings:
            sim = compute_similarity(
                step_emb,
                goal_embedding,
                metric=self.similarity_metric
            )
            similarities_to_goal.append(sim)

        similarities_to_goal = torch.stack(similarities_to_goal)

        # Check if there's a progression toward the goal
        # Option 1: Are later steps more similar to goal than earlier steps?
        first_half_sim = similarities_to_goal[:len(similarities_to_goal)//2].mean()
        second_half_sim = similarities_to_goal[len(similarities_to_goal)//2:].mean()

        progression_score = (second_half_sim - first_half_sim + 1) / 2  # Normalize to [0, 1]

        # Option 2: Average similarity to goal
        avg_similarity = similarities_to_goal.mean()

        # Combine both aspects
        score = 0.6 * avg_similarity + 0.4 * progression_score

        return score

    def compute_semantic_variance(
        self,
        step_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute semantic variance: how spread out are the steps?

        Lower variance indicates more coherent chain.

        Args:
            step_embeddings: Tensor of shape (num_steps, embed_dim)

        Returns:
            Inverse variance score (higher is better)
        """
        if len(step_embeddings) < 2:
            return torch.tensor(1.0)

        # Compute pairwise similarities
        pairwise_sim = compute_pairwise_similarity(
            step_embeddings,
            metric=self.similarity_metric
        )

        # Average similarity (excluding diagonal)
        n = len(step_embeddings)
        mask = ~torch.eye(n, dtype=bool, device=step_embeddings.device)
        avg_similarity = pairwise_sim[mask].mean()

        return avg_similarity

    def forward(
        self,
        step_embeddings: torch.Tensor,
        question_embedding: Optional[torch.Tensor] = None,
        answer_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute overall internal coherence.

        Args:
            step_embeddings: Embeddings of reasoning steps (num_steps, embed_dim)
            question_embedding: Optional question embedding for context
            answer_embedding: Optional answer/goal embedding

        Returns:
            Dictionary of coherence scores
        """
        # Local smoothness
        smoothness = self.compute_local_smoothness(step_embeddings)

        # Goal-directedness (if answer available)
        if answer_embedding is not None:
            goal_directedness = self.compute_goal_directedness(
                step_embeddings,
                answer_embedding
            )
        else:
            # Use last step as goal
            goal_directedness = self.compute_goal_directedness(
                step_embeddings[:-1] if len(step_embeddings) > 1 else step_embeddings,
                step_embeddings[-1]
            )

        # Semantic density
        density = self.compute_semantic_variance(step_embeddings)

        # Combined score
        overall_score = (
            self.smoothness_weight * smoothness +
            self.goal_weight * goal_directedness +
            0.0 * density  # Can adjust weights
        )

        return {
            'overall': overall_score,
            'smoothness': smoothness,
            'goal_directedness': goal_directedness,
            'semantic_density': density
        }

    def compute_for_batch(
        self,
        batch_step_embeddings: List[torch.Tensor],
        batch_answer_embeddings: Optional[List[torch.Tensor]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute coherence for a batch of chains.

        Args:
            batch_step_embeddings: List of step embedding tensors
            batch_answer_embeddings: Optional list of answer embeddings

        Returns:
            List of coherence score dictionaries
        """
        results = []

        for i, step_embeds in enumerate(batch_step_embeddings):
            answer_emb = batch_answer_embeddings[i] if batch_answer_embeddings else None

            scores = self.forward(
                step_embeddings=step_embeds,
                answer_embedding=answer_emb
            )
            results.append(scores)

        return results


class HMMCoherenceModel(nn.Module):
    """
    Hidden Markov Model approach to coherence.

    Models the chain as a sequence where each step should have high
    transition probability from the previous step.
    """

    def __init__(self, embedding_dim: int, num_states: int = 8):
        """
        Initialize HMM-based coherence model.

        Args:
            embedding_dim: Dimensionality of embeddings
            num_states: Number of hidden states
        """
        super().__init__()
        self.num_states = num_states

        # Learn state embeddings
        self.state_embeddings = nn.Parameter(
            torch.randn(num_states, embedding_dim)
        )

        # Transition matrix
        self.transition_logits = nn.Parameter(
            torch.randn(num_states, num_states)
        )

    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood of chain under HMM.

        Args:
            step_embeddings: Step embeddings (num_steps, embed_dim)

        Returns:
            Log-likelihood score
        """
        # Compute emission probabilities (similarity to states)
        emission_logits = torch.matmul(
            step_embeddings,
            self.state_embeddings.T
        )  # (num_steps, num_states)

        # Simple forward algorithm
        num_steps = len(step_embeddings)
        log_prob = 0.0

        for t in range(1, num_steps):
            # Get most likely states for current and previous steps
            prev_state = emission_logits[t-1].argmax()
            curr_state = emission_logits[t].argmax()

            # Add transition log-probability
            transition_probs = torch.softmax(self.transition_logits, dim=-1)
            log_prob += torch.log(transition_probs[prev_state, curr_state] + 1e-8)

        # Normalize by number of transitions
        if num_steps > 1:
            log_prob = log_prob / (num_steps - 1)

        return log_prob
