"""
Pool-level internal coherence metrics for ranking candidate reasoning chains.

Given a pool of candidate reasoning chains (each with their own step embeddings
and answer embedding), scores and ranks candidates by how internally coherent
their reasoning is — smoothness, goal-directedness, and semantic density.

Key design decisions:
- Each candidate supplies its own answer embedding (goal target).
- Scoring is relative/contrastive: each candidate is compared against the
  rest of the pool, not scored in isolation.
- The composite ranking signal is a contrastive z-score over a weighted
  combination of the three InternalCoherenceMetric sub-scores, with an
  EB / James-Stein variance penalty applied across the pool dimension.
- Optional answer-agreement signal: pairwise soft majority-vote over answer
  embeddings interpolated into the composite score.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .answer_agreement_mixin import AnswerAgreementMixin
from .internal_coherence import InternalCoherenceMetric


class CandidatePoolInternalCoherenceMetric(AnswerAgreementMixin, nn.Module):
    """
    Rank a pool of candidate reasoning chains by internal coherence quality.

    Each candidate provides its own step embeddings and answer embedding.
    Scoring is contrastive: a candidate's value comes from being distinctively
    more coherent than the rest of the pool, not just adequate in isolation.

    Scoring strategy
    ----------------
    1. **Absolute scores** — each candidate is independently evaluated with
       ``InternalCoherenceMetric``, extracting smoothness, goal_directedness,
       and semantic_density.

    2. **Composite absolute score** — a weighted sum of the three sub-scores,
       with weights matching those in ``InternalCoherenceMetric`` by default
       (smoothness=0.7, goal=0.3, density=0.0) but overridable here.

    3. **EB variance-penalised pool score** — James-Stein shrinkage is applied
       *across the candidate axis* to the composite absolute scores, pulling
       noisy outlier candidates toward the pool mean.

    4. **Contrastive z-score** — the EB-shrunk score is z-normalised against
       the pool distribution, giving each candidate a signed deviation from
       the pool average.

    5. **Answer agreement** (optional) — pairwise cosine similarity between
       each candidate's answer embedding and every other candidate's, averaged
       over the pool (self excluded).  This is a soft majority-vote signal:
       candidates whose answer matches most others score higher.  EB-shrunk
       and z-scored before mixing.

    6. **Sub-score contrastive z-scores** — z-scores are also computed
       independently for each of the three sub-scores, so callers can inspect
       which dimension (smoothness / goal / density) drives a candidate's rank.

    7. **Composite rank score** — configurable weighted blend of the
       contrastive z-score, the normalised absolute score, and (optionally)
       the answer-agreement signal.

    Args:
        similarity_metric: Passed through to InternalCoherenceMetric.
        aggregation: Step-similarity aggregation method for smoothness.
        smoothness_weight: Weight of smoothness in the composite score.
        goal_directedness_weight: Weight of goal-directedness in the composite.
        density_weight: Weight of semantic density in the composite.
        variance_penalty_weight: λ for the EB variance penalty (pool axis).
        contrastive_weight: Weight of the contrastive z-score in the final rank.
        absolute_weight: Weight of the normalised absolute score in the final rank.
        answer_agreement_weight: Weight of the answer-agreement signal in the
            composite score.  Set to 0.0 (default) to disable entirely and
            preserve the original two-term composite.  The three weights do not
            need to sum to 1 — each component is independently normalised to
            [0, 1] before weighting.
    """

    def __init__(
        self,
        similarity_metric: str = "cosine",
        aggregation: str = "mean",
        smoothness_weight: float = 0.7,
        goal_directedness_weight: float = 0.3,
        density_weight: float = 0.0,
        variance_penalty_weight: float = 1.0,
        contrastive_weight: float = 0.6,
        absolute_weight: float = 0.4,
        answer_agreement_weight: float = 0.0,
    ):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.smoothness_weight = smoothness_weight
        self.goal_directedness_weight = goal_directedness_weight
        self.density_weight = density_weight
        self.variance_penalty_weight = variance_penalty_weight
        self.contrastive_weight = contrastive_weight
        self.absolute_weight = absolute_weight
        self.answer_agreement_weight = answer_agreement_weight

        self._base_metric = InternalCoherenceMetric(
            similarity_metric=similarity_metric,
            aggregation=aggregation,
            goal_directedness_weight=goal_directedness_weight,
            smoothness_weight=smoothness_weight,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _composite(
        self,
        smoothness: torch.Tensor,
        goal_directedness: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted combination of the three sub-scores."""
        return (
            self.smoothness_weight * smoothness
            + self.goal_directedness_weight * goal_directedness
            + self.density_weight * density
        )

    def _pool_eb_shrink(self, scores: torch.Tensor) -> torch.Tensor:
        """
        James-Stein / Empirical-Bayes shrinkage across the pool (candidate) axis.

        Candidates whose scores deviate from the pool mean in a way that looks
        noisy (high pooled variance relative to their squared deviation) are
        pulled toward the pool mean, preventing volatile high-scorers from
        dominating.

        Args:
            scores: (N,) per-candidate scores.

        Returns:
            (N,) EB-shrunk scores.
        """
        mu = scores.mean()
        var_pool = scores.var(unbiased=False)
        sq_dev = (scores - mu) ** 2
        epsilon = 1e-6
        B = var_pool / (sq_dev + var_pool + epsilon)  # (N,) shrinkage factor
        return (1 - B) * scores + B * mu

    def _pool_z_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Z-normalise scores against the pool distribution.

        z_i = (s_i - μ_pool) / (σ_pool + ε)

        Positive z ⟹ above-average; magnitude = standard deviations above.
        """
        mu = scores.mean()
        sigma = scores.std(unbiased=False)
        return (scores - mu) / (sigma + 1e-6)

    def _pool_contrastive_margin(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each candidate, compute its margin over the mean of the *rest*.

        margin_i = s_i - mean(s_{j ≠ i})

        Also returns the overall best-vs-rest margin (scalar).
        """
        N = scores.size(0)
        pool_sum = scores.sum()
        # mean of all others for each candidate i: (sum - s_i) / (N - 1)
        if N > 1:
            rest_mean = (pool_sum - scores) / (N - 1)
        else:
            rest_mean = scores  # single candidate edge case
        per_candidate_margin = scores - rest_mean

        best_score = scores.max()
        best_rest_mean = (pool_sum - best_score) / max(N - 1, 1)
        pool_margin = best_score - best_rest_mean

        return per_candidate_margin, pool_margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_pool(
        self,
        pool_step_embeddings: List[torch.Tensor],
        pool_answer_embeddings: List[torch.Tensor],
        pool_question_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Score and rank a pool of candidates by internal coherence.

        Args:
            pool_step_embeddings: List of length ``N``; each element is a
                ``(num_steps_i, embed_dim)`` tensor of reasoning step embeddings
                for candidate *i*.
            pool_answer_embeddings: List of length ``N``; each element is a
                ``(embed_dim,)`` or ``(1, embed_dim)`` answer/goal embedding
                for candidate *i*.
            pool_question_embeddings: Optional list of length ``N``; question
                embeddings forwarded to the base metric if supplied.

        Returns:
            A dictionary containing:

            Per-candidate tensors (shape ``(N,)``):
                ``absolute_smoothness``
                    Raw smoothness score from InternalCoherenceMetric.
                ``absolute_goal_directedness``
                    Raw goal-directedness score.
                ``absolute_semantic_density``
                    Raw semantic density score.
                ``absolute_composite``
                    Weighted composite of the three sub-scores.
                ``pool_eb_shrunk``
                    EB-shrunk composite score across the pool dimension.
                ``contrastive_z_score``
                    Z-score of each candidate's EB-shrunk score vs the pool.
                ``contrastive_z_smoothness``
                    Z-score of raw smoothness vs the pool (diagnostic).
                ``contrastive_z_goal``
                    Z-score of raw goal-directedness vs the pool (diagnostic).
                ``contrastive_z_density``
                    Z-score of raw semantic density vs the pool (diagnostic).
                ``per_candidate_margin``
                    Each candidate's composite score minus mean of the rest.
                ``composite_score``
                    Weighted blend of contrastive z-score and normalised
                    absolute composite — primary ranking signal.
                ``rank``
                    Integer rank of each candidate (0 = best).

            Pool-level scalars:
                ``pool_margin``
                    Best candidate's composite score minus mean of the rest.
                ``pool_mean_composite``
                    Mean composite absolute score across the pool.
                ``pool_std_composite``
                    Std of composite absolute scores across the pool.
                ``best_candidate_idx``
                    Index of the top-ranked candidate.
        """
        N = len(pool_step_embeddings)
        if N == 0:
            raise ValueError(
                "pool_step_embeddings must contain at least one candidate."
            )
        if len(pool_answer_embeddings) != N:
            raise ValueError(
                f"pool_answer_embeddings length ({len(pool_answer_embeddings)}) "
                f"must match pool_step_embeddings length ({N})."
            )

        # ----------------------------------------------------------------
        # 1. Absolute scores — independent per candidate
        # ----------------------------------------------------------------
        abs_smoothness_list = []
        abs_goal_list = []
        abs_density_list = []

        for i, (step_embeds, answer_emb) in enumerate(
            zip(pool_step_embeddings, pool_answer_embeddings)
        ):
            question_emb = (
                pool_question_embeddings[i] if pool_question_embeddings else None
            )
            scores = self._base_metric(
                step_embeddings=step_embeds.float(),
                question_embedding=(
                    question_emb.float() if question_emb is not None else None
                ),
                answer_embedding=answer_emb.float(),
            )
            abs_smoothness_list.append(scores["smoothness"])
            abs_goal_list.append(scores["goal_directedness"])
            abs_density_list.append(scores["semantic_density"])

        abs_smoothness = torch.stack(abs_smoothness_list)  # (N,)
        abs_goal = torch.stack(abs_goal_list)  # (N,)
        abs_density = torch.stack(abs_density_list)  # (N,)
        abs_composite = self._composite(abs_smoothness, abs_goal, abs_density)  # (N,)

        # ----------------------------------------------------------------
        # 2. EB shrinkage across the pool (candidate axis)
        # ----------------------------------------------------------------
        pool_eb_shrunk = self._pool_eb_shrink(abs_composite)  # (N,)

        # ----------------------------------------------------------------
        # 3. Contrastive z-scores
        # ----------------------------------------------------------------
        # Primary z-score: on the EB-shrunk composite (main ranking signal)
        contrastive_z = self._pool_z_scores(pool_eb_shrunk)  # (N,)

        # Diagnostic z-scores per sub-score (not used for ranking)
        z_smoothness = self._pool_z_scores(abs_smoothness)  # (N,)
        z_goal = self._pool_z_scores(abs_goal)  # (N,)
        z_density = self._pool_z_scores(abs_density)  # (N,)

        # ----------------------------------------------------------------
        # 4. Per-candidate margin over mean of the rest
        # ----------------------------------------------------------------
        per_candidate_margin, pool_margin = self._pool_contrastive_margin(
            pool_eb_shrunk
        )

        # ----------------------------------------------------------------
        # 5. Answer agreement (soft majority vote)
        # ----------------------------------------------------------------
        if self.answer_agreement_weight > 0.0:
            agreement_scores = self.compute_answer_agreement(pool_answer_embeddings)
            agr_raw = agreement_scores["answer_agreement_raw"]  # (N,)
            agr_eb = agreement_scores["answer_agreement_eb"]  # (N,)
            agr_z = agreement_scores["answer_agreement_z"]  # (N,)
            agr_normed = agreement_scores["answer_agreement_normed"]  # (N,)
            agr_matrix = agreement_scores["answer_sim_matrix"]  # (N, N)
        else:
            _zero = torch.zeros(N, device=abs_composite.device)
            agr_raw = agr_eb = agr_z = agr_normed = _zero
            agr_matrix = torch.zeros(N, N, device=abs_composite.device)

        # ----------------------------------------------------------------
        # 6. Composite rank score
        # ----------------------------------------------------------------
        # Each component independently normalised to [0, 1] before weighting.
        c_min, c_max = abs_composite.min(), abs_composite.max()
        abs_normed = (abs_composite - c_min) / (c_max - c_min + 1e-6)  # (N,)

        contrastive_normed = torch.sigmoid(contrastive_z)  # (N,) → (0, 1)

        composite_score = (
            self.contrastive_weight * contrastive_normed
            + self.absolute_weight * abs_normed
            + self.answer_agreement_weight * agr_normed
        )  # (N,)

        # ----------------------------------------------------------------
        # 7. Ranking
        # ----------------------------------------------------------------
        order = torch.argsort(composite_score, descending=True)
        ranks = torch.zeros(N, dtype=torch.long)
        for rank_pos, cand_idx in enumerate(order):
            ranks[cand_idx] = rank_pos

        best_idx = int(order[0].item())

        return {
            # Per-candidate (N,)
            "absolute_smoothness": abs_smoothness,
            "absolute_goal_directedness": abs_goal,
            "absolute_semantic_density": abs_density,
            "absolute_composite": abs_composite,
            "pool_eb_shrunk": pool_eb_shrunk,
            "contrastive_z_score": contrastive_z,
            "contrastive_z_smoothness": z_smoothness,
            "contrastive_z_goal": z_goal,
            "contrastive_z_density": z_density,
            "per_candidate_margin": per_candidate_margin,
            "answer_agreement_raw": agr_raw,
            "answer_agreement_eb": agr_eb,
            "answer_agreement_z": agr_z,
            "composite_score": composite_score,
            "rank": ranks,
            # Pool-level tensors / scalars
            "answer_sim_matrix": agr_matrix,
            "pool_margin": pool_margin,
            "pool_mean_composite": abs_composite.mean(),
            "pool_std_composite": abs_composite.std(unbiased=False),
            "best_candidate_idx": torch.tensor(best_idx),
        }

    def rank_candidates(
        self,
        pool_step_embeddings: List[torch.Tensor],
        pool_answer_embeddings: List[torch.Tensor],
        pool_question_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        Convenience wrapper: returns candidates sorted best-first.

        Returns:
            List of ``(original_index, per_candidate_scores_dict)`` tuples
            ordered by composite score descending. Each dict contains all
            scalar scores for that candidate extracted from ``score_pool``.
        """
        results = self.score_pool(
            pool_step_embeddings, pool_answer_embeddings, pool_question_embeddings
        )

        scalar_keys = [
            "absolute_smoothness",
            "absolute_goal_directedness",
            "absolute_semantic_density",
            "absolute_composite",
            "pool_eb_shrunk",
            "contrastive_z_score",
            "contrastive_z_smoothness",
            "contrastive_z_goal",
            "contrastive_z_density",
            "per_candidate_margin",
            "answer_agreement_raw",
            "answer_agreement_eb",
            "answer_agreement_z",
            "composite_score",
            "rank",
        ]
        N = len(pool_step_embeddings)
        per_candidate = [{k: results[k][i] for k in scalar_keys} for i in range(N)]

        return sorted(
            enumerate(per_candidate),
            key=lambda x: x[1]["composite_score"].item(),
            reverse=True,
        )

    def forward(
        self,
        pool_step_embeddings: List[torch.Tensor],
        pool_answer_embeddings: List[torch.Tensor],
        pool_question_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Alias for score_pool — makes the class usable as a standard nn.Module."""
        return self.score_pool(
            pool_step_embeddings, pool_answer_embeddings, pool_question_embeddings
        )
