"""
Pool-level cross-modal coherence metrics for ranking candidate answer embeddings.

Given a pool of candidate reasoning chains (each with their own step embeddings)
and a shared set of modal embeddings (image, audio, etc.), this module scores and
ranks candidates by how well their reasoning is grounded in the modality.

Key design decisions:
- Modality embeddings are **shared** across all candidates (passed once).
- Scoring is **relative/contrastive**: each candidate is compared against the
  rest of the pool, not just scored in isolation.
- Carries over: entropy-gated routing and EB / James-Stein variance penalisation
  from CrossModalCoherenceMetric.
- Optional answer-agreement signal: pairwise soft majority-vote over answer
  embeddings (last step of each chain) interpolated into the composite score.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .answer_agreement_mixin import AnswerAgreementMixin
from .cross_modal_coherence import CrossModalCoherenceMetric


class CandidatePoolCoherenceMetric(AnswerAgreementMixin, nn.Module):
    """
    Rank a pool of candidate answer-step embeddings by cross-modal grounding quality.

    Each candidate provides its own step embeddings; the modal embeddings (image,
    audio, …) are identical for every candidate in the pool — they represent the
    shared perceptual context that all answers should be grounded in.

    Scoring strategy
    ----------------
    1. **Absolute scores** – each candidate is independently evaluated with
       entropy-gated routing and EB variance-penalised alignment (reusing the
       logic from CrossModalCoherenceMetric).

    2. **Relative / contrastive scores** – each candidate's absolute score is
       compared against the *pool aggregate* (mean + std).  This surfaces
       candidates that are *distinctively* better grounded, not just
       adequate in isolation.

    3. **Answer agreement** (optional) – pairwise cosine similarity between
       each candidate's answer embedding (last reasoning step) and every other
       candidate's answer embedding, averaged over the pool (self excluded).
       This is a soft majority-vote signal: candidates whose answer matches
       most others score higher.  EB-shrunk and z-scored before mixing.

    4. **Pool margin score** – the gap between the best candidate and the
       mean of the rest.  Large margin ⟹ one candidate is clearly dominant.

    5. **Composite rank score** – configurable weighted combination of the
       contrastive z-score, the absolute EB-penalised alignment, and
       (optionally) the answer-agreement signal.

    Args:
        similarity_metric: Passed through to CrossModalCoherenceMetric.
        temperature: Temperature for attention / entropy computations.
        variance_penalty_weight: λ for the EB variance penalty term.
        contrastive_weight: Weight of the relative z-score in the composite score.
        absolute_weight: Weight of the absolute EB score in the composite score.
        answer_agreement_weight: Weight of the answer-agreement signal in the
            composite score.  Set to 0.0 (default) to disable entirely and
            preserve the original two-term composite.  The three weights do not
            need to sum to 1 — each component is independently normalised to
            [0, 1] before weighting.
    """

    def __init__(
        self,
        similarity_metric: str = "cosine",
        temperature: float = 0.07,
        variance_penalty_weight: float = 1.0,
        contrastive_weight: float = 0.6,
        absolute_weight: float = 0.4,
        answer_agreement_weight: float = 0.7,
    ):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.variance_penalty_weight = variance_penalty_weight
        self.contrastive_weight = contrastive_weight
        self.absolute_weight = absolute_weight
        self.answer_agreement_weight = answer_agreement_weight

        # Re-use the single-candidate scorer for absolute metrics
        self._base_metric = CrossModalCoherenceMetric(
            similarity_metric=similarity_metric,
            temperature=temperature,
            variance_penalty_weight=variance_penalty_weight,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _absolute_scores_for_candidate(
        self,
        step_embeddings: torch.Tensor,
        modal_embeddings: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Run the base metric on a single candidate and extract the two
        absolute scores we care about:
          - ``entropy_gated_routing``
          - ``eb_variance_penalised``  (omni-modal) or a fallback when
            only a single modality tensor is supplied.
        """
        scores = self._base_metric(
            step_embeddings=step_embeddings.float(),
            modal_embeddings=(
                modal_embeddings
                if isinstance(modal_embeddings, dict)
                else modal_embeddings.float()
            ),
        )

        # eb_variance_penalised is only emitted for omni-modal (dict) inputs.
        # For single-modality, approximate it with variance-penalised alignment
        # computed here directly.
        if "eb_variance_penalised" not in scores:
            per_step = scores["per_step_coherence"]
            mu = per_step.mean()
            var = per_step.var(unbiased=False)
            pooled_var = var  # single candidate → pooled == step var
            epsilon = 1e-6
            B = pooled_var / (var + pooled_var + epsilon)
            shrunk_var = (1 - B) * var + B * pooled_var
            eb_val = torch.clamp(
                mu - self.variance_penalty_weight * shrunk_var, min=0.0
            )
            scores["eb_variance_penalised"] = eb_val

        return scores

    # ------------------------------------------------------------------
    # Pool-level contrastive helpers
    # ------------------------------------------------------------------

    def _pool_contrastive_z_scores(self, absolute_values: torch.Tensor) -> torch.Tensor:
        """
        Convert a 1-D tensor of per-candidate absolute scores into
        z-scores relative to the pool distribution.

        z_i = (s_i - μ_pool) / (σ_pool + ε)

        A positive z-score means the candidate is above the pool average;
        the magnitude indicates how many standard deviations above.
        """
        mu = absolute_values.mean()
        sigma = absolute_values.std(unbiased=False)
        return (absolute_values - mu) / (sigma + 1e-6)

    def _pool_eb_variance_penalty(self, absolute_values: torch.Tensor) -> torch.Tensor:
        """
        Apply a James-Stein / Empirical-Bayes shrinkage penalty **across the
        pool dimension** (candidates, not steps).

        This down-weights candidates whose scores are outliers in a noisy
        direction — if the pool variance is high relative to a candidate's
        deviation, we shrink that candidate's score toward the pool mean.

        Returns a tensor of EB-shrunk scores (same shape as absolute_values).
        """
        mu_pool = absolute_values.mean()
        var_pool = absolute_values.var(unbiased=False)

        candidate_sq_dev = (absolute_values - mu_pool) ** 2
        epsilon = 1e-6

        # Shrinkage factor per candidate
        B = var_pool / (candidate_sq_dev + var_pool + epsilon)

        # Shrunk estimate: pull each candidate toward the pool mean
        shrunk_scores = (1 - B) * absolute_values + B * mu_pool
        return shrunk_scores

    def _pool_entropy_gated_routing(
        self, per_candidate_per_step: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool-level entropy-gated routing.

        Treats **candidates** as the "modalities" axis and **steps** as the
        shared dimension, mirroring the logic in
        ``CrossModalCoherenceMetric.compute_entropy_gated_routing``.

        For each step, this rewards candidates that are *decisively* better
        grounded than all others in the pool at that step, penalising
        candidates that score similarly to everyone (generic/ungrounded steps
        look the same across candidates).

        Args:
            per_candidate_per_step: (num_candidates, num_steps) tensor of
                per-step coherence scores.

        Returns:
            (num_candidates,) tensor of pool-entropy-gated scores.
        """
        num_candidates = per_candidate_per_step.size(0)

        # Hard max: each step's score for the *best* candidate at that step
        max_scores_per_step, _ = per_candidate_per_step.max(dim=0)  # (num_steps,)

        # Routing probability over candidates for each step (sharp softmax)
        routing_probs = F.softmax(
            per_candidate_per_step / 0.05, dim=0
        )  # (num_candidates, num_steps)

        # Shannon entropy per step (high ⟹ all candidates equally grounded ⟹ step is generic)
        entropy = -(routing_probs * torch.log(routing_probs + 1e-8)).sum(
            dim=0
        )  # (num_steps,)
        max_entropy = torch.log(
            torch.tensor(
                num_candidates,
                dtype=torch.float32,
                device=per_candidate_per_step.device,
            )
        )
        decisiveness = 1.0 - (entropy / (max_entropy + 1e-8))  # (num_steps,)

        # Each candidate's contribution: its score × decisiveness weight at each step,
        # aggregated as a mean over steps.
        # Shape: (num_candidates, num_steps) * (num_steps,) → mean over steps
        gated_per_candidate = (per_candidate_per_step * decisiveness.unsqueeze(0)).mean(
            dim=1
        )  # (num_candidates,)

        return gated_per_candidate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_pool(
        self,
        pool_step_embeddings: List[torch.Tensor],
        modal_embeddings: Union[torch.Tensor, Dict[str, torch.Tensor]],
        pool_answer_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Score and rank a pool of candidates against shared modal embeddings.

        Args:
            pool_step_embeddings: List of length ``N`` where each element is a
                ``(num_steps_i, embed_dim)`` tensor — the reasoning step
                embeddings for candidate *i*.
            modal_embeddings: Shared perceptual context.  Either:
                - ``(num_modals, embed_dim)`` tensor for a single modality, or
                - ``Dict[str, Tensor]`` for omni-modal inputs.
                This is the **same** object for every candidate.
            pool_answer_embeddings: Optional list of length ``N``; explicit
                answer embeddings for each candidate.  When omitted, the last
                step embedding of each candidate is used as a proxy.  Only
                consumed when ``answer_agreement_weight > 0``.

        Returns:
            A dictionary containing:

            Per-candidate tensors (shape ``(N,)``):
                ``absolute_entropy_gated``
                    Absolute entropy-gated routing score per candidate.
                ``absolute_eb_penalised``
                    Absolute EB variance-penalised alignment per candidate.
                ``contrastive_z_score``
                    Z-score of each candidate's EB score vs the pool.
                ``pool_eb_shrunk``
                    EB-shrunk score across the pool dimension.
                ``pool_entropy_gated``
                    Pool-level entropy-gated routing score per candidate.
                ``answer_agreement_raw``
                    Self-excluded pairwise mean similarity (soft majority vote).
                    Zero tensor when ``answer_agreement_weight == 0``.
                ``answer_agreement_eb``
                    EB-shrunk answer agreement scores.
                ``answer_agreement_z``
                    Z-score of EB-shrunk agreement scores.
                ``composite_score``
                    Weighted combination of contrastive z-score, absolute EB
                    score, and (optionally) answer-agreement signal.
                ``rank``
                    Integer rank of each candidate (0 = best).

            Pool-level tensors / scalars:
                ``answer_sim_matrix``
                    (N, N) pairwise answer similarity matrix (diagnostic).
                    Zero matrix when ``answer_agreement_weight == 0``.
                ``pool_margin``
                    Score gap between the best and mean of the rest.
                ``pool_mean_absolute``
                    Mean absolute EB score across the pool.
                ``pool_std_absolute``
                    Std of absolute EB scores across the pool.
                ``best_candidate_idx``
                    Index of the top-ranked candidate.
        """
        num_candidates = len(pool_step_embeddings)
        if num_candidates == 0:
            raise ValueError(
                "pool_step_embeddings must contain at least one candidate."
            )

        # ----------------------------------------------------------------
        # 1. Absolute scores — independent per candidate
        # ----------------------------------------------------------------
        abs_entropy_gated = []
        abs_eb_penalised = []
        per_candidate_per_step_list = []

        for step_embeds in pool_step_embeddings:
            scores = self._absolute_scores_for_candidate(step_embeds, modal_embeddings)
            abs_entropy_gated.append(scores["entropy_gated_routing"])
            abs_eb_penalised.append(scores["eb_variance_penalised"])

            # Collect per-step coherence for pool-level entropy gating below
            # Shape: (num_steps,)
            per_candidate_per_step_list.append(scores["per_step_coherence"])

        abs_entropy_gated_t = torch.stack(abs_entropy_gated)  # (N,)
        abs_eb_penalised_t = torch.stack(abs_eb_penalised)  # (N,)

        # Pad per-step tensors to the same length so we can stack them.
        # Steps beyond a candidate's actual length are masked with 0.
        max_steps = max(t.size(0) for t in per_candidate_per_step_list)
        padded = []
        for t in per_candidate_per_step_list:
            pad_len = max_steps - t.size(0)
            if pad_len > 0:
                t = F.pad(t, (0, pad_len), value=0.0)
            padded.append(t)
        per_candidate_per_step = torch.stack(padded)  # (N, max_steps)

        # ----------------------------------------------------------------
        # 2. Relative / contrastive scores
        # ----------------------------------------------------------------

        # 2a. Z-scores of the EB-penalised absolute scores across the pool
        contrastive_z = self._pool_contrastive_z_scores(abs_eb_penalised_t)  # (N,)

        # 2b. EB shrinkage across the pool dimension
        pool_eb_shrunk = self._pool_eb_variance_penalty(abs_eb_penalised_t)  # (N,)

        # 2c. Pool-level entropy-gated routing (candidates as "modalities")
        pool_entropy_gated = self._pool_entropy_gated_routing(
            per_candidate_per_step
        )  # (N,)

        # ----------------------------------------------------------------
        # 3. Answer agreement (soft majority vote)
        # ----------------------------------------------------------------
        # Resolve answer embeddings: explicit arg takes priority; fall back to
        # last step of each candidate chain as a proxy.
        _answer_embs: List[torch.Tensor]
        if pool_answer_embeddings is not None:
            _answer_embs = pool_answer_embeddings
        else:
            _answer_embs = [steps[-1] for steps in pool_step_embeddings]

        if self.answer_agreement_weight > 0.0:
            agreement_scores = self.compute_answer_agreement(_answer_embs)
            agr_raw = agreement_scores["answer_agreement_raw"]  # (N,)
            agr_eb = agreement_scores["answer_agreement_eb"]  # (N,)
            agr_z = agreement_scores["answer_agreement_z"]  # (N,)
            agr_normed = agreement_scores["answer_agreement_normed"]  # (N,)
            agr_matrix = agreement_scores["answer_sim_matrix"]  # (N, N)
        else:
            # Disabled — emit zero tensors so the output dict is always consistent
            _zero = torch.zeros(num_candidates, device=abs_eb_penalised_t.device)
            agr_raw = agr_eb = agr_z = agr_normed = _zero
            agr_matrix = torch.zeros(
                num_candidates, num_candidates, device=abs_eb_penalised_t.device
            )

        # ----------------------------------------------------------------
        # 4. Composite score & ranking
        # ----------------------------------------------------------------
        # Each component is independently normalised to [0, 1] before weighting
        # so the weights have a consistent interpretation regardless of scale.

        eb_min, eb_max = abs_eb_penalised_t.min(), abs_eb_penalised_t.max()
        abs_eb_normed = (abs_eb_penalised_t - eb_min) / (eb_max - eb_min + 1e-6)  # (N,)

        contrastive_normed = torch.sigmoid(contrastive_z)  # (N,) → (0, 1)

        composite = (
            self.contrastive_weight * contrastive_normed
            + self.absolute_weight * abs_eb_normed
            + self.answer_agreement_weight * agr_normed
        )  # (N,)

        # Ranks: argsort descending → position of each candidate
        order = torch.argsort(composite, descending=True)
        ranks = torch.zeros(num_candidates, dtype=torch.long)
        for rank_pos, cand_idx in enumerate(order):
            ranks[cand_idx] = rank_pos

        best_idx = int(order[0].item())

        # ----------------------------------------------------------------
        # 5. Pool-level summary statistics
        # ----------------------------------------------------------------
        best_score = abs_eb_penalised_t[best_idx]
        rest_mask = torch.ones(num_candidates, dtype=torch.bool)
        rest_mask[best_idx] = False
        if rest_mask.any():
            rest_mean = abs_eb_penalised_t[rest_mask].mean()
        else:
            rest_mean = best_score  # single candidate edge case
        pool_margin = best_score - rest_mean

        return {
            # Per-candidate (N,)
            "absolute_entropy_gated": abs_entropy_gated_t,
            "absolute_eb_penalised": abs_eb_penalised_t,
            "contrastive_z_score": contrastive_z,
            "pool_eb_shrunk": pool_eb_shrunk,
            "pool_entropy_gated": pool_entropy_gated,
            "answer_agreement_raw": agr_raw,
            "answer_agreement_eb": agr_eb,
            "answer_agreement_z": agr_z,
            "composite_score": composite,
            "rank": ranks,
            # Pool-level tensors / scalars
            "answer_sim_matrix": agr_matrix,
            "pool_margin": pool_margin,
            "pool_mean_absolute": abs_eb_penalised_t.mean(),
            "pool_std_absolute": abs_eb_penalised_t.std(unbiased=False),
            "best_candidate_idx": torch.tensor(best_idx),
        }

    def rank_candidates(
        self,
        pool_step_embeddings: List[torch.Tensor],
        modal_embeddings: Union[torch.Tensor, Dict[str, torch.Tensor]],
        pool_answer_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        Convenience wrapper: returns candidates sorted best-first.

        Returns:
            List of ``(original_index, per_candidate_scores_dict)`` tuples
            ordered by composite score descending.  Each per-candidate dict
            contains all scalar scores for that candidate extracted from the
            full ``score_pool`` output.
        """
        results = self.score_pool(
            pool_step_embeddings, modal_embeddings, pool_answer_embeddings
        )

        # Build per-candidate score dicts
        scalar_keys = [
            "absolute_entropy_gated",
            "absolute_eb_penalised",
            "contrastive_z_score",
            "pool_eb_shrunk",
            "pool_entropy_gated",
            "answer_agreement_raw",
            "answer_agreement_eb",
            "answer_agreement_z",
            "composite_score",
            "rank",
        ]
        num_candidates = len(pool_step_embeddings)
        per_candidate = [
            {k: results[k][i] for k in scalar_keys} for i in range(num_candidates)
        ]

        # Sort by composite score descending
        ranked = sorted(
            enumerate(per_candidate),
            key=lambda x: x[1]["composite_score"].item(),
            reverse=True,
        )

        return ranked

    def forward(
        self,
        pool_step_embeddings: List[torch.Tensor],
        modal_embeddings: Union[torch.Tensor, Dict[str, torch.Tensor]],
        pool_answer_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Alias for score_pool — makes the class usable as a standard nn.Module."""
        return self.score_pool(
            pool_step_embeddings, modal_embeddings, pool_answer_embeddings
        )
