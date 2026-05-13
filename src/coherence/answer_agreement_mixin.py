"""
Shared answer-agreement signal for pool-level candidate ranking.

Computes a soft majority-vote score for each candidate by measuring
how similar its answer embedding is to every other candidate's answer
embedding (pairwise mean similarity, self excluded).

Designed to be mixed into CandidatePoolCoherenceMetric and
CandidatePoolInternalCoherenceMetric — not intended for direct use.
"""

from typing import List

import torch
import torch.nn.functional as F


class AnswerAgreementMixin:
    """
    Mixin that adds pairwise answer-agreement scoring to a pool metric.

    Requires the host class to have a ``similarity_metric`` attribute
    (``"cosine"`` or ``"dot"``).

    Answer agreement pipeline
    -------------------------
    1. **Pairwise similarity matrix** (N × N) between all answer embeddings.
    2. **Self-excluded row mean** — each candidate's mean similarity to *every
       other* candidate.  This is a soft analogue of majority-vote frequency:
       a candidate whose answer matches most others scores close to 1; an
       outlier answer scores close to 0 (for cosine) or lower (for dot).
    3. **EB / James-Stein shrinkage** across the pool axis — same logic used
       elsewhere in the pool metrics, prevents a single noisy high-agreement
       candidate from dominating.
    4. **Z-score** of the EB-shrunk agreement scores, giving a signed
       deviation from the pool mean that can be mixed with other z-scored
       signals on a common scale.
    5. **Normalised [0, 1]** version of the raw agreement scores for direct
       interpolation into a weighted composite.
    """

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _pairwise_similarity_matrix(
        self, answer_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute an (N, N) pairwise similarity matrix.

        Args:
            answer_embeddings: (N, embed_dim) — one embedding per candidate.

        Returns:
            (N, N) similarity matrix.  Diagonal entries are self-similarity
            (1.0 for cosine) and are excluded when computing row means.
        """
        if getattr(self, "similarity_metric", "cosine") == "cosine":
            normed = F.normalize(answer_embeddings, p=2, dim=-1)
            sim_matrix = torch.matmul(normed, normed.T)  # (N, N), range [-1, 1]
            # Rescale to [0, 1] to match the convention used in the base metrics
            sim_matrix = (sim_matrix + 1.0) / 2.0
        else:
            # Dot product — caller is responsible for embedding scale
            sim_matrix = torch.matmul(answer_embeddings, answer_embeddings.T)

        return sim_matrix

    def _self_excluded_row_mean(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """
        For each row i, compute the mean of all off-diagonal entries.

        mean_i = (sum_j sim[i,j] - sim[i,i]) / (N - 1)

        This is the soft majority-vote score: how much does candidate i's
        answer agree with the rest of the pool on average?

        Args:
            sim_matrix: (N, N) pairwise similarity matrix.

        Returns:
            (N,) agreement scores.
        """
        N = sim_matrix.size(0)
        row_sum = sim_matrix.sum(dim=1)  # (N,)
        self_sim = torch.diag(sim_matrix)  # (N,)
        if N > 1:
            return (row_sum - self_sim) / (N - 1)
        else:
            return self_sim  # edge case: single candidate

    def _eb_shrink(self, scores: torch.Tensor) -> torch.Tensor:
        """
        James-Stein / EB shrinkage across the pool (candidate) axis.

        Pulls candidates whose scores deviate from the pool mean in a
        direction that looks noisy toward the pool mean.

        Identical logic to ``_pool_eb_shrink`` / ``_pool_eb_variance_penalty``
        in the host classes — reproduced here so the mixin is self-contained.
        """
        mu = scores.mean()
        var_pool = scores.var(unbiased=False)
        sq_dev = (scores - mu) ** 2
        B = var_pool / (sq_dev + var_pool + 1e-6)
        return (1 - B) * scores + B * mu

    def _z_score(self, scores: torch.Tensor) -> torch.Tensor:
        """Z-normalise scores against the pool distribution."""
        mu = scores.mean()
        sigma = scores.std(unbiased=False)
        return (scores - mu) / (sigma + 1e-6)

    def _normalise_01(self, scores: torch.Tensor) -> torch.Tensor:
        """Min-max normalise scores to [0, 1] within the pool."""
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo + 1e-6)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compute_answer_agreement(
        self, pool_answer_embeddings: List[torch.Tensor]
    ) -> dict:
        """
        Compute all answer-agreement signals for the pool.

        Args:
            pool_answer_embeddings: List of N answer embedding tensors, each
                of shape ``(embed_dim,)`` or ``(1, embed_dim)``.

        Returns:
            Dictionary with keys:

            ``answer_agreement_raw``   (N,) — self-excluded pairwise mean sim.
            ``answer_agreement_eb``    (N,) — EB-shrunk raw scores.
            ``answer_agreement_z``     (N,) — z-score of EB-shrunk scores.
            ``answer_agreement_normed``(N,) — [0,1]-normalised raw scores,
                                              ready for composite interpolation.
            ``answer_sim_matrix``      (N, N) — full pairwise similarity matrix
                                                (diagnostic).
        """
        # Stack and flatten to (N, D)
        stacked = torch.stack(
            [
                e.float().squeeze(0) if e.dim() > 1 else e.float()
                for e in pool_answer_embeddings
            ]
        )  # (N, D)

        sim_matrix = self._pairwise_similarity_matrix(stacked)  # (N, N)
        raw = self._self_excluded_row_mean(sim_matrix)  # (N,)
        eb = self._eb_shrink(raw)  # (N,)
        z = self._z_score(eb)  # (N,)
        normed = self._normalise_01(raw)  # (N,)

        return {
            "answer_agreement_raw": raw,
            "answer_agreement_eb": eb,
            "answer_agreement_z": z,
            "answer_agreement_normed": normed,
            "answer_sim_matrix": sim_matrix,
        }
