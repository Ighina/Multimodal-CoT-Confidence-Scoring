"""
Unit tests for coherence metrics.
"""

import pytest
import torch
from src.coherence import (
    InternalCoherenceMetric,
    CrossModalCoherenceMetric,
    ChainConfidenceScorer
)


class TestInternalCoherence:
    """Test internal coherence metric."""

    def test_local_smoothness(self):
        """Test local smoothness computation."""
        metric = InternalCoherenceMetric(
            similarity_metric="cosine",
            aggregation="mean"
        )

        # Create embeddings with high similarity
        step_embeddings = torch.randn(5, 128)
        step_embeddings = torch.nn.functional.normalize(step_embeddings, dim=1)

        smoothness = metric.compute_local_smoothness(step_embeddings)

        assert isinstance(smoothness, torch.Tensor)
        assert smoothness.item() >= -1.0 and smoothness.item() <= 1.0

    def test_goal_directedness(self):
        """Test goal-directedness computation."""
        metric = InternalCoherenceMetric()

        step_embeddings = torch.randn(5, 128)
        goal_embedding = torch.randn(128)

        goal_score = metric.compute_goal_directedness(
            step_embeddings, goal_embedding
        )

        assert isinstance(goal_score, torch.Tensor)
        assert goal_score.item() >= 0.0 and goal_score.item() <= 1.0

    def test_forward(self):
        """Test full forward pass."""
        metric = InternalCoherenceMetric()

        step_embeddings = torch.randn(5, 128)
        answer_embedding = torch.randn(128)

        result = metric(
            step_embeddings=step_embeddings,
            answer_embedding=answer_embedding
        )

        assert 'overall' in result
        assert 'smoothness' in result
        assert 'goal_directedness' in result
        assert 'semantic_density' in result

        # Check score range
        assert result['overall'].item() >= 0.0
        assert result['overall'].item() <= 1.0


class TestCrossModalCoherence:
    """Test cross-modal coherence metric."""

    def test_alignment(self):
        """Test step-image alignment."""
        metric = CrossModalCoherenceMetric()

        step_embeddings = torch.randn(5, 128)
        image_embeddings = torch.randn(2, 128)

        alignment = metric.compute_step_image_alignment(
            step_embeddings, image_embeddings
        )

        assert isinstance(alignment, torch.Tensor)
        assert alignment.item() >= -1.0 and alignment.item() <= 1.0

    def test_contrastive(self):
        """Test contrastive coherence."""
        metric = CrossModalCoherenceMetric(contrastive_margin=0.2)

        step_embeddings = torch.randn(5, 128)
        positive_embeddings = torch.randn(2, 128)
        negative_embeddings = torch.randn(3, 128)

        result = metric.compute_contrastive_coherence(
            step_embeddings,
            positive_embeddings,
            negative_embeddings
        )

        assert 'contrastive_score' in result
        assert 'positive_alignment' in result
        assert 'negative_alignment' in result

    def test_forward(self):
        """Test full forward pass."""
        metric = CrossModalCoherenceMetric()

        step_embeddings = torch.randn(5, 128)
        image_embeddings = torch.randn(2, 128)

        result = metric(
            step_embeddings=step_embeddings,
            image_embeddings=image_embeddings
        )

        assert 'overall' in result
        assert 'alignment' in result
        assert 'per_step_coherence' in result


class TestChainConfidenceScorer:
    """Test chain confidence scorer."""

    def test_confidence_computation(self):
        """Test confidence score computation."""
        scorer = ChainConfidenceScorer()

        step_embeddings = torch.randn(5, 128)
        image_embeddings = torch.randn(2, 128)

        result = scorer(
            step_embeddings=step_embeddings,
            image_embeddings=image_embeddings
        )

        assert 'confidence' in result
        assert 'internal' in result
        assert 'cross_modal' in result

        # Check confidence range
        confidence = result['confidence'].item()
        assert confidence >= 0.0 and confidence <= 1.0

    def test_batch_computation(self):
        """Test batch computation."""
        scorer = ChainConfidenceScorer()

        batch_step_embeddings = [
            torch.randn(5, 128),
            torch.randn(4, 128),
            torch.randn(6, 128)
        ]
        batch_image_embeddings = [
            torch.randn(2, 128),
            torch.randn(1, 128),
            torch.randn(3, 128)
        ]

        results = scorer.compute_for_batch(
            batch_step_embeddings=batch_step_embeddings,
            batch_image_embeddings=batch_image_embeddings
        )

        assert len(results) == 3
        for result in results:
            assert 'confidence' in result
            confidence = result['confidence'].item()
            assert confidence >= 0.0 and confidence <= 1.0


if __name__ == '__main__':
    pytest.main([__file__])
