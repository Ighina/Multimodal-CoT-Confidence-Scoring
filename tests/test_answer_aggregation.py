"""
Unit tests for answer aggregation functionality.
"""

import pytest
import torch
import sys
sys.path.append('..')

from src.coherence.answer_aggregation import (
    WeightedFrequency,
    SelfConsistencyAggregator,
    EnsembleAggregator
)


class TestWeightedFrequencyExact:
    """Tests for exact matching mode."""

    def test_basic_aggregation(self):
        """Test basic exact matching aggregation."""
        aggregator = WeightedFrequency(mode="exact", normalize=True)

        answers = ["A", "B", "A", "C", "A"]
        confidence_scores = [0.9, 0.5, 0.8, 0.3, 0.7]

        result = aggregator.aggregate(answers, confidence_scores)

        # A should have: 0.9 + 0.8 + 0.7 = 2.4
        # B should have: 0.5
        # C should have: 0.3
        # Total: 3.2
        assert "A" in result
        assert "B" in result
        assert "C" in result

        # Check normalization
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

        # A should have the highest weight
        assert result["A"] > result["B"]
        assert result["A"] > result["C"]

    def test_without_normalization(self):
        """Test aggregation without normalization."""
        aggregator = WeightedFrequency(mode="exact", normalize=False)

        answers = ["A", "A", "B"]
        confidence_scores = [0.6, 0.4, 0.5]

        result = aggregator.aggregate(answers, confidence_scores)

        # A should have: 0.6 + 0.4 = 1.0 (unnormalized)
        # B should have: 0.5 (unnormalized)
        assert abs(result["A"] - 1.0) < 1e-6
        assert abs(result["B"] - 0.5) < 1e-6

    def test_empty_input(self):
        """Test with empty input."""
        aggregator = WeightedFrequency(mode="exact")

        answers = []
        confidence_scores = []

        result = aggregator.aggregate(answers, confidence_scores)
        assert len(result) == 0

    def test_single_answer(self):
        """Test with single answer."""
        aggregator = WeightedFrequency(mode="exact", normalize=True)

        answers = ["A"]
        confidence_scores = [0.8]

        result = aggregator.aggregate(answers, confidence_scores)

        assert len(result) == 1
        assert abs(result["A"] - 1.0) < 1e-6  # Should be normalized to 1.0

    def test_get_best_answer(self):
        """Test getting best answer."""
        aggregator = WeightedFrequency(mode="exact")

        answers = ["A", "B", "A", "C"]
        confidence_scores = [0.9, 0.3, 0.8, 0.2]

        best_answer, best_conf = aggregator.get_best_answer(
            answers, confidence_scores, return_confidence=True
        )

        assert best_answer == "A"
        assert best_conf > 0

        # Test without confidence
        best_answer_only = aggregator.get_best_answer(
            answers, confidence_scores, return_confidence=False
        )
        assert best_answer_only == "A"


class TestWeightedFrequencySemantic:
    """Tests for semantic matching mode."""

    def test_semantic_clustering(self):
        """Test semantic similarity clustering."""
        aggregator = WeightedFrequency(
            mode="semantic",
            similarity_threshold=0.85,
            clustering_method="agglomerative",
            normalize=True
        )

        answers = ["answer 1", "answer 2", "answer 3"]
        confidence_scores = [0.8, 0.6, 0.5]

        # Create embeddings where answers 1 and 2 are similar
        answer_embeddings = torch.tensor([
            [1.0, 0.0, 0.0],  # answer 1
            [0.95, 0.05, 0.0],  # answer 2 - similar to answer 1
            [0.0, 0.0, 1.0],  # answer 3 - different
        ])

        result = aggregator.aggregate(answers, confidence_scores, answer_embeddings)

        # Should have 2 clusters: (1,2) and (3)
        assert len(result) == 2

        # Check normalization
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_semantic_single_cluster(self):
        """Test when all answers cluster together."""
        aggregator = WeightedFrequency(
            mode="semantic",
            similarity_threshold=0.5,  # Low threshold = more clustering
            clustering_method="agglomerative",
            normalize=True
        )

        answers = ["A", "B", "C"]
        confidence_scores = [0.5, 0.3, 0.2]

        # All embeddings very similar
        answer_embeddings = torch.tensor([
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
        ])

        result = aggregator.aggregate(answers, confidence_scores, answer_embeddings)

        # Should cluster into 1 group
        assert len(result) == 1

        # Total confidence should be normalized to 1
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_missing_embedding_model_error(self):
        """Test error when embeddings not provided and no model available."""
        aggregator = WeightedFrequency(
            mode="semantic",
            embedding_model=None
        )

        answers = ["A", "B"]
        confidence_scores = [0.5, 0.5]

        with pytest.raises(ValueError, match="Must provide either answer_embeddings or embedding_model"):
            aggregator.aggregate(answers, confidence_scores, answer_embeddings=None)


class TestSelfConsistencyAggregator:
    """Tests for self-consistency aggregator."""

    def test_uniform_weighting(self):
        """Test uniform weighting (standard self-consistency)."""
        aggregator = SelfConsistencyAggregator(
            use_confidence_weighting=False,
            mode="exact"
        )

        answers = ["A", "A", "B"]
        confidence_scores = [0.9, 0.1, 0.8]  # Varying confidences

        result = aggregator.aggregate(answers, confidence_scores)

        # With uniform weighting: A gets 2/3, B gets 1/3
        assert abs(result["A"] - 2/3) < 1e-6
        assert abs(result["B"] - 1/3) < 1e-6

    def test_confidence_weighting(self):
        """Test confidence-weighted self-consistency."""
        aggregator = SelfConsistencyAggregator(
            use_confidence_weighting=True,
            mode="exact"
        )

        answers = ["A", "A", "B"]
        confidence_scores = [0.9, 0.1, 0.8]

        result = aggregator.aggregate(answers, confidence_scores)

        # A gets 0.9 + 0.1 = 1.0, B gets 0.8
        # Normalized: A = 1.0/1.8, B = 0.8/1.8
        expected_A = 1.0 / 1.8
        expected_B = 0.8 / 1.8

        assert abs(result["A"] - expected_A) < 1e-6
        assert abs(result["B"] - expected_B) < 1e-6


class TestEnsembleAggregator:
    """Tests for ensemble aggregator."""

    def test_ensemble_combination(self):
        """Test ensemble combining multiple aggregators."""
        exact_agg = WeightedFrequency(mode="exact", normalize=True)
        exact_agg2 = WeightedFrequency(mode="exact", normalize=True)

        ensemble = EnsembleAggregator(
            aggregators=[exact_agg, exact_agg2],
            weights=[0.5, 0.5]
        )

        answers = ["A", "A", "B"]
        confidence_scores = [0.6, 0.4, 0.5]

        result = ensemble.aggregate(answers, confidence_scores)

        # Since both aggregators are identical, result should be same as single
        single_result = exact_agg.aggregate(answers, confidence_scores)

        for ans in result:
            assert abs(result[ans] - single_result[ans]) < 1e-6

    def test_ensemble_uniform_weights(self):
        """Test ensemble with default uniform weights."""
        agg1 = WeightedFrequency(mode="exact")
        agg2 = WeightedFrequency(mode="exact")

        ensemble = EnsembleAggregator(aggregators=[agg1, agg2])

        answers = ["A", "B"]
        confidence_scores = [0.7, 0.3]

        result = ensemble.aggregate(answers, confidence_scores)

        # Should work without errors
        assert "A" in result
        assert "B" in result

    def test_ensemble_weight_validation(self):
        """Test ensemble weight validation."""
        agg1 = WeightedFrequency(mode="exact")
        agg2 = WeightedFrequency(mode="exact")

        with pytest.raises(ValueError, match="Number of weights must match"):
            EnsembleAggregator(
                aggregators=[agg1, agg2],
                weights=[0.5]  # Wrong number of weights
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_mismatched_lengths(self):
        """Test error when answers and scores have different lengths."""
        aggregator = WeightedFrequency(mode="exact")

        answers = ["A", "B"]
        confidence_scores = [0.5]  # Wrong length

        with pytest.raises(ValueError, match="Number of answers must match"):
            aggregator.aggregate(answers, confidence_scores)

    def test_torch_tensor_conversion(self):
        """Test that torch tensors are handled correctly."""
        aggregator = WeightedFrequency(mode="exact")

        answers = ["A", "B", "A"]
        confidence_scores = torch.tensor([0.7, 0.3, 0.6])

        result = aggregator.aggregate(answers, confidence_scores)

        assert "A" in result
        assert "B" in result

    def test_unknown_mode(self):
        """Test error for unknown aggregation mode."""
        aggregator = WeightedFrequency(mode="invalid_mode")

        answers = ["A", "B"]
        confidence_scores = [0.5, 0.5]

        with pytest.raises(ValueError, match="Unknown mode"):
            aggregator.aggregate(answers, confidence_scores)

    def test_unknown_clustering_method(self):
        """Test error for unknown clustering method."""
        aggregator = WeightedFrequency(
            mode="semantic",
            clustering_method="invalid_method"
        )

        answers = ["A", "B"]
        confidence_scores = [0.5, 0.5]
        answer_embeddings = torch.randn(2, 10)

        with pytest.raises(ValueError, match="Unknown clustering method"):
            aggregator.aggregate(answers, confidence_scores, answer_embeddings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
