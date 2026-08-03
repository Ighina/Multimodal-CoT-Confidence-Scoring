"""
Example usage of WeightedFrequency answer aggregation.

Demonstrates both exact matching (for multiple-choice) and semantic matching
(for open-ended questions) with confidence-weighted aggregation.
"""

import torch
import sys
sys.path.append('..')

from src.coherence.answer_aggregation import (
    WeightedFrequency,
    SelfConsistencyAggregator,
    EnsembleAggregator
)


def example_multiple_choice():
    """Example: Multiple-choice question with exact matching."""
    print("=" * 60)
    print("Example 1: Multiple-Choice Question (Exact Matching)")
    print("=" * 60)

    # Simulated answers from multiple reasoning chains
    answers = ["A", "B", "A", "C", "A", "B", "A", "D", "A"]

    # Confidence scores from chain_confidence.py for each answer
    confidence_scores = [0.85, 0.45, 0.92, 0.30, 0.78, 0.60, 0.88, 0.25, 0.90]

    # Create aggregator for exact matching
    aggregator = WeightedFrequency(mode="exact", normalize=True)

    # Aggregate answers
    result = aggregator.aggregate(answers, confidence_scores)

    print(f"\nAnswers: {answers}")
    print(f"Confidence scores: {[f'{s:.2f}' for s in confidence_scores]}")
    print(f"\nAggregated results:")
    for answer, weight in sorted(result.items(), key=lambda x: x[1], reverse=True):
        print(f"  {answer}: {weight:.4f} ({weight*100:.2f}%)")

    # Get best answer
    best_answer, best_confidence = aggregator.get_best_answer(
        answers, confidence_scores, return_confidence=True
    )
    print(f"\nBest answer: {best_answer} (confidence: {best_confidence:.4f})")

    print("\n")


def example_open_ended():
    """Example: Open-ended question with semantic similarity."""
    print("=" * 60)
    print("Example 2: Open-Ended Question (Semantic Similarity)")
    print("=" * 60)

    # Simulated open-ended answers (some are semantically similar)
    answers = [
        "The answer is 42",
        "It equals 42",
        "The result is 42",
        "The answer is 43",
        "42 is the answer",
        "The solution is 43",
        "42"
    ]

    confidence_scores = [0.85, 0.78, 0.92, 0.45, 0.88, 0.40, 0.90]

    # Simulate answer embeddings (in practice, these would come from an embedding model)
    # Here we create synthetic embeddings where similar answers have similar embeddings
    answer_embeddings = torch.tensor([
        [0.9, 0.1, 0.05, 0.0],  # "The answer is 42"
        [0.85, 0.15, 0.08, 0.02],  # "It equals 42" - similar to above
        [0.88, 0.12, 0.06, 0.01],  # "The result is 42" - similar to above
        [0.2, 0.8, 0.1, 0.05],  # "The answer is 43" - different
        [0.92, 0.08, 0.04, 0.0],  # "42 is the answer" - similar to first group
        [0.15, 0.85, 0.08, 0.03],  # "The solution is 43" - similar to "43" group
        [0.95, 0.05, 0.02, 0.0],  # "42" - very similar to "42" group
    ])

    # Create aggregator for semantic similarity
    aggregator = WeightedFrequency(
        mode="semantic",
        similarity_threshold=0.85,
        clustering_method="agglomerative",
        normalize=True
    )

    # Aggregate answers
    result = aggregator.aggregate(answers, confidence_scores, answer_embeddings)

    print(f"\nAnswers: {answers}")
    print(f"Confidence scores: {[f'{s:.2f}' for s in confidence_scores]}")
    print(f"\nAggregated results (by semantic cluster):")
    for answer, weight in sorted(result.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{answer}': {weight:.4f} ({weight*100:.2f}%)")

    # Get best answer
    best_answer, best_confidence = aggregator.get_best_answer(
        answers, confidence_scores, answer_embeddings, return_confidence=True
    )
    print(f"\nBest answer: '{best_answer}' (confidence: {best_confidence:.4f})")

    print("\n")


def example_self_consistency():
    """Example: Self-consistency with uniform vs confidence weighting."""
    print("=" * 60)
    print("Example 3: Self-Consistency Comparison")
    print("=" * 60)

    answers = ["A", "B", "A", "C", "A", "B", "A"]
    confidence_scores = [0.90, 0.30, 0.85, 0.25, 0.40, 0.35, 0.95]

    # Standard self-consistency (uniform weighting)
    sc_uniform = SelfConsistencyAggregator(
        use_confidence_weighting=False,
        mode="exact"
    )
    result_uniform = sc_uniform.aggregate(answers)

    # Confidence-weighted self-consistency
    sc_weighted = SelfConsistencyAggregator(
        use_confidence_weighting=True,
        mode="exact"
    )
    result_weighted = sc_weighted.aggregate(answers, confidence_scores)

    print(f"\nAnswers: {answers}")
    print(f"Confidence scores: {[f'{s:.2f}' for s in confidence_scores]}")

    print(f"\nStandard self-consistency (uniform weighting):")
    for answer, weight in sorted(result_uniform.items(), key=lambda x: x[1], reverse=True):
        print(f"  {answer}: {weight:.4f} ({weight*100:.2f}%)")

    print(f"\nConfidence-weighted self-consistency:")
    for answer, weight in sorted(result_weighted.items(), key=lambda x: x[1], reverse=True):
        print(f"  {answer}: {weight:.4f} ({weight*100:.2f}%)")

    print("\nObservation: With uniform weighting, A gets 4/7 = 57.14%")
    print("             With confidence weighting, A gets higher weight due to high confidence scores")

    print("\n")


def example_ensemble():
    """Example: Ensemble of aggregation strategies."""
    print("=" * 60)
    print("Example 4: Ensemble Aggregation")
    print("=" * 60)

    answers = ["42", "The answer is 42", "42", "43", "Forty-two"]
    confidence_scores = [0.90, 0.85, 0.88, 0.45, 0.82]

    # Simulate embeddings
    answer_embeddings = torch.tensor([
        [1.0, 0.0, 0.0],  # "42"
        [0.9, 0.1, 0.0],  # "The answer is 42"
        [1.0, 0.0, 0.0],  # "42"
        [0.0, 1.0, 0.0],  # "43"
        [0.85, 0.0, 0.15],  # "Forty-two"
    ])

    # Create multiple aggregators
    exact_agg = WeightedFrequency(mode="exact")
    semantic_agg = WeightedFrequency(mode="semantic", similarity_threshold=0.80)

    # Create ensemble
    ensemble = EnsembleAggregator(
        aggregators=[exact_agg, semantic_agg],
        weights=[0.5, 0.5]
    )

    # Compare results
    result_exact = exact_agg.aggregate(answers, confidence_scores)
    result_semantic = semantic_agg.aggregate(answers, confidence_scores, answer_embeddings)
    result_ensemble = ensemble.aggregate(answers, confidence_scores, answer_embeddings)

    print(f"\nAnswers: {answers}")
    print(f"Confidence scores: {[f'{s:.2f}' for s in confidence_scores]}")

    print(f"\nExact matching only:")
    for answer, weight in sorted(result_exact.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{answer}': {weight:.4f}")

    print(f"\nSemantic matching only:")
    for answer, weight in sorted(result_semantic.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{answer}': {weight:.4f}")

    print(f"\nEnsemble (50% exact + 50% semantic):")
    for answer, weight in sorted(result_ensemble.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{answer}': {weight:.4f}")

    print("\n")


if __name__ == "__main__":
    example_multiple_choice()
    example_open_ended()
    example_self_consistency()
    example_ensemble()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The WeightedFrequency aggregator provides:

1. Exact matching for multiple-choice questions
   - Groups identical answers
   - Sums confidence scores for each unique answer
   - Normalizes by total confidence sum

2. Semantic matching for open-ended questions
   - Clusters similar answers using embeddings
   - Uses DBSCAN or Agglomerative clustering
   - Selects representative answer per cluster
   - Aggregates confidence scores within clusters

3. Flexible integration with chain_confidence.py
   - Uses confidence scores as weights
   - Can combine with various confidence metrics
   - Supports ensemble aggregation strategies

4. Compatible with self-consistency approaches
   - Can use uniform or confidence weighting
   - Improves upon standard majority voting
""")
