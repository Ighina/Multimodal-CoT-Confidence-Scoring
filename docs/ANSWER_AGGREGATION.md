# Answer Aggregation with Confidence Weighting

This document explains how to use the `WeightedFrequency` aggregation system to combine multiple reasoning chain outputs using confidence scores from `chain_confidence.py`.

## Overview

The answer aggregation module provides confidence-weighted voting for combining multiple reasoning chain outputs. It supports both:

1. **Multiple-choice questions**: Uses exact string matching to group identical answers
2. **Open-ended questions**: Uses semantic similarity and clustering to group similar answers

The key innovation is using confidence scores (computed by `ChainConfidenceScorer`) as weights when aggregating answers, giving more influence to high-confidence chains.

## Quick Start

### Multiple-Choice Questions (Exact Matching)

```python
from src.coherence.answer_aggregation import WeightedFrequency

# Create aggregator for exact matching
aggregator = WeightedFrequency(mode="exact", normalize=True)

# Your answers and confidence scores from multiple reasoning chains
answers = ["A", "B", "A", "C", "A"]
confidence_scores = [0.85, 0.45, 0.92, 0.30, 0.78]

# Aggregate
result = aggregator.aggregate(answers, confidence_scores)
# Result: {"A": 0.737, "B": 0.130, "C": 0.087}
# (A gets: (0.85 + 0.92 + 0.78) / (0.85 + 0.45 + 0.92 + 0.30 + 0.78) = 2.55 / 3.46)

# Get best answer
best_answer, best_confidence = aggregator.get_best_answer(
    answers, confidence_scores, return_confidence=True
)
print(f"Best answer: {best_answer} (confidence: {best_confidence:.3f})")
```

### Open-Ended Questions (Semantic Similarity)

```python
import torch
from src.coherence.answer_aggregation import WeightedFrequency

# Create aggregator for semantic matching
aggregator = WeightedFrequency(
    mode="semantic",
    similarity_threshold=0.85,  # Answers with similarity > 0.85 are grouped
    clustering_method="agglomerative",  # or "dbscan"
    normalize=True
)

# Your answers with varying phrasings
answers = [
    "The answer is 42",
    "It equals 42",
    "The result is 42",
    "The answer is 43",
    "42 is the correct answer"
]

confidence_scores = [0.85, 0.78, 0.92, 0.45, 0.88]

# Pre-computed answer embeddings (from your embedding model)
answer_embeddings = get_embeddings(answers)  # Shape: (5, embed_dim)

# Aggregate - similar answers are clustered together
result = aggregator.aggregate(answers, confidence_scores, answer_embeddings)

# Get best answer (returns representative from highest-confidence cluster)
best_answer, best_confidence = aggregator.get_best_answer(
    answers, confidence_scores, answer_embeddings, return_confidence=True
)
```

## Complete Pipeline with ChainConfidenceScorer

Here's how to integrate with the full confidence scoring pipeline:

```python
import torch
from src.coherence.chain_confidence import ChainConfidenceScorer
from src.coherence.answer_aggregation import WeightedFrequency

# Step 1: Generate multiple reasoning chains (e.g., 10 samples)
num_chains = 10
all_answers = []
all_confidence_scores = []

# Step 2: Compute confidence for each chain
confidence_scorer = ChainConfidenceScorer(
    internal_weight=0.5,
    cross_modal_weight=0.4,
    density_weight=0.1
)

for chain_idx in range(num_chains):
    # Get embeddings for this chain
    step_embeddings = get_step_embeddings(chain_idx)  # (num_steps, embed_dim)
    modal_embeddings = get_modal_embeddings(chain_idx)  # (num_modals, embed_dim)

    # Compute confidence score
    results = confidence_scorer(
        step_embeddings=step_embeddings,
        modal_embeddings=modal_embeddings
    )

    confidence_score = results['confidence'].item()
    answer = extract_answer(chain_idx)

    all_answers.append(answer)
    all_confidence_scores.append(confidence_score)

# Step 3: Aggregate answers using confidence weights
aggregator = WeightedFrequency(mode="exact", normalize=True)
aggregated_results = aggregator.aggregate(all_answers, all_confidence_scores)

# Step 4: Get final answer
best_answer, final_confidence = aggregator.get_best_answer(
    all_answers, all_confidence_scores, return_confidence=True
)

print(f"Final answer: {best_answer}")
print(f"Aggregated confidence: {final_confidence:.3f}")
print(f"\nAll candidates:")
for answer, weight in sorted(aggregated_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {answer}: {weight:.3f}")
```

## Algorithm Details

### Exact Matching Mode

1. **Grouping**: Groups answers by exact string equality
2. **Weighting**: Sums confidence scores for each unique answer
3. **Normalization**: Divides by total sum of all confidence scores

Mathematically:
```
weight(answer_i) = Σ confidence_j / Σ confidence_all
                   for all j where answer_j == answer_i
```

### Semantic Similarity Mode

1. **Embedding**: Converts answers to embedding vectors
2. **Clustering**: Uses DBSCAN or Agglomerative Clustering to group similar answers
   - Similarity threshold controls cluster granularity
   - Cosine similarity metric
3. **Aggregation**: Sums confidence scores within each cluster
4. **Representative selection**: Chooses the answer with highest individual confidence from each cluster
5. **Normalization**: Normalizes cluster weights by total sum

### Clustering Methods

**DBSCAN** (Density-Based Spatial Clustering):
- Good for finding clusters of varying shapes
- `eps = 1.0 - similarity_threshold` (distance threshold)
- `min_samples` controls minimum cluster size
- Can identify noise points (outliers)

**Agglomerative Clustering**:
- Hierarchical clustering with bottom-up approach
- More stable for small datasets
- `distance_threshold = 1.0 - similarity_threshold`
- Uses average linkage with cosine metric

## Advanced Features

### Self-Consistency Comparison

Compare standard self-consistency (uniform weighting) vs confidence-weighted:

```python
from src.coherence.answer_aggregation import SelfConsistencyAggregator

# Standard self-consistency
sc_uniform = SelfConsistencyAggregator(
    use_confidence_weighting=False,
    mode="exact"
)
result_uniform = sc_uniform.aggregate(answers)

# Confidence-weighted
sc_weighted = SelfConsistencyAggregator(
    use_confidence_weighting=True,
    mode="exact"
)
result_weighted = sc_weighted.aggregate(answers, confidence_scores)
```

### Ensemble Aggregation

Combine multiple aggregation strategies:

```python
from src.coherence.answer_aggregation import EnsembleAggregator

# Create multiple aggregators
exact_agg = WeightedFrequency(mode="exact")
semantic_agg = WeightedFrequency(mode="semantic", similarity_threshold=0.85)

# Ensemble them
ensemble = EnsembleAggregator(
    aggregators=[exact_agg, semantic_agg],
    weights=[0.4, 0.6]  # 40% exact, 60% semantic
)

# Aggregate
result = ensemble.aggregate(answers, confidence_scores, answer_embeddings)
```

## Configuration Options

### WeightedFrequency Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "exact" | "exact" for multiple-choice, "semantic" for open-ended |
| `similarity_threshold` | float | 0.85 | Minimum similarity to cluster answers together (semantic mode) |
| `clustering_method` | str | "dbscan" | "dbscan" or "agglomerative" |
| `min_cluster_size` | int | 1 | Minimum samples per cluster (DBSCAN only) |
| `embedding_model` | nn.Module | None | Model to compute embeddings (if not pre-computed) |
| `normalize` | bool | True | Whether to normalize weights by total sum |

### When to Use Each Mode

**Use Exact Mode When:**
- Multiple-choice questions (A/B/C/D)
- Binary questions (Yes/No)
- Numerical answers with fixed precision
- Short, standardized answer formats

**Use Semantic Mode When:**
- Open-ended questions
- Free-form text answers
- Answers with paraphrasing variations
- Long-form explanations
- Multiple valid phrasings of same answer

## Best Practices

1. **Generate diverse chains**: Use different sampling strategies or temperatures to get varied reasoning paths
2. **Sufficient samples**: Use at least 5-10 chains for reliable aggregation
3. **Confidence calibration**: Ensure confidence scores are well-calibrated (0-1 range, meaningful differences)
4. **Embedding quality**: For semantic mode, use high-quality embeddings (e.g., from a fine-tuned model)
5. **Threshold tuning**: Adjust `similarity_threshold` based on your answer distribution
   - Lower (0.7-0.8): More aggressive clustering, groups more answers
   - Higher (0.9-0.95): Conservative clustering, keeps answers separate
6. **Validation**: Compare exact vs semantic modes to verify clustering quality

## Examples

See `examples/answer_aggregation_example.py` for complete working examples.

## Integration with Confidence Metrics

The aggregation module is designed to work with confidence scores from:

- `ChainConfidenceScorer`: Overall chain confidence combining internal and cross-modal coherence
- `InternalCoherenceMetric`: Text-based coherence scores
- `CrossModalCoherenceMetric`: Image-text alignment scores
- Custom confidence metrics

All confidence scores should be in range [0, 1] where higher means more confident.

## Performance Considerations

- **Exact mode**: O(n) where n is number of answers
- **Semantic mode**: O(n²) for pairwise similarity, O(n log n) for clustering
- **Embedding computation**: Most expensive if not pre-computed
- **Recommendation**: Pre-compute embeddings when possible

## Testing

Run the test suite:

```bash
cd tests
pytest test_answer_aggregation.py -v
```

## Future Extensions

Potential enhancements:
- Adaptive similarity thresholds based on answer distribution
- Confidence calibration before aggregation
- Multi-stage aggregation (exact → semantic refinement)
- Uncertainty quantification for final answer
- Support for ranked/multi-label answers
