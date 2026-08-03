"""
Complete pipeline example: From reasoning chains to aggregated answer.

This demonstrates the full workflow:
1. Generate multiple reasoning chains
2. Compute confidence scores using ChainConfidenceScorer
3. Aggregate answers using WeightedFrequency
4. Compare with baseline (uniform weighting)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import sys
sys.path.append('..')

from src.coherence.chain_confidence import ChainConfidenceScorer
from src.coherence.answer_aggregation import WeightedFrequency, SelfConsistencyAggregator


class ReasoningChainSimulator:
    """
    Simulates multiple reasoning chains with varying quality.

    In practice, these would come from your actual model generating
    multiple samples via temperature sampling, beam search, etc.
    """

    def __init__(self, embed_dim: int = 768):
        self.embed_dim = embed_dim

    def generate_chain(
        self,
        num_steps: int,
        num_modals: int,
        quality: str = "high"
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a simulated reasoning chain.

        Args:
            num_steps: Number of reasoning steps
            num_modals: Number of modal inputs (images, audio, etc.)
            quality: "high", "medium", or "low" quality chain

        Returns:
            Dictionary with embeddings and answer
        """
        # Quality affects coherence of embeddings
        if quality == "high":
            noise_scale = 0.1
            answer_correctness = 0.9  # 90% chance of correct answer
        elif quality == "medium":
            noise_scale = 0.3
            answer_correctness = 0.5
        else:  # low
            noise_scale = 0.6
            answer_correctness = 0.2

        # Generate step embeddings with some coherence
        base_direction = torch.randn(self.embed_dim)
        base_direction = base_direction / base_direction.norm()

        step_embeddings = []
        for i in range(num_steps):
            # Steps progress in a general direction with some noise
            step_emb = base_direction * (0.5 + i * 0.1) + torch.randn(self.embed_dim) * noise_scale
            step_emb = step_emb / step_emb.norm()
            step_embeddings.append(step_emb)

        step_embeddings = torch.stack(step_embeddings)

        # Generate modal embeddings (aligned with high-quality chains)
        modal_embeddings = []
        for _ in range(num_modals):
            if quality == "high":
                # High quality: modals aligned with reasoning direction
                modal_emb = base_direction + torch.randn(self.embed_dim) * 0.2
            else:
                # Lower quality: less alignment
                modal_emb = torch.randn(self.embed_dim)
            modal_emb = modal_emb / modal_emb.norm()
            modal_embeddings.append(modal_emb)

        modal_embeddings = torch.stack(modal_embeddings)

        # Generate answer (correct answer is "A", incorrect could be "B", "C", "D")
        if np.random.rand() < answer_correctness:
            answer = "A"  # Correct answer
        else:
            answer = np.random.choice(["B", "C", "D"])  # Wrong answers

        return {
            'step_embeddings': step_embeddings,
            'modal_embeddings': modal_embeddings,
            'answer': answer
        }


def run_complete_pipeline(
    num_chains: int = 10,
    num_steps: int = 5,
    num_modals: int = 3,
    embed_dim: int = 768
) -> Tuple[str, float, Dict]:
    """
    Run the complete pipeline from chain generation to aggregated answer.

    Args:
        num_chains: Number of reasoning chains to generate
        num_steps: Steps per chain
        num_modals: Modal inputs per chain
        embed_dim: Embedding dimensionality

    Returns:
        Tuple of (best_answer, confidence, detailed_results)
    """
    print("=" * 80)
    print("COMPLETE PIPELINE: Chain-of-Thought with Confidence-Weighted Aggregation")
    print("=" * 80)

    # Step 1: Initialize components
    print("\n[1] Initializing components...")
    chain_scorer = ChainConfidenceScorer(
        internal_weight=0.5,
        cross_modal_weight=0.4,
        density_weight=0.1
    )

    aggregator = WeightedFrequency(mode="exact", normalize=True)
    baseline_aggregator = SelfConsistencyAggregator(
        use_confidence_weighting=False,
        mode="exact"
    )

    simulator = ReasoningChainSimulator(embed_dim=embed_dim)

    # Step 2: Generate chains with varying quality
    print(f"\n[2] Generating {num_chains} reasoning chains...")
    chains = []

    # Mix of quality levels
    quality_distribution = (
        ["high"] * (num_chains // 2) +  # 50% high quality
        ["medium"] * (num_chains // 3) +  # 33% medium
        ["low"] * (num_chains - num_chains // 2 - num_chains // 3)  # rest low
    )

    for i, quality in enumerate(quality_distribution):
        chain = simulator.generate_chain(
            num_steps=num_steps,
            num_modals=num_modals,
            quality=quality
        )
        chain['quality'] = quality
        chain['id'] = i
        chains.append(chain)

    print(f"   Generated: {quality_distribution.count('high')} high, "
          f"{quality_distribution.count('medium')} medium, "
          f"{quality_distribution.count('low')} low quality chains")

    # Step 3: Compute confidence scores
    print("\n[3] Computing confidence scores for each chain...")
    all_answers = []
    all_confidences = []
    chain_details = []

    for chain in chains:
        # Compute confidence using ChainConfidenceScorer
        with torch.no_grad():
            results = chain_scorer(
                step_embeddings=chain['step_embeddings'],
                modal_embeddings=chain['modal_embeddings']
            )

        confidence = results['confidence'].item()
        answer = chain['answer']

        all_answers.append(answer)
        all_confidences.append(confidence)

        chain_details.append({
            'id': chain['id'],
            'quality': chain['quality'],
            'answer': answer,
            'confidence': confidence,
            'internal_score': results['internal']['overall'].item(),
            'cross_modal_score': results['cross_modal']['overall'].item()
        })

        print(f"   Chain {chain['id']:2d} [{chain['quality']:6s}]: "
              f"Answer={answer}, Confidence={confidence:.3f}")

    # Step 4: Aggregate with confidence weighting
    print("\n[4] Aggregating answers...")

    # Confidence-weighted aggregation
    weighted_result = aggregator.aggregate(all_answers, all_confidences)
    best_answer_weighted, conf_weighted = aggregator.get_best_answer(
        all_answers, all_confidences, return_confidence=True
    )

    # Baseline: uniform weighting (standard self-consistency)
    baseline_result = baseline_aggregator.aggregate(all_answers)
    best_answer_baseline, conf_baseline = baseline_aggregator.get_best_answer(
        all_answers, return_confidence=True
    )

    print("\n   Confidence-weighted aggregation:")
    for answer, weight in sorted(weighted_result.items(), key=lambda x: x[1], reverse=True):
        print(f"      {answer}: {weight:.3f} ({weight*100:.1f}%)")
    print(f"   → Best answer: {best_answer_weighted} (confidence: {conf_weighted:.3f})")

    print("\n   Baseline (uniform weighting):")
    for answer, weight in sorted(baseline_result.items(), key=lambda x: x[1], reverse=True):
        print(f"      {answer}: {weight:.3f} ({weight*100:.1f}%)")
    print(f"   → Best answer: {best_answer_baseline} (confidence: {conf_baseline:.3f})")

    # Step 5: Analysis
    print("\n[5] Analysis...")

    # Count answers by quality level
    quality_stats = {}
    for detail in chain_details:
        quality = detail['quality']
        if quality not in quality_stats:
            quality_stats[quality] = {'answers': [], 'confidences': []}
        quality_stats[quality]['answers'].append(detail['answer'])
        quality_stats[quality]['confidences'].append(detail['confidence'])

    print("\n   Answer distribution by quality:")
    for quality in ['high', 'medium', 'low']:
        if quality in quality_stats:
            answers = quality_stats[quality]['answers']
            avg_conf = np.mean(quality_stats[quality]['confidences'])
            correct_rate = answers.count('A') / len(answers) * 100
            print(f"      {quality.capitalize():6s}: {len(answers)} chains, "
                  f"avg_conf={avg_conf:.3f}, correct_rate={correct_rate:.1f}%")

    # Impact analysis
    print("\n   Impact of confidence weighting:")
    improvement = conf_weighted - conf_baseline
    print(f"      Confidence change: {improvement:+.3f}")

    if best_answer_weighted == best_answer_baseline:
        print(f"      Same answer selected, but with adjusted confidence")
    else:
        print(f"      Different answer selected: {best_answer_baseline} → {best_answer_weighted}")

    # Average confidence by answer
    answer_conf_map = {}
    for ans, conf in zip(all_answers, all_confidences):
        if ans not in answer_conf_map:
            answer_conf_map[ans] = []
        answer_conf_map[ans].append(conf)

    print("\n   Average confidence by answer:")
    for answer in sorted(answer_conf_map.keys()):
        avg = np.mean(answer_conf_map[answer])
        count = len(answer_conf_map[answer])
        print(f"      {answer}: {avg:.3f} (n={count})")

    # Return results
    detailed_results = {
        'weighted_result': weighted_result,
        'baseline_result': baseline_result,
        'chain_details': chain_details,
        'quality_stats': quality_stats
    }

    return best_answer_weighted, conf_weighted, detailed_results


def run_semantic_example(num_chains: int = 8, embed_dim: int = 768):
    """
    Example with semantic similarity for open-ended questions.
    """
    print("\n\n" + "=" * 80)
    print("SEMANTIC AGGREGATION: Open-Ended Question")
    print("=" * 80)

    # Simulate open-ended answers with paraphrasing
    answers_pool = {
        "42": ["42", "The answer is 42", "It equals 42", "42 is correct"],
        "43": ["43", "The answer is 43", "I think it's 43"],
        "44": ["44", "Forty-four"]
    }

    # Generate chains
    print(f"\n[1] Generating {num_chains} chains with open-ended answers...")
    all_answers = []
    all_confidences = []
    all_embeddings = []

    for i in range(num_chains):
        # Select answer type based on quality
        if i < num_chains * 0.7:  # 70% answer "42" (correct)
            answer_type = "42"
            confidence = np.random.uniform(0.7, 0.95)
        elif i < num_chains * 0.9:  # 20% answer "43"
            answer_type = "43"
            confidence = np.random.uniform(0.3, 0.6)
        else:  # 10% answer "44"
            answer_type = "44"
            confidence = np.random.uniform(0.2, 0.5)

        # Pick random phrasing
        answer = np.random.choice(answers_pool[answer_type])
        all_answers.append(answer)
        all_confidences.append(confidence)

        # Create embedding (similar embeddings for same semantic meaning)
        base_emb = torch.randn(embed_dim)
        # Add type-specific signal
        if answer_type == "42":
            base_emb[0] = 5.0  # Strong signal for "42"
        elif answer_type == "43":
            base_emb[0] = -5.0  # Different signal
        else:
            base_emb[1] = 5.0  # Different dimension

        # Add small noise for paraphrasing variation
        embedding = base_emb + torch.randn(embed_dim) * 0.3
        embedding = embedding / embedding.norm()
        all_embeddings.append(embedding)

        print(f"   Chain {i:2d}: '{answer}' (confidence={confidence:.3f})")

    answer_embeddings = torch.stack(all_embeddings)

    # Aggregate with semantic similarity
    print("\n[2] Aggregating with semantic similarity...")
    semantic_aggregator = WeightedFrequency(
        mode="semantic",
        similarity_threshold=0.85,
        clustering_method="agglomerative",
        normalize=True
    )

    result = semantic_aggregator.aggregate(
        all_answers, all_confidences, answer_embeddings
    )

    print("\n   Clustered results:")
    for answer, weight in sorted(result.items(), key=lambda x: x[1], reverse=True):
        print(f"      '{answer}': {weight:.3f} ({weight*100:.1f}%)")

    best_answer, best_conf = semantic_aggregator.get_best_answer(
        all_answers, all_confidences, answer_embeddings, return_confidence=True
    )

    print(f"\n   → Best answer: '{best_answer}' (confidence: {best_conf:.3f})")

    # Compare with exact matching
    print("\n[3] Comparison with exact matching (no clustering)...")
    exact_aggregator = WeightedFrequency(mode="exact", normalize=True)
    exact_result = exact_aggregator.aggregate(all_answers, all_confidences)

    print(f"   Found {len(exact_result)} unique answers (no clustering):")
    for answer, weight in sorted(exact_result.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      '{answer}': {weight:.3f}")

    print(f"\n   Semantic clustering reduced {len(exact_result)} unique answers "
          f"to {len(result)} semantic clusters")


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run multiple-choice example
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Multiple-Choice Question")
    print("=" * 80)
    best_answer, confidence, details = run_complete_pipeline(
        num_chains=10,
        num_steps=5,
        num_modals=3
    )

    print("\n" + "=" * 80)
    print(f"FINAL RESULT: {best_answer} (confidence: {confidence:.3f})")
    print("=" * 80)

    # Run open-ended example
    run_semantic_example(num_chains=12)

    print("\n\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. Confidence-weighted aggregation gives more weight to high-quality chains
2. Low-confidence chains (from poor reasoning) have less influence
3. This approach outperforms uniform weighting when confidence is calibrated
4. Semantic clustering enables aggregation for open-ended questions
5. The confidence scores from ChainConfidenceScorer provide meaningful weights
    """)
