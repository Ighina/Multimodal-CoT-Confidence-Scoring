"""
Quick start example for multimodal CoT confidence scoring.

This script demonstrates the basic workflow:
1. Setup encoders
2. Encode reasoning chain
3. Compute confidence score
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image

from src.embeddings import TextEncoder, MultimodalEncoder
from src.coherence import (
    InternalCoherenceMetric,
    CrossModalCoherenceMetric,
    ChainConfidenceScorer
)


def main():
    """Run quick start example."""

    print("="*60)
    print("Multimodal CoT Confidence Scoring - Quick Start")
    print("="*60)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # 1. Initialize encoders
    print("\n1. Initializing encoders...")
    text_encoder = TextEncoder(
        model_name="sentence-transformers/all-mpnet-base-v2",
        device=device
    )

    multimodal_encoder = MultimodalEncoder(
        model_name="openai/clip-vit-large-patch14",
        device=device
    )

    print(f"   Text embedding dimension: {text_encoder.get_embedding_dim()}")
    print(f"   Multimodal embedding dimension: {multimodal_encoder.get_embedding_dim()}")

    # 2. Example reasoning chain
    print("\n2. Example reasoning chain:")
    question = "What animal is shown in the image?"

    reasoning_steps = [
        "First, let's examine the key features visible in the image.",
        "The image shows a four-legged animal with distinctive features.",
        "I can observe fur patterns and physical characteristics.",
        "Based on these visual observations, this appears to be a cat.",
        "Therefore, the answer is: cat"
    ]

    print(f"   Question: {question}")
    print(f"   Number of reasoning steps: {len(reasoning_steps)}")
    for i, step in enumerate(reasoning_steps, 1):
        print(f"   Step {i}: {step}")

    # 3. Encode the reasoning chain
    print("\n3. Encoding reasoning chain...")
    step_embeddings = text_encoder.encode_cot_steps(
        steps=reasoning_steps,
        question=question
    )
    print(f"   Step embeddings shape: {step_embeddings.shape}")

    # 4. Encode images (mock for this example)
    print("\n4. Encoding images...")
    # In practice, you would load actual images:
    # images = [Image.open("path/to/image.jpg")]
    # image_embeddings = multimodal_encoder.encode_images(images)

    # For demo, create random embeddings
    image_embeddings = torch.randn(
        1, multimodal_encoder.get_embedding_dim()
    ).to(device)
    print(f"   Image embeddings shape: {image_embeddings.shape}")

    # 5. Setup coherence metrics
    print("\n5. Setting up coherence metrics...")
    internal_metric = InternalCoherenceMetric(
        similarity_metric="cosine",
        aggregation="mean",
        goal_directedness_weight=0.3,
        smoothness_weight=0.7
    )

    cross_modal_metric = CrossModalCoherenceMetric(
        similarity_metric="cosine",
        contrastive_margin=0.2,
        use_attention=True
    )

    # 6. Create confidence scorer
    print("\n6. Creating confidence scorer...")
    scorer = ChainConfidenceScorer(
        internal_metric=internal_metric,
        cross_modal_metric=cross_modal_metric,
        internal_weight=0.5,
        cross_modal_weight=0.5,
        density_weight=0.0
    )

    # 7. Compute confidence
    print("\n7. Computing confidence score...")
    with torch.no_grad():
        result = scorer(
            step_embeddings=step_embeddings,
            image_embeddings=image_embeddings
        )

    # 8. Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    confidence = result['confidence'].item()
    print(f"\nOverall Confidence Score: {confidence:.4f}")

    print("\nInternal Coherence Components:")
    for key, value in result['internal'].items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"  {key}: {value.item():.4f}")

    print("\nCross-Modal Coherence Components:")
    for key, value in result['cross_modal'].items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"  {key}: {value.item():.4f}")

    print("\nComponent Weights:")
    weights = result['weights']
    print(f"  Internal:     {weights[0].item():.2f}")
    print(f"  Cross-modal:  {weights[1].item():.2f}")
    print(f"  Density:      {weights[2].item():.2f}")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if confidence > 0.7:
        interpretation = "HIGH - The reasoning appears coherent and well-grounded"
    elif confidence > 0.5:
        interpretation = "MEDIUM - The reasoning is acceptable but could be stronger"
    else:
        interpretation = "LOW - The reasoning may have issues or inconsistencies"

    print(f"\nConfidence Level: {interpretation}")

    print("\n" + "="*60)
    print("Next Steps:")
    print("  1. Load real images and generate actual CoT chains")
    print("  2. Evaluate on UNO-Bench or your own dataset")
    print("  3. Compare with baseline methods")
    print("  4. Run ablation studies")
    print("  5. Train learned confidence head for better calibration")
    print("="*60)


if __name__ == "__main__":
    main()
