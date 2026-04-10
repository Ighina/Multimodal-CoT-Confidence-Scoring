"""
Refactored experiment runner for multimodal CoT confidence scoring.

This module provides a simplified, modular interface for running experiments
by delegating to specialized modules:
- create_cots.py: CoT generation
- embedding_extraction.py: Embedding extraction
- coherence_modeling.py: Coherence scoring

Supports image, audio, video, and text-based reasoning pipelines.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.uno_bench_loader import UNOBenchLoader

from create_cots import generate_cots, load_cots, save_cots, filter_cots_by_count
from embedding_extraction import (
    setup_encoders,
    extract_embeddings_batch,
    load_embeddings,
    save_embeddings,
)
from coherence_modeling import (
    setup_coherence_metrics,
    setup_confidence_scorer,
    compute_coherence_scores,
    save_scores,
)


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logger for the experiment.

    Args:
        log_file: Optional path to log file

    Returns:
        Logger instance
    """
    logger = logging.getLogger("multimodal_cot_experiment")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Run modular multimodal CoT confidence scoring experiments"
    )

    # Dataset arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="uno-bench",
        help="Path to UNO-Bench dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--modality_filter", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    # CoT generation arguments
    parser.add_argument(
        "--cot_model_name",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct"
    )
    parser.add_argument("--cot_model_type", type=str, default="qwen2_audio")
    parser.add_argument("--use_openai_api", action="store_true")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--num_chains", type=int, default=20)
    parser.add_argument("--num_chains_for_scoring", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    # Encoder arguments
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument("--multimodal_encoder", type=str, default=None)
    parser.add_argument("--audio_encoder", type=str, default=None)
    parser.add_argument(
        "--omnimodal_encoder",
        type=str,
        default=None,
        help="Omnimodal encoder (e.g., LCO-Embedding/LCO-Embedding-Omni-7B)"
    )

    # Coherence metric arguments
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "dot"]
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "min", "max"]
    )
    parser.add_argument("--goal_directedness_weight", type=float, default=0.6)
    parser.add_argument("--smoothness_weight", type=float, default=0.4)
    parser.add_argument("--contrastive_margin", type=float, default=0.2)
    parser.add_argument("--use_attention", action="store_true")

    # Confidence scoring arguments
    parser.add_argument("--use_density_model", action="store_true")
    parser.add_argument(
        "--density_model_type",
        type=str,
        default="kde",
        choices=["kde", "gmm"]
    )
    parser.add_argument("--use_nli", action="store_true")
    parser.add_argument(
        "--nli_model_name",
        type=str,
        default="cross-encoder/nli-deberta-v3-large"
    )
    parser.add_argument("--use_prm", action="store_true")
    parser.add_argument("--prm_model_name", type=str, default="sail/ActPRM-X")

    # Weights for final confidence aggregation
    parser.add_argument("--internal_weight", type=float, default=0.3)
    parser.add_argument("--cross_modal_weight", type=float, default=0.3)
    parser.add_argument("--density_weight", type=float, default=0.1)
    parser.add_argument("--nli_weight", type=float, default=0.15)
    parser.add_argument("--prm_weight", type=float, default=0.15)

    # Save and load arguments
    parser.add_argument("--save_cots", type=str, default=None)
    parser.add_argument("--save_embeddings", type=str, default=None)
    parser.add_argument("--save_scores", type=str, default=None)
    parser.add_argument("--load_cots", type=str, default=None)
    parser.add_argument("--load_embeddings", type=str, default=None)

    # Pipeline control arguments
    parser.add_argument(
        "--skip_cot_generation",
        action="store_true",
        help="Skip CoT generation (requires --load_cots)"
    )
    parser.add_argument(
        "--skip_embedding_extraction",
        action="store_true",
        help="Skip embedding extraction (requires --load_embeddings)"
    )

    # General arguments
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    logger = setup_logger(args.log_file)

    logger.info("=" * 80)
    logger.info("Multimodal CoT Confidence Scoring Experiment (Modular)")
    logger.info("=" * 80)

    # Step 1: Load dataset
    logger.info("Loading dataset...")
    unobench = UNOBenchLoader(
        data_path=args.data_path,
        split=args.split,
        modality_filter=args.modality_filter
    )
    samples = list(unobench)
    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info(f"Loaded {len(samples)} samples")

    # Step 2: Generate or load CoT chains
    cots = None
    if not args.skip_cot_generation:
        cots = generate_cots(
            samples=samples,
            model_name=args.cot_model_name,
            model_type=args.cot_model_type,
            num_chains=args.num_chains,
            max_new_tokens=args.max_new_tokens,
            use_openai_api=args.use_openai_api,
            openai_api_key=args.openai_api_key,
            logger=logger,
            verbose=args.verbose,
        )
    elif args.load_cots:
        cots = load_cots(cots_path=args.load_cots, logger=logger)
    else:
        logger.error("Must either generate CoTs or provide --load_cots path")
        return

    # Save CoTs if requested
    if args.save_cots and cots:
        save_cots(cots=cots, save_path=args.save_cots, logger=logger)

    # Filter CoTs if requested
    if args.num_chains_for_scoring is not None and cots is not None:
        cots = filter_cots_by_count(
            cots=cots,
            num_chains=args.num_chains_for_scoring,
            logger=logger
        )

    # Step 3: Setup coherence metrics and scorer
    logger.info("Setting up coherence metrics and confidence scorer...")
    internal_metric, cross_modal_metric = setup_coherence_metrics(
        similarity_metric=args.similarity_metric,
        aggregation=args.aggregation,
        goal_directedness_weight=args.goal_directedness_weight,
        smoothness_weight=args.smoothness_weight,
        contrastive_margin=args.contrastive_margin,
        use_attention=args.use_attention,
    )

    confidence_scorer = setup_confidence_scorer(
        internal_metric=internal_metric,
        cross_modal_metric=cross_modal_metric,
        use_density_model=args.use_density_model,
        density_model_type=args.density_model_type,
        use_nli=args.use_nli,
        nli_model_name=args.nli_model_name,
        use_prm=args.use_prm,
        prm_model_name=args.prm_model_name,
        internal_weight=args.internal_weight,
        cross_modal_weight=args.cross_modal_weight,
        density_weight=args.density_weight,
        nli_weight=args.nli_weight,
        prm_weight=args.prm_weight,
        device=device,
        logger=logger,
    )

    # Step 4: Extract or load embeddings
    embeddings = None
    if not args.skip_embedding_extraction:
        text_encoder, multimodal_encoder, audio_encoder, omnimodal_encoder = setup_encoders(
            text_encoder=args.text_encoder,
            multimodal_encoder=args.multimodal_encoder,
            audio_encoder=args.audio_encoder,
            omnimodal_encoder=args.omnimodal_encoder,
            device=device,
            logger=logger,
        )

        embeddings = extract_embeddings_batch(
            samples=samples,
            cot_chains=cots,
            text_encoder=text_encoder,
            multimodal_encoder=multimodal_encoder,
            audio_encoder=audio_encoder,
            omnimodal_encoder=omnimodal_encoder,
            device=device,
            logger=logger,
        )
    elif args.load_embeddings:
        embeddings = load_embeddings(
            embeddings_path=args.load_embeddings,
            samples=samples,
            device=device,
            logger=logger,
        )
        if args.num_chains_for_scoring is not None:
            embeddings = [
                sample_embs[:args.num_chains_for_scoring]
                for sample_embs in embeddings
            ]
    else:
        logger.error("Must extract embeddings or provide --load_embeddings path")
        return

    # Save embeddings if requested
    if args.save_embeddings and embeddings:
        save_embeddings(
            embeddings=embeddings,
            save_path=args.save_embeddings,
            samples=samples,
            logger=logger,
        )

    # Step 5: Compute coherence scores
    scores = compute_coherence_scores(
        embeddings=embeddings,
        confidence_scorer=confidence_scorer,
        logger=logger,
    )

    # Step 6: Save scores
    if args.save_scores and scores:
        save_scores(scores=scores, save_path=args.save_scores, logger=logger)

    logger.info("=" * 80)
    logger.info("Experiment complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
