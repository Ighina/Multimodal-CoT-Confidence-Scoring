"""
Integrated experiment runner for multimodal CoT confidence scoring.
Supports both image and audio modalities with full pipeline integration.
"""

import argparse
import json
from pathlib import Path
import torch
from typing import Optional, List, Dict
import logging

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.uno_bench_loader import UNOBenchLoader
from src.dataset.cot_generator import CoTGenerator, CoTChain
from src.embeddings import TextEncoder, MultimodalEncoder, AudioEncoder
from src.coherence import (
    InternalCoherenceMetric,
    CrossModalCoherenceMetric,
    ChainConfidenceScorer,
)
from src.coherence_models import KDEDensityModel, GMMDensityModel


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger for the experiment."""
    logger = logging.getLogger("multimodal_cot_experiment")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def convert_to_dict(chain):
    """Convert a CoTChain object to a dictionary."""
    return {
        "metadata": chain.metadata,
        "final_answer": chain.final_answer,
        "text": chain.text,
        "steps": chain.steps,
        "log_probs": chain.log_probs,
    }


def setup_encoders(args, device: str, logger: logging.Logger):
    """Setup text, multimodal (image), and audio encoders."""
    logger.info("Setting up encoders...")

    # Text encoder for CoT steps

    if args.audio_encoder.lower().find("clap") != -1:
        args.text_encoder = "clap"
    else:
        text_encoder = TextEncoder(model_name=args.text_encoder, device=device)
    logger.info(f"Text encoder loaded: {args.text_encoder}")

    # Multimodal encoder for images (if needed)
    multimodal_encoder = None
    if args.multimodal_encoder:
        multimodal_encoder = MultimodalEncoder(
            model_name=args.multimodal_encoder, device=device
        )
        logger.info(f"Multimodal encoder loaded: {args.multimodal_encoder}")

    # Audio encoder (if needed)
    audio_encoder = None
    if args.audio_encoder:
        use_clap = "clap" in args.audio_encoder.lower()
        audio_encoder = AudioEncoder(
            model_name=args.audio_encoder, device=device, use_clap=use_clap
        )
        logger.info(f"Audio encoder loaded: {args.audio_encoder}")

        text_encoder = "clap"

    return text_encoder, multimodal_encoder, audio_encoder


def setup_coherence_metrics(args):
    """Setup coherence metrics."""
    internal_metric = InternalCoherenceMetric(
        similarity_metric=args.similarity_metric,
        aggregation=args.aggregation,
        goal_directedness_weight=args.goal_directedness_weight,
        smoothness_weight=args.smoothness_weight,
    )

    cross_modal_metric = CrossModalCoherenceMetric(
        similarity_metric=args.similarity_metric,
        contrastive_margin=args.contrastive_margin,
        use_attention=args.use_attention,
    )

    return internal_metric, cross_modal_metric


def setup_confidence_scorer(args, internal_metric, cross_modal_metric):
    """Setup chain confidence scorer."""
    density_model = None
    if args.use_density_model:
        if args.density_model_type == "kde":
            density_model = KDEDensityModel()
        elif args.density_model_type == "gmm":
            density_model = GMMDensityModel()

    scorer = ChainConfidenceScorer(
        internal_metric=internal_metric,
        cross_modal_metric=cross_modal_metric,
        density_model=density_model,
        internal_weight=args.internal_weight,
        cross_modal_weight=args.cross_modal_weight,
        density_weight=args.density_weight,
    )

    return scorer


def extract_embeddings(
    samples: List,
    cot_chains: List,
    text_encoder: TextEncoder,
    multimodal_encoder: Optional[MultimodalEncoder],
    audio_encoder: Optional[AudioEncoder],
    device: str,
    logger: logging.Logger,
) -> List[List[Dict[str, torch.Tensor]]]:
    """
    Extract embeddings for all samples and chains.
    Supports both image and audio modalities.
    """
    logger.info("Extracting embeddings...")
    all_embeddings = []

    for idx, (sample, chains) in enumerate(zip(samples, cot_chains)):
        if (idx + 1) % 10 == 0:
            logger.info(f"Processing sample {idx + 1}/{len(samples)}")

        sample_embeddings = []

        for chain in chains:
            # Extract step embeddings
            if text_encoder != "clap":
                step_embeddings = text_encoder.encode_cot_steps(
                    chain.steps, question=sample.question
                )

            # Extract modality-specific embeddings
            modal_embeddings = None

            # Handle images
            if sample.images and multimodal_encoder:
                modal_embeddings = multimodal_encoder.encode_images(sample.images)

            # Handle audio (takes precedence if both exist)
            if sample.audio_paths and audio_encoder:
                try:
                    if text_encoder == "clap":
                        step_embeddings = audio_encoder.encode_text_for_audio_alignment(
                            chain.steps
                        )
                    modal_embeddings = audio_encoder.encode_audio_from_file(
                        sample.audio_paths
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to encode audio for sample {sample.id}: {e}"
                    )
                    # Fall back to zero embeddings
                    if modal_embeddings is None:
                        modal_embeddings = torch.zeros(
                            audio_encoder.get_embedding_dim()
                            if audio_encoder
                            else multimodal_encoder.get_embedding_dim()
                        ).to(device)

            # If no modality data, use zero embeddings
            if modal_embeddings is None:
                embedding_dim = (
                    multimodal_encoder.get_embedding_dim()
                    if multimodal_encoder
                    else (
                        audio_encoder.get_embedding_dim() if audio_encoder else 512
                    )  # Default fallback
                )
                modal_embeddings = torch.zeros(embedding_dim).to(device)

            # Extract question and answer embeddings
            if text_encoder == "clap":
                question_embedding = audio_encoder.encode_text_for_audio_alignment(
                    [sample.question]
                )
                answer_embedding = audio_encoder.encode_text_for_audio_alignment(
                    [chain.final_answer]
                )
            else:
                question_embedding = text_encoder(sample.question)
                answer_embedding = text_encoder(chain.final_answer)

            chain_embedding_data = {
                "step_embeddings": step_embeddings.cpu(),
                "modal_embeddings": modal_embeddings.cpu(),  # Generic name for image/audio
                "question_embedding": question_embedding.cpu(),
                "answer_embedding": answer_embedding.cpu(),
                "modality_type": sample.modality,  # Track which modality this is
            }

            sample_embeddings.append(chain_embedding_data)

        all_embeddings.append(sample_embeddings)

    logger.info("Embedding extraction complete")
    return all_embeddings


def compute_coherence_scores(
    embeddings: List[List[Dict[str, torch.Tensor]]],
    confidence_scorer: ChainConfidenceScorer,
    logger: logging.Logger,
) -> List[List[Dict[str, float]]]:
    """Compute coherence scores for all samples and chains."""
    logger.info("Computing coherence scores...")
    all_scores = []

    for idx, sample_embeddings in enumerate(embeddings):
        if (idx + 1) % 10 == 0:
            logger.info(f"Scoring sample {idx + 1}/{len(embeddings)}")

        sample_scores = []

        for chain_emb in sample_embeddings:
            # Compute scores using the confidence scorer
            scores = confidence_scorer(
                step_embeddings=chain_emb["step_embeddings"],
                modal_embeddings=chain_emb["modal_embeddings"],
                question_embedding=chain_emb["question_embedding"],
                answer_embedding=chain_emb["answer_embedding"],
            )

            # Convert tensors to floats for JSON serialization
            def convert_scores(obj):
                """Recursively convert tensors to floats."""
                if isinstance(obj, torch.Tensor):
                    if obj.numel() == 1:
                        return obj.item()
                    else:
                        return obj.cpu().numpy().tolist()
                elif isinstance(obj, dict):
                    return {k: convert_scores(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_scores(item) for item in obj]
                else:
                    return obj

            scores_dict = convert_scores(scores)
            sample_scores.append(scores_dict)

        all_scores.append(sample_scores)

    logger.info("Coherence scoring complete")
    return all_scores


def process_sample_sequential(
    sample,
    sample_idx: int,
    cot_chains: List,
    text_encoder,
    multimodal_encoder: Optional[MultimodalEncoder],
    audio_encoder: Optional[AudioEncoder],
    confidence_scorer: ChainConfidenceScorer,
    device: str,
    save_embeddings_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[List[Dict[str, torch.Tensor]], List[Dict[str, float]]]:
    """
    Process a single sample: extract embeddings and compute coherence scores.
    Optionally saves embeddings to a file and returns scores.

    This function processes one sample at a time to minimize memory usage.
    """
    if logger and (sample_idx + 1) % 10 == 0:
        logger.info(f"Processing sample {sample_idx + 1}")

    # Extract embeddings for this sample
    sample_embeddings = []

    for chain in cot_chains:
        # Extract step embeddings
        if text_encoder != "clap":
            step_embeddings = text_encoder.encode_cot_steps(
                chain.steps, question=sample.question
            )

        # Extract modality-specific embeddings
        modal_embeddings = None

        # Handle images
        if sample.images and multimodal_encoder:
            modal_embeddings = multimodal_encoder.encode_images(sample.images)

        # Handle audio (takes precedence if both exist)
        if sample.audio_paths and audio_encoder:
            try:
                if text_encoder == "clap":
                    step_embeddings = audio_encoder.encode_text_for_audio_alignment(
                        chain.steps
                    )
                modal_embeddings = audio_encoder.encode_audio_from_file(
                    sample.audio_paths
                )
            except Exception as e:
                if logger:
                    logger.warning(
                        f"Failed to encode audio for sample {sample.id}: {e}"
                    )
                # Fall back to zero embeddings
                if modal_embeddings is None:
                    modal_embeddings = torch.zeros(
                        audio_encoder.get_embedding_dim()
                        if audio_encoder
                        else multimodal_encoder.get_embedding_dim()
                    ).to("cpu")

        # If no modality data, use zero embeddings
        if modal_embeddings is None:
            embedding_dim = (
                multimodal_encoder.get_embedding_dim()
                if multimodal_encoder
                else (
                    audio_encoder.get_embedding_dim() if audio_encoder else 512
                )  # Default fallback
            )
            modal_embeddings = torch.zeros(embedding_dim).to("cpu")

        # Extract question and answer embeddings
        if text_encoder == "clap":
            question_embedding = audio_encoder.encode_text_for_audio_alignment(
                [sample.question]
            )
            answer_embedding = audio_encoder.encode_text_for_audio_alignment(
                [chain.final_answer]
            )
        else:
            question_embedding = text_encoder(sample.question)
            answer_embedding = text_encoder(chain.final_answer)

        try:
            step_embeddings = step_embeddings.cpu()
        except:
            step_embeddings = torch.zeros(
                (len(chain.steps), text_encoder.get_embedding_dim())
            ).to("cpu")

        chain_embedding_data = {
            "step_embeddings": step_embeddings,
            "modal_embeddings": modal_embeddings.cpu(),
            "question_embedding": question_embedding.cpu(),
            "answer_embedding": answer_embedding.cpu(),
            "modality_type": sample.modality,
        }

        sample_embeddings.append(chain_embedding_data)

    # Save embeddings if directory is provided
    if save_embeddings_dir:
        save_embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_file = save_embeddings_dir / f"{sample.id}.json"

        # Convert tensors to lists for JSON serialization
        embeddings_serializable = []
        for chain_emb in sample_embeddings:
            chain_dict = {
                k: (v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in chain_emb.items()
            }
            embeddings_serializable.append(chain_dict)

        with open(embedding_file, "w") as f:
            json.dump(embeddings_serializable, f, indent=2)

        if logger:
            logger.debug(f"Saved embeddings to {embedding_file}")

    # Compute coherence scores for this sample
    sample_scores = []

    for chain_emb in sample_embeddings:
        # Compute scores using the confidence scorer
        scores = confidence_scorer(
            step_embeddings=chain_emb["step_embeddings"],
            modal_embeddings=chain_emb["modal_embeddings"],
            question_embedding=chain_emb["question_embedding"],
            answer_embedding=chain_emb["answer_embedding"],
        )

        # Convert tensors to floats for JSON serialization
        def convert_scores(obj):
            """Recursively convert tensors to floats."""
            if isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                else:
                    return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_scores(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_scores(item) for item in obj]
            else:
                return obj

        scores_dict = convert_scores(scores)
        sample_scores.append(scores_dict)

    # Return embeddings and scores (embeddings can be discarded by caller to save memory)
    return sample_embeddings, sample_scores


def load_embeddings_from_path(
    embeddings_path: str,
    samples: List,
    device: str,
    logger: Optional[logging.Logger] = None,
) -> List[List[Dict[str, torch.Tensor]]]:
    """
    Load embeddings from either a directory of individual files or a single file.

    Args:
        embeddings_path: Path to either a directory containing individual embedding files
                        or a single JSON file with all embeddings
        samples: List of samples (needed to get sample IDs for directory loading)
        device: Device to load tensors to
        logger: Optional logger

    Returns:
        List of embeddings per sample
    """
    path = Path(embeddings_path)

    if path.is_dir():
        # Load individual files from directory
        if logger:
            logger.info(f"Loading embeddings from directory {embeddings_path}...")

        embeddings = []
        for sample in samples:
            embedding_file = path / f"{sample.id}.json"

            if not embedding_file.exists():
                if logger:
                    logger.warning(
                        f"Embedding file not found for sample {sample.id}, skipping"
                    )
                continue

            with open(embedding_file, "r") as f:
                sample_embeddings_data = json.load(f)

            # Convert loaded data back to tensors
            sample_list = []
            for chain_emb in sample_embeddings_data:
                chain_dict = {
                    k: (torch.tensor(v).to(device) if isinstance(v, list) else v)
                    for k, v in chain_emb.items()
                }
                sample_list.append(chain_dict)

            embeddings.append(sample_list)

        if logger:
            logger.info(f"Loaded embeddings for {len(embeddings)} samples")

    else:
        # Load from single file (legacy behavior)
        if logger:
            logger.info(f"Loading embeddings from file {embeddings_path}...")

        with open(embeddings_path, "r") as f:
            embeddings_data = json.load(f)

        # Convert loaded data back to tensors
        embeddings = []
        for sample_embs in embeddings_data:
            sample_list = []
            for chain_emb in sample_embs:
                chain_dict = {
                    k: (torch.tensor(v).to(device) if isinstance(v, list) else v)
                    for k, v in chain_emb.items()
                }
                sample_list.append(chain_dict)
            embeddings.append(sample_list)

        if logger:
            logger.info(f"Loaded embeddings for {len(embeddings)} samples")

    return embeddings


def save_results(
    cots: Optional[List] = None,
    embeddings: Optional[List] = None,
    scores: Optional[List] = None,
    save_cots: Optional[str] = None,
    save_embeddings: Optional[str] = None,
    save_scores: Optional[str] = None,
    samples: Optional[List] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Save results to specified paths.

    For embeddings:
    - If save_embeddings is a directory path, saves individual files per sample
    - If save_embeddings is a file path, saves all embeddings in one file (legacy behavior)
    """
    if cots and save_cots:
        with open(save_cots, "w") as f:
            json.dump(
                [[convert_to_dict(chain) for chain in chains] for chains in cots],
                f,
                indent=2,
            )
        if logger:
            logger.info(f"Saved CoTs to {save_cots}")

    if embeddings and save_embeddings:
        embeddings_path = Path(save_embeddings)

        # Determine if this should be saved as directory or single file
        # Check if path ends with .json or has an extension (treat as file)
        # Otherwise treat as directory
        save_as_directory = embeddings_path.suffix == "" or embeddings_path.is_dir()

        if save_as_directory:
            # Save individual files in directory
            if not samples:
                if logger:
                    logger.error(
                        "Cannot save embeddings to directory without sample IDs"
                    )
                return

            embeddings_path.mkdir(parents=True, exist_ok=True)

            for sample, sample_embs in zip(samples, embeddings):
                embedding_file = embeddings_path / f"{sample.id}.json"

                # Convert tensors to lists for JSON serialization
                embeddings_serializable = []
                for chain_emb in sample_embs:
                    chain_dict = {
                        k: (
                            v.detach().cpu().numpy().tolist()
                            if isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in chain_emb.items()
                    }
                    embeddings_serializable.append(chain_dict)

                with open(embedding_file, "w") as f:
                    json.dump(embeddings_serializable, f, indent=2)

            if logger:
                logger.info(
                    f"Saved {len(embeddings)} embedding files to directory {save_embeddings}"
                )
        else:
            # Save as single file (legacy behavior)
            embeddings_serializable = []
            for sample_embs in embeddings:
                sample_list = []
                for chain_emb in sample_embs:
                    chain_dict = {
                        k: (
                            v.detach().cpu().numpy().tolist()
                            if isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in chain_emb.items()
                    }
                    sample_list.append(chain_dict)
                embeddings_serializable.append(sample_list)

            with open(save_embeddings, "w") as f:
                json.dump(embeddings_serializable, f, indent=2)
            if logger:
                logger.info(f"Saved embeddings to {save_embeddings}")

    if scores and save_scores:
        with open(save_scores, "w") as f:
            json.dump(scores, f, indent=2)
        if logger:
            logger.info(f"Saved scores to {save_scores}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multimodal CoT confidence scoring experiments with support for images and audio"
    )

    # Dataset arguments
    parser.add_argument(
        "--data_path", type=str, default="uno-bench", help="Path to UNO-Bench dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--modality_filter",
        type=str,
        default=None,
        help='Filter by modality (e.g., "UNOBench-Audio", "UNOBench-Image", None for all)',
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)",
    )

    # CoT generation arguments
    parser.add_argument(
        "--cot_model_name",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Model name for CoT generation",
    )
    parser.add_argument(
        "--cot_model_type",
        type=str,
        default="qwen2_audio",
        help="Model type for CoT generation",
    )
    parser.add_argument(
        "--num_chains",
        type=int,
        default=20,
        help="Number of CoT chains to generate per sample",
    )
    parser.add_argument(
        "--num_chains_for_scoring",
        type=int,
        default=None,
        help="Number of chains to use for scoring (subset of generated/loaded chains). "
        "If None, uses all chains. Useful for iterating experiments with different chain counts.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens for CoT generation",
    )

    # Encoder arguments
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Text encoder model for CoT steps",
    )
    parser.add_argument(
        "--multimodal_encoder",
        type=str,
        default=None,
        help='Multimodal encoder for images (e.g., "openai/clip-vit-large-patch14")',
    )
    parser.add_argument(
        "--audio_encoder",
        type=str,
        default=None,
        help='Audio encoder model (e.g., "facebook/wav2vec2-base-960h" or CLAP model)',
    )

    # Coherence metric arguments
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "dot"],
        help="Similarity metric for coherence computation",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "min", "max"],
        help="Aggregation method for internal coherence",
    )
    parser.add_argument(
        "--goal_directedness_weight",
        type=float,
        default=0.6,
        help="Weight for goal directedness in internal coherence",
    )
    parser.add_argument(
        "--smoothness_weight",
        type=float,
        default=0.4,
        help="Weight for step smoothness in internal coherence",
    )
    parser.add_argument(
        "--contrastive_margin",
        type=float,
        default=0.2,
        help="Margin for contrastive loss in cross-modal coherence",
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
        help="Use attention weighting in cross-modal coherence",
    )

    # Confidence scoring arguments
    parser.add_argument(
        "--use_density_model",
        action="store_true",
        help="Use density model for confidence scoring",
    )
    parser.add_argument(
        "--density_model_type",
        type=str,
        default="kde",
        choices=["kde", "gmm"],
        help="Type of density model to use",
    )
    parser.add_argument(
        "--internal_weight",
        type=float,
        default=0.4,
        help="Weight for internal coherence in confidence score",
    )
    parser.add_argument(
        "--cross_modal_weight",
        type=float,
        default=0.4,
        help="Weight for cross-modal coherence in confidence score",
    )
    parser.add_argument(
        "--density_weight",
        type=float,
        default=0.2,
        help="Weight for density score in confidence score",
    )

    # Save arguments
    parser.add_argument(
        "--save_cots",
        type=str,
        default=None,
        help="Path to save generated CoTs (JSON format)",
    )
    parser.add_argument(
        "--save_embeddings",
        type=str,
        default=None,
        help="Path to save embeddings (JSON format)",
    )
    parser.add_argument(
        "--save_scores",
        type=str,
        default=None,
        help="Path to save coherence scores (JSON format)",
    )
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file")

    # General arguments
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip_cot_generation",
        action="store_true",
        help="Skip CoT generation (load from file instead)",
    )
    parser.add_argument(
        "--load_cots",
        type=str,
        default=None,
        help="Path to load pre-generated CoTs from",
    )
    parser.add_argument(
        "--skip_embedding_extraction",
        action="store_true",
        help="Skip embedding extraction (load from file instead)",
    )
    parser.add_argument(
        "--load_embeddings",
        type=str,
        default=None,
        help="Path to load pre-computed embeddings from",
    )
    parser.add_argument(
        "--sequential_processing",
        action="store_true",
        help="Process samples sequentially (extract embeddings and score one at a time) to reduce memory usage. "
        "When enabled, embeddings are saved/loaded as individual files in a directory.",
    )

    args = parser.parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    logger = setup_logger(args.log_file)

    logger.info("=" * 80)
    logger.info("Multimodal CoT Confidence Scoring Experiment")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.data_path}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Modality filter: {args.modality_filter}")

    # Load dataset
    logger.info("Loading UNO-Bench dataset...")
    unobench = UNOBenchLoader(
        data_path=args.data_path, split=args.split, modality_filter=args.modality_filter
    )

    # Limit samples if specified
    samples = list(unobench)
    if args.max_samples:
        samples = samples[: args.max_samples]
    logger.info(f"Loaded {len(samples)} samples")

    # Log dataset statistics
    stats = unobench.get_statistics()
    logger.info(f"Dataset statistics: {stats}")

    # Generate or load CoT chains
    cots = None
    if not args.skip_cot_generation:
        logger.info("Generating CoT chains...")
        generator = CoTGenerator(
            model_name=args.cot_model_name, model_type=args.cot_model_type
        )

        cots = []
        for idx, sample in enumerate(samples):
            logger.info(f"Generating CoTs for sample {idx + 1}/{len(samples)}")
            try:
                chains = generator.generate_cot_from_sample(
                    sample,
                    max_new_tokens=args.max_new_tokens,
                    num_chains=args.num_chains,
                )
            except:
                continue
            cots.append(chains)
    elif args.load_cots:
        logger.info(f"Loading CoTs from {args.load_cots}...")
        with open(args.load_cots, "r") as f:
            cots = json.load(f)
        # Note: You may need to reconstruct CoTChain objects from loaded data
        logger.info(f"Loaded {len(cots)} sample CoTs")
        cots = [[CoTChain(**chain) for chain in cot] for cot in cots]
    else:
        logger.error("Must either generate CoTs or provide --load_cots path")
        return

    if args.save_cots:
        save_results(
            cots=cots,
            embeddings=None,
            scores=None,
            save_cots=args.save_cots,
            save_embeddings=None,
            save_scores=None,
            samples=samples,
            logger=logger,
        )

    # Limit number of chains for scoring if specified
    if args.num_chains_for_scoring is not None and cots is not None:
        logger.info(
            f"Limiting to {args.num_chains_for_scoring} chains per sample for scoring"
        )
        cots = [chains[: args.num_chains_for_scoring] for chains in cots]
        logger.info(f"Using {len(cots[0])} chains per sample")

    # Setup coherence metrics and confidence scorer
    logger.info("Setting up coherence metrics and confidence scorer...")
    internal_metric, cross_modal_metric = setup_coherence_metrics(args)
    confidence_scorer = setup_confidence_scorer(
        args, internal_metric, cross_modal_metric
    )

    # Process samples sequentially or in batch
    embeddings = None
    scores = None

    if args.sequential_processing:
        # Sequential processing: extract embeddings and score one sample at a time
        # This mode minimizes memory usage by not keeping all embeddings in memory
        logger.info("Using sequential processing mode (memory efficient)")

        if not args.skip_embedding_extraction:
            # Setup encoders
            text_encoder, multimodal_encoder, audio_encoder = setup_encoders(
                args, device, logger
            )

            # Process each sample sequentially
            scores = []
            save_embeddings_dir = (
                Path(args.save_embeddings) if args.save_embeddings else None
            )

            for idx, (sample, sample_chains) in enumerate(zip(samples, cots)):
                _, sample_scores = process_sample_sequential(
                    sample=sample,
                    sample_idx=idx,
                    cot_chains=sample_chains,
                    text_encoder=text_encoder,
                    multimodal_encoder=multimodal_encoder,
                    audio_encoder=audio_encoder,
                    confidence_scorer=confidence_scorer,
                    device=device,
                    save_embeddings_dir=save_embeddings_dir,
                    logger=logger,
                )
                scores.append(sample_scores)

            logger.info("Sequential processing complete")

        elif args.load_embeddings:
            # Load embeddings sequentially and score them
            embeddings_path = Path(args.load_embeddings)

            if not embeddings_path.is_dir():
                logger.error(
                    "Sequential processing requires embeddings to be stored in a directory. "
                    f"Provided path {args.load_embeddings} is not a directory."
                )
                return

            scores = []
            for idx, sample in enumerate(samples):
                embedding_file = embeddings_path / f"{sample.id}.json"

                if not embedding_file.exists():
                    logger.warning(
                        f"Embedding file not found for sample {sample.id}, skipping"
                    )
                    continue

                with open(embedding_file, "r") as f:
                    sample_embeddings_data = json.load(f)

                # Convert loaded data back to tensors
                sample_embeddings = []
                for chain_emb in sample_embeddings_data:
                    chain_dict = {
                        k: (torch.tensor(v).to(device) if isinstance(v, list) else v)
                        for k, v in chain_emb.items()
                    }
                    sample_embeddings.append(chain_dict)

                # Limit chains if specified
                if args.num_chains_for_scoring is not None:
                    sample_embeddings = sample_embeddings[: args.num_chains_for_scoring]

                # Compute coherence scores for this sample
                sample_scores = []
                for chain_emb in sample_embeddings:
                    scores_result = confidence_scorer(
                        step_embeddings=chain_emb["step_embeddings"],
                        modal_embeddings=chain_emb["modal_embeddings"],
                        question_embedding=chain_emb["question_embedding"],
                        answer_embedding=chain_emb["answer_embedding"],
                    )

                    # Convert tensors to floats for JSON serialization
                    def convert_scores(obj):
                        if isinstance(obj, torch.Tensor):
                            if obj.numel() == 1:
                                return obj.item()
                            else:
                                return obj.cpu().numpy().tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_scores(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_scores(item) for item in obj]
                        else:
                            return obj

                    scores_dict = convert_scores(scores_result)
                    sample_scores.append(scores_dict)

                scores.append(sample_scores)

                if (idx + 1) % 10 == 0:
                    logger.info(f"Scored sample {idx + 1}/{len(samples)}")

            logger.info("Sequential scoring complete")

        else:
            logger.error(
                "Must either extract embeddings or provide --load_embeddings path"
            )
            return

    else:
        # Batch processing: extract all embeddings first, then score
        logger.info("Using batch processing mode")

        if not args.skip_embedding_extraction:
            # Setup encoders
            text_encoder, multimodal_encoder, audio_encoder = setup_encoders(
                args, device, logger
            )

            # Extract embeddings
            embeddings = extract_embeddings(
                samples=samples,
                cot_chains=cots,
                text_encoder=text_encoder,
                multimodal_encoder=multimodal_encoder,
                audio_encoder=audio_encoder,
                device=device,
                logger=logger,
            )
        elif args.load_embeddings:
            # Load embeddings from file or directory
            embeddings = load_embeddings_from_path(
                embeddings_path=args.load_embeddings,
                samples=samples,
                device=device,
                logger=logger,
            )

            # Limit embeddings if num_chains_for_scoring is specified
            if args.num_chains_for_scoring is not None:
                logger.info(
                    f"Limiting embeddings to {args.num_chains_for_scoring} chains per sample"
                )
                embeddings = [
                    sample_embs[: args.num_chains_for_scoring]
                    for sample_embs in embeddings
                ]
        else:
            logger.error(
                "Must either extract embeddings or provide --load_embeddings path"
            )
            return

        # Compute coherence scores
        scores = compute_coherence_scores(
            embeddings=embeddings, confidence_scorer=confidence_scorer, logger=logger
        )

    # Save results
    logger.info("Saving results...")
    save_results(
        cots=cots,
        embeddings=embeddings if args.save_embeddings else None,
        scores=scores,
        save_cots=None,
        save_embeddings=args.save_embeddings,
        save_scores=args.save_scores,
        samples=samples,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("Experiment complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
