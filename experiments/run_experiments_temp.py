"""
Integrated experiment runner for multimodal CoT confidence scoring.
Supports image, audio, and text-based logical reasoning (NLI/PRM) pipelines.
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
from src.dataset.cot_generator import CoTGenerator, CoTChain, OpenAICoTGenerator
from src.embeddings import (
    TextEncoder,
    MultimodalEncoder,
    AudioEncoder,
    OmnimodalEncoder,
)
from src.coherence import (
    InternalCoherenceMetric,
    CrossModalCoherenceMetric,
    ChainConfidenceScorer,
    NLICoherenceMetric,  # <-- Added
    PRMCoherenceMetric,  # <-- Added
)
from src.coherence_models import KDEDensityModel, GMMDensityModel

UNO_BENCH_AUDIO_URL = (
    "https://huggingface.co/datasets/meituan-longcat/UNO-Bench/blob/main/audios/"
)
UNO_BENCH_VIDEO_URL = (
    "https://huggingface.co/datasets/meituan-longcat/UNO-Bench/tree/main/videos/"
)


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger for the experiment."""
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

    if args.omnimodal_encoder:
        omnimodal_encoder = OmnimodalEncoder(
            model_name=args.omnimodal_encoder, device=device
        )
        text_encoder = None
        multimodal_encoder = None
        audio_encoder = None
        logger.info(f"Omnimodal encoder loaded: {args.omnimodal_encoder}")
    elif (
        args.text_encoder == args.multimodal_encoder == args.audio_encoder
        and args.text_encoder is not None
    ):
        # this is the case in which we use the omnimodal encoder, but we encode modalities separately
        omnimodal_encoder = None
        text_encoder = OmnimodalEncoder(model_name=args.text_encoder, device=device)
        multimodal_encoder = text_encoder
        audio_encoder = text_encoder
        logger.info(f"Omnimodal encoder loaded for all modalities: {args.text_encoder}")
    else:
        omnimodal_encoder = None
        if args.audio_encoder and "clap" in args.audio_encoder.lower():
            args.text_encoder = "clap"
        else:
            text_encoder = TextEncoder(model_name=args.text_encoder, device=device)
        logger.info(f"Text encoder loaded: {args.text_encoder}")

        multimodal_encoder = None
        if args.multimodal_encoder:
            multimodal_encoder = MultimodalEncoder(
                model_name=args.multimodal_encoder, device=device
            )
            logger.info(f"Multimodal encoder loaded: {args.multimodal_encoder}")

        audio_encoder = None
        if args.audio_encoder:
            use_clap = "clap" in args.audio_encoder.lower()
            audio_encoder = AudioEncoder(
                model_name=args.audio_encoder, device=device, use_clap=use_clap
            )
            logger.info(f"Audio encoder loaded: {args.audio_encoder}")
            text_encoder = "clap"

    return text_encoder, multimodal_encoder, audio_encoder, omnimodal_encoder


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


def setup_confidence_scorer(
    args, internal_metric, cross_modal_metric, device: str, logger: logging.Logger
):
    """Setup chain confidence scorer including optional Text NLI/PRM."""
    density_model = None
    if args.use_density_model:
        if args.density_model_type == "kde":
            density_model = KDEDensityModel()
        elif args.density_model_type == "gmm":
            density_model = GMMDensityModel()

    # Setup NLI Metric
    nli_metric = None
    if args.use_nli:
        logger.info(f"Loading NLI Metric: {args.nli_model_name}")
        nli_device = 0 if device == "cuda" else -1
        nli_metric = NLICoherenceMetric(
            model_name=args.nli_model_name, device=nli_device
        )

    # Setup PRM Metric
    prm_metric = None
    if args.use_prm:
        logger.info(f"Loading PRM Metric: {args.prm_model_name}")
        prm_metric = PRMCoherenceMetric(model_name=args.prm_model_name, device=device)

    scorer = ChainConfidenceScorer(
        internal_metric=internal_metric,
        cross_modal_metric=cross_modal_metric,
        density_model=density_model,
        nli_metric=nli_metric,
        prm_metric=prm_metric,
        internal_weight=args.internal_weight,
        cross_modal_weight=args.cross_modal_weight,
        density_weight=args.density_weight,
        nli_weight=args.nli_weight,
        prm_weight=args.prm_weight,
    )

    return scorer


def extract_embeddings(
    samples: List,
    cot_chains: List,
    text_encoder: Optional[TextEncoder],
    multimodal_encoder: Optional[MultimodalEncoder],
    audio_encoder: Optional[AudioEncoder],
    omnimodal_encoder: Optional[OmnimodalEncoder],
    device: str,
    logger: logging.Logger,
    encode_from_url: bool = True,
) -> List[List[Dict[str, torch.Tensor]]]:
    """
    Extract embeddings for all samples and chains.
    Now includes string metadata needed for NLI and PRM.
    """
    assert not (
        omnimodal_encoder and (text_encoder or multimodal_encoder or audio_encoder)
    ), "Cannot use omnimodal encoder together with separate text/multimodal/audio encoders"
    assert (omnimodal_encoder is not None) or (
        text_encoder is not None
    ), "At least one encoder must be provided"

    logger.info("Extracting embeddings...")
    all_embeddings = []

    for idx, (sample, chains) in enumerate(zip(samples, cot_chains)):
        if (idx + 1) % 10 == 0:
            logger.info(f"Processing sample {idx + 1}/{len(samples)}")

        sample_embeddings = []

        for chain in chains:
            if text_encoder != "clap" and text_encoder is not None:
                step_embeddings = text_encoder.encode_cot_steps(
                    chain.steps, question=sample.question
                )
                question_embedding = text_encoder(sample.question)
                answer_embedding = text_encoder(chain.final_answer)
            elif text_encoder == "clap" and audio_encoder is not None:
                step_embeddings = audio_encoder.encode_text_for_audio_alignment(
                    chain.steps
                )
                question_embedding = audio_encoder.encode_text_for_audio_alignment(
                    [sample.question]
                )
                answer_embedding = audio_encoder.encode_text_for_audio_alignment(
                    [chain.final_answer]
                )

            modal_embeddings = {"image": None, "audio": None, "video": None}
            existing_modalities = []

            if sample.images and multimodal_encoder:
                modal_embeddings["image"] = multimodal_encoder.encode_images(
                    sample.images
                )
                existing_modalities.append("image")

            if sample.audio_paths and audio_encoder:
                if text_encoder == "clap":
                    step_embeddings = audio_encoder.encode_text_for_audio_alignment(
                        chain.steps
                    )
                if encode_from_url:
                    audio_paths = (
                        [
                            UNO_BENCH_AUDIO_URL + Path(path).name
                            for path in sample.audio_paths
                        ]
                        if sample.audio_paths
                        else []
                    )
                    modal_embeddings["audio"] = audio_encoder.encode_audio(audio_paths)
                else:
                    modal_embeddings["audio"] = audio_encoder.encode_audio_from_file(
                        sample.audio_paths
                    )
                existing_modalities.append("audio")

            if sample.video_paths and multimodal_encoder:
                if encode_from_url:
                    video_paths = (
                        [
                            UNO_BENCH_VIDEO_URL + Path(path).name
                            for path in sample.video_paths
                        ]
                        if sample.video_paths
                        else []
                    )
                    video_embeddings = multimodal_encoder.encode_videos(video_paths)
                else:
                    video_embeddings = multimodal_encoder.encode_videos_from_file(
                        sample.video_paths
                    )
                modal_embeddings["video"] = video_embeddings
                existing_modalities.append("video")

            if omnimodal_encoder:
                step_embeddings = omnimodal_encoder.encode_text(chain.steps)
                question_embedding = omnimodal_encoder.encode_text(sample.question)
                answer_embedding = omnimodal_encoder.encode_text(chain.final_answer)

                if encode_from_url:
                    audio_paths = (
                        [
                            UNO_BENCH_AUDIO_URL + Path(path).name
                            for path in sample.audio_paths
                        ]
                        if sample.audio_paths
                        else []
                    )
                    video_paths = (
                        [
                            UNO_BENCH_VIDEO_URL + Path(path).name
                            for path in sample.video_paths
                        ]
                        if sample.video_paths
                        else []
                    )
                else:
                    audio_paths = sample.audio_paths
                    video_paths = sample.video_paths

                modal_embeddings["omnimodal"] = omnimodal_encoder.encode(
                    text=sample.question,
                    images=sample.images,
                    audio_paths=audio_paths,
                    video_paths=video_paths,
                )

                existing_modalities.append("omnimodal")

            # TODO: handle cases in which multimodal encoding fails
            # if modal_embeddings is None:
            #     embedding_dim = (
            #         multimodal_encoder.get_embedding_dim()
            #         if multimodal_encoder
            #         else (audio_encoder.get_embedding_dim() if audio_encoder else 512)
            #     )
            #     modal_embeddings = torch.zeros(embedding_dim).to("cpu")

            if len(existing_modalities) == 1:
                modal_embeddings = modal_embeddings[existing_modalities[0]].cpu()
            else:
                for modality in existing_modalities:
                    if modal_embeddings[modality] is not None:
                        modal_embeddings[modality] = modal_embeddings[modality].cpu()

            chain_embedding_data = {
                "step_embeddings": step_embeddings.cpu(),
                "modal_embeddings": modal_embeddings,
                "question_embedding": question_embedding.cpu(),
                "answer_embedding": answer_embedding.cpu(),
                "modality_type": sample.modality,
                # --- NEW TEXT METADATA FOR NLI/PRM ---
                "text_steps": chain.steps,
                "text_query": sample.question,
                "text_final_answer": chain.final_answer,
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
            # Pass both tensors and text metadata
            scores = confidence_scorer(
                step_embeddings=chain_emb["step_embeddings"],
                modal_embeddings=chain_emb["modal_embeddings"],
                question_embedding=chain_emb["question_embedding"],
                answer_embedding=chain_emb["answer_embedding"],
                text_steps=chain_emb.get("text_steps"),
                text_query=chain_emb.get("text_query"),
                text_final_answer=chain_emb.get("text_final_answer"),
            )

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

            scores_dict = convert_scores(scores)
            sample_scores.append(scores_dict)

        all_scores.append(sample_scores)

    logger.info("Coherence scoring complete")
    return all_scores


def process_sample_sequential(
    sample,
    sample_idx: int,
    cot_chains: List,
    text_encoder: Optional[TextEncoder],
    multimodal_encoder: Optional[MultimodalEncoder],
    audio_encoder: Optional[AudioEncoder],
    omnimodal_encoder: Optional[OmnimodalEncoder],
    confidence_scorer: ChainConfidenceScorer,
    device: str,
    save_embeddings_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    encode_from_url: bool = True,
    modal_embeddings: Optional[Path] = None
) -> tuple[List[Dict[str, torch.Tensor]], List[Dict[str, float]]]:

    assert not (
        omnimodal_encoder and (text_encoder or multimodal_encoder or audio_encoder)
    ), "Cannot use omnimodal encoder together with separate text/multimodal/audio encoders"
    assert (omnimodal_encoder is not None) or (
        text_encoder is not None
    ), "At least one encoder must be provided"

    if logger and (sample_idx + 1) % 10 == 0:
        logger.info(f"Processing sample {sample_idx + 1}")

    sample_embeddings = []

    for chain in cot_chains:
        
        if not modal_embeddings:
          if text_encoder != "clap" and text_encoder is not None:
            step_embeddings = text_encoder.encode_cot_steps(
                chain.steps, question=sample.question
            )
            question_embedding = text_encoder(sample.question)
            answer_embedding = text_encoder(chain.final_answer)
          elif text_encoder == "clap" and audio_encoder is not None:
            step_embeddings = audio_encoder.encode_text_for_audio_alignment(chain.steps)
            question_embedding = audio_encoder.encode_text_for_audio_alignment(
                [sample.question]
            )
            answer_embedding = audio_encoder.encode_text_for_audio_alignment(
                [chain.final_answer]
            )
          
          modal_embeddings = {"image": None, "audio": None, "video": None}
          existing_modalities = []

          if sample.images and multimodal_encoder:
            modal_embeddings["image"] = multimodal_encoder.encode_images(sample.images)
            existing_modalities.append("image")

          if sample.audio_paths and audio_encoder:
            if text_encoder == "clap":
                step_embeddings = audio_encoder.encode_text_for_audio_alignment(
                    chain.steps
                )
            if encode_from_url:
                audio_paths = (
                    [
                        UNO_BENCH_AUDIO_URL + Path(path).name
                        for path in sample.audio_paths
                    ]
                    if sample.audio_paths
                    else []
                )
                modal_embeddings["audio"] = audio_encoder.encode_audio(audio_paths)
            else:
                modal_embeddings["audio"] = audio_encoder.encode_audio_from_file(
                    sample.audio_paths
                )
            existing_modalities.append("audio")

          if sample.video_paths and multimodal_encoder:
            if encode_from_url:
                video_paths = (
                    [
                        UNO_BENCH_VIDEO_URL + Path(path).name
                        for path in sample.video_paths
                    ]
                    if sample.video_paths
                    else []
                )
                video_embeddings = multimodal_encoder.encode_videos(video_paths)
            else:
                video_embeddings = multimodal_encoder.encode_videos_from_file(
                    sample.video_paths
                )
            modal_embeddings["video"] = video_embeddings
            existing_modalities.append("video")

          if omnimodal_encoder:
            question_embedding, answer_embedding, step_embeddings = omnimodal_encoder.encode_text(sample.question, chain.steps)
            question_embedding = question_embedding.to("cpu")
            answer_embedding = answer_embedding.to("cpu")
            step_embeddings = step_embeddings.to("cpu")
            
            if encode_from_url:

                audio_paths = (
                    [
                        Path(path).name
                        for path in sample.audio_paths
                    ]
                    if sample.audio_paths
                    else []
                )
                video_paths = (
                    [
                        Path(path).name
                        for path in sample.video_paths
                    ]
                    if sample.video_paths
                    else []
                )
                image_paths = (
                    [
                        Path(path).name
                        for path in sample.image_paths
                    ]
                    if sample.image_paths
                    else []
                )
            else:
                raise NotImplementedError
                audio_paths = sample.audio_paths
                video_paths = sample.video_paths
                image_paths = sample.image_paths

            modal_embeddings["omnimodal"] = omnimodal_encoder.encode(
                text=sample.question,
                audio_inputs=audio_paths,
                video_inputs=video_paths,
                image_inputs=image_paths
            )

            existing_modalities.append("omnimodal")
        
        else:
          # We directly get the multimodal embeddings from the pre-computed file
          assert omnimodal_encoder, "If using pre-computed modal embeddings you need to pass the same omnimodal encoder to compute the text embeddings!!!"
          existing_modalities = list(modal_embeddings.keys())
          
          question_embedding, answer_embedding, step_embeddings = omnimodal_encoder.encode_text(sample.question, chain.steps)
          question_embedding = question_embedding.to("cpu")
          answer_embedding = answer_embedding.to("cpu")
          step_embeddings = step_embeddings.to("cpu")

        # TODO: handle cases in which multimodal encoding fails
        # if modal_embeddings is None:
        #     embedding_dim = (
        #         multimodal_encoder.get_embedding_dim()
        #         if multimodal_encoder
        #         else (audio_encoder.get_embedding_dim() if audio_encoder else 512)
        #     )
        #     modal_embeddings = torch.zeros(embedding_dim).to("cpu")

        if len(existing_modalities) == 1:
            try:
                modal_embeddings = modal_embeddings[existing_modalities[0]].cpu()
            except AttributeError:
                modal_embeddings = torch.tensor(modal_embeddings[existing_modalities[0]]).cpu()
        else:
            for modality in existing_modalities:
                if modal_embeddings[modality] is not None:
                    try:
                        modal_embeddings[modality] = modal_embeddings[modality].cpu()
                    except AttributeError:
                        modal_embeddings[modality] = torch.tensor(modal_embeddings[modality]).cpu()
        # try:
        #     step_embeddings = step_embeddings.cpu()
        # except:
        #     if text_encoder == "clap":
        #         step_embeddings = torch.zeros(
        #             (1, audio_encoder.get_embedding_dim())
        #         ).to("cpu")
        #     else:
        #         step_embeddings = torch.zeros((1, text_encoder.get_embedding_dim())).to(
        #             "cpu"
        #         )

        chain_embedding_data = {
            "step_embeddings": step_embeddings.cpu(),
            "modal_embeddings": modal_embeddings,
            "question_embedding": question_embedding.cpu(),
            "answer_embedding": answer_embedding.cpu(),
            "modality_type": sample.modality,
            # --- NEW TEXT METADATA FOR NLI/PRM ---
            "text_steps": chain.steps,
            "text_query": sample.question,
            "text_final_answer": chain.final_answer,
        }

        sample_embeddings.append(chain_embedding_data)

    if save_embeddings_dir:
        save_embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_file = save_embeddings_dir / f"{sample.id}.json"

        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        embeddings_serializable = [
            make_serializable(chain_emb) for chain_emb in sample_embeddings
        ]

        with open(embedding_file, "w") as f:
            json.dump(embeddings_serializable, f, indent=2)

        if logger:
            logger.debug(f"Saved embeddings to {embedding_file}")

    sample_scores = []

    for chain_emb in sample_embeddings:
        scores = confidence_scorer(
            step_embeddings=chain_emb["step_embeddings"],
            modal_embeddings=chain_emb["modal_embeddings"],
            question_embedding=chain_emb["question_embedding"],
            answer_embedding=chain_emb["answer_embedding"],
            text_steps=chain_emb.get("text_steps"),
            text_query=chain_emb.get("text_query"),
            text_final_answer=chain_emb.get("text_final_answer"),
        )

        def convert_scores(obj):
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

    return sample_embeddings, sample_scores


def load_embeddings_from_path(
    embeddings_path: str,
    samples: List,
    device: str,
    logger: Optional[logging.Logger] = None,
) -> List[List[Dict[str, torch.Tensor]]]:

    path = Path(embeddings_path)
    text_fields = {"text_steps", "text_query", "text_final_answer", "modality_type"}

    # --- NEW: Recursive helper to restore tensors from dictionaries/lists ---
    def restore_tensors(obj, key_name=None):
        # Base case: text metadata fields stay exactly as they are
        if key_name in text_fields:
            return obj

        if isinstance(obj, dict):
            # Recursively restore values in nested dictionaries
            return {k: restore_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Convert numerical lists back to tensors and push to device.
            # (Safety check: ignore empty lists or lists of strings)
            if len(obj) == 0 or isinstance(obj[0], str):
                return obj
            return torch.tensor(obj, dtype=torch.float32).to(device)

        return obj

    # ------------------------------------------------------------------------

    if path.is_dir():
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

            sample_list = []
            for chain_emb in sample_embeddings_data:
                # Use the recursive helper here
                chain_dict = {
                    k: restore_tensors(v, key_name=k) for k, v in chain_emb.items()
                }
                sample_list.append(chain_dict)

            embeddings.append(sample_list)

        if logger:
            logger.info(f"Loaded embeddings for {len(embeddings)} samples")

    else:
        if logger:
            logger.info(f"Loading embeddings from file {embeddings_path}...")

        with open(embeddings_path, "r") as f:
            embeddings_data = json.load(f)

        embeddings = []
        for sample_embs in embeddings_data:
            sample_list = []
            for chain_emb in sample_embs:
                # Use the recursive helper here too
                chain_dict = {
                    k: restore_tensors(v, key_name=k) for k, v in chain_emb.items()
                }
                sample_list.append(chain_dict)
            embeddings.append(sample_list)

        if logger:
            logger.info(f"Loaded embeddings for {len(embeddings)} samples")

    return embeddings


# ... (save_results remains identical to your script) ...
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
    if save_cots:
        assert cots, "cots list is empty! All generation efforts have failed!"
        with open(save_cots, "w") as f:
            json.dump(
                [[convert_to_dict(chain) for chain in chains] for chains in cots],
                f,
                indent=2,
            )
        if logger:
            logger.info(f"Saved CoTs to {save_cots}")

    if embeddings and save_embeddings:

        # --- NEW: Recursive helper to handle nested dicts/lists of tensors ---
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        # ---------------------------------------------------------------------

        embeddings_path = Path(save_embeddings)

        save_as_directory = embeddings_path.suffix == "" or embeddings_path.is_dir()

        if save_as_directory:
            if not samples:
                if logger:
                    logger.error(
                        "Cannot save embeddings to directory without sample IDs"
                    )
                return

            embeddings_path.mkdir(parents=True, exist_ok=True)

            for sample, sample_embs in zip(samples, embeddings):
                embedding_file = embeddings_path / f"{sample.id}.json"

                # Apply the recursive helper to the entire chain dictionary
                embeddings_serializable = [
                    make_serializable(chain_emb) for chain_emb in sample_embs
                ]

                with open(embedding_file, "w") as f:
                    json.dump(embeddings_serializable, f, indent=2)

            if logger:
                logger.info(
                    f"Saved {len(embeddings)} embedding files to directory {save_embeddings}"
                )
        else:
            embeddings_serializable = []
            for sample_embs in embeddings:
                # Apply the recursive helper here as well
                sample_list = [
                    make_serializable(chain_emb) for chain_emb in sample_embs
                ]
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
    """Main function to run the multimodal CoT confidence scoring experiment.
    In order to use Omnimodal models on the input modalities separately,
    the same model should be passed as both the text, multimodal (i.e. image/video) and audio encoder, and the code handles it accordingly.
    The omnimodal argument, instead, will encode all modalities in a single embedding with the omnimodal encoder.
    """
    parser = argparse.ArgumentParser(
        description="Run multimodal CoT confidence scoring experiments with support for images, audio, and text-based metrics (NLI/PRM)"
    )

    # --- Dataset & Generation Args (Skipping definition blocks for brevity in visual, matching your original script) ---
    parser.add_argument(
        "--data_path", type=str, default="uno-bench", help="Path to UNO-Bench dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--modality_filter", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--cot_model_name", type=str, default="Qwen/Qwen2-Audio-7B-Instruct"
    )
    parser.add_argument("--cot_model_type", type=str, default="qwen2_audio")
    parser.add_argument("--use_openai_api", action="store_true")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--num_chains", type=int, default=20)
    parser.add_argument("--num_chains_for_scoring", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument(
        "--text_encoder", type=str, default=None
    )
    parser.add_argument("--multimodal_encoder", type=str, default=None)
    parser.add_argument("--audio_encoder", type=str, default=None)
    parser.add_argument("--omnimodal_encoder", type=str, default=None)
    parser.add_argument(
        "--encode_from_url",
        action="store_true",
        help="Download/encode media directly from HF URLs",
    )
    parser.add_argument(
        "--modal_embeddings",
        type=str,
        default=None,
        help="If included, this is the path to the pre-computed modal embeddings (still different from loading embeddings, as text embeddings are computed on the fly)"
            )

    # Coherence metric arguments (Embedding Base)
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "dot"],
    )
    parser.add_argument(
        "--aggregation", type=str, default="mean", choices=["mean", "min", "max"]
    )
    parser.add_argument("--goal_directedness_weight", type=float, default=0.6)
    parser.add_argument("--smoothness_weight", type=float, default=0.4)
    parser.add_argument("--contrastive_margin", type=float, default=0.2)
    parser.add_argument("--use_attention", action="store_true")

    # Confidence scoring arguments
    parser.add_argument("--use_density_model", action="store_true")
    parser.add_argument(
        "--density_model_type", type=str, default="kde", choices=["kde", "gmm"]
    )

    # --- NEW: NLI AND PRM ARGUMENTS ---
    parser.add_argument(
        "--use_nli", action="store_true", help="Enable NLI metric scoring"
    )
    parser.add_argument(
        "--nli_model_name", type=str, default="cross-encoder/nli-deberta-v3-large"
    )
    parser.add_argument(
        "--use_prm", action="store_true", help="Enable PRM metric scoring"
    )
    parser.add_argument("--prm_model_name", type=str, default="sail/ActPRM-X")

    # Weights for final confidence aggregation
    parser.add_argument("--internal_weight", type=float, default=0.3)
    parser.add_argument("--cross_modal_weight", type=float, default=0.3)
    parser.add_argument("--density_weight", type=float, default=0.1)
    parser.add_argument("--nli_weight", type=float, default=0.15)
    parser.add_argument("--prm_weight", type=float, default=0.15)

    # Save and General arguments
    parser.add_argument("--save_cots", type=str, default=None)
    parser.add_argument("--save_embeddings", type=str, default=None)
    parser.add_argument("--save_scores", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_cot_generation", action="store_true")
    parser.add_argument("--load_cots", type=str, default=None)
    parser.add_argument("--skip_embedding_extraction", action="store_true")
    parser.add_argument("--load_embeddings", type=str, default=None)
    parser.add_argument("--sequential_processing", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    logger = setup_logger(args.log_file)

    logger.info("=" * 80)
    logger.info("Multimodal CoT Confidence Scoring Experiment")
    logger.info("=" * 80)

    # Load dataset
    unobench = UNOBenchLoader(
        data_path=args.data_path, split=args.split, modality_filter=args.modality_filter
    )
    samples = list(unobench)
    if args.max_samples:
        samples = samples[: args.max_samples]

    # Generate or load CoT chains
    cots = None
    if not args.skip_cot_generation:
        logger.info("Generating CoT chains...")
        if args.use_openai_api:
            generator = OpenAICoTGenerator(
                model_name=args.cot_model_name, api_key=args.openai_api_key
            )
        else:
            generator = CoTGenerator(
                model_name=args.cot_model_name, model_type=args.cot_model_type
            )

        cots = []
        for idx, sample in enumerate(samples):
            try:
                chains = generator.generate_cot_from_sample(
                    sample,
                    max_new_tokens=args.max_new_tokens,
                    num_chains=args.num_chains,
                )
            except Exception as e:
                if args.verbose:
                    print(f"Could not process sample at index {idx}.\nException: {e}")
                continue
            cots.append(chains)
    elif args.load_cots:
        with open(args.load_cots, "r") as f:
            cots = json.load(f)
        cots = [[CoTChain(**chain) for chain in cot] for cot in cots]
    else:
        logger.error("Must either generate CoTs or provide --load_cots path")
        return

    if args.save_cots:
        save_results(
            cots=cots, save_cots=args.save_cots, samples=samples, logger=logger
        )

    if args.num_chains_for_scoring is not None and cots is not None:
        cots = [chains[: args.num_chains_for_scoring] for chains in cots]

    # Setup coherence metrics and confidence scorer (Passing device to init the PRM correctly)
    logger.info("Setting up coherence metrics and confidence scorer...")
    internal_metric, cross_modal_metric = setup_coherence_metrics(args)
    confidence_scorer = setup_confidence_scorer(
        args, internal_metric, cross_modal_metric, device, logger
    )

    embeddings = None
    scores = None

    if args.sequential_processing:
        logger.info("Using sequential processing mode")

        if not args.skip_embedding_extraction:
            text_encoder, multimodal_encoder, audio_encoder, omnimodal_encoder = (
                setup_encoders(args, device, logger)
            )
            scores = []
            save_embeddings_dir = (
                Path(args.save_embeddings) if args.save_embeddings else None
            )

            if args.modal_embeddings:
                with open(args.modal_embeddings) as f:
                    modal_embeddings = json.load(f)

            for idx, (sample, sample_chains) in enumerate(zip(samples, cots)):
                if args.modal_embeddings:
                    modal_embeddings_sample = modal_embeddings[idx]
                else:
                    modal_embeddings_sample = None
                _, sample_scores = process_sample_sequential(
                    sample=sample,
                    sample_idx=idx,
                    cot_chains=sample_chains,
                    text_encoder=text_encoder,
                    multimodal_encoder=multimodal_encoder,
                    audio_encoder=audio_encoder,
                    omnimodal_encoder=omnimodal_encoder,
                    confidence_scorer=confidence_scorer,
                    device=device,
                    save_embeddings_dir=save_embeddings_dir,
                    logger=logger,
                    encode_from_url=args.encode_from_url,
                    modal_embeddings=modal_embeddings_sample
                )
                scores.append(sample_scores)

        elif args.load_embeddings:
            embeddings_path = Path(args.load_embeddings)
            if not embeddings_path.is_dir():
                logger.error(
                    "Sequential processing requires a directory for --load_embeddings"
                )
                return

            scores = []
            text_fields = [
                "text_steps",
                "text_query",
                "text_final_answer",
                "modality_type",
            ]

            for idx, sample in enumerate(samples):
                embedding_file = embeddings_path / f"{sample.id}.json"
                if not embedding_file.exists():
                    continue

                with open(embedding_file, "r") as f:
                    sample_embeddings_data = json.load(f)

                sample_chains = cots[idx] if cots else None
                sample_embeddings = []

                def restore_tensors(obj, key_name=None):
                    if key_name in text_fields:
                        return obj
                    if isinstance(obj, dict):
                        return {k: restore_tensors(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        if len(obj) == 0 or isinstance(obj[0], str):
                            return obj
                        return torch.tensor(obj, dtype=torch.float32).to(device)
                    return obj

                for chain_idx, chain_emb in enumerate(sample_embeddings_data):

                    chain_dict = {
                        k: restore_tensors(v, key_name=k) for k, v in chain_emb.items()
                    }

                    # --- BACKWARD COMPATIBILITY INJECTION ---
                    if (
                        "text_steps" not in chain_dict
                        and sample_chains
                        and chain_idx < len(sample_chains)
                    ):
                        chain_dict["text_steps"] = sample_chains[chain_idx].steps
                        chain_dict["text_query"] = sample.question
                        chain_dict["text_final_answer"] = sample_chains[
                            chain_idx
                        ].final_answer

                    sample_embeddings.append(chain_dict)

                if args.num_chains_for_scoring is not None:
                    sample_embeddings = sample_embeddings[: args.num_chains_for_scoring]

                sample_scores = []
                for chain_emb in sample_embeddings:
                    scores_result = confidence_scorer(
                        step_embeddings=chain_emb["step_embeddings"],
                        modal_embeddings=chain_emb["modal_embeddings"],
                        question_embedding=chain_emb["question_embedding"],
                        answer_embedding=chain_emb["answer_embedding"],
                        text_steps=chain_emb.get("text_steps"),
                        text_query=chain_emb.get("text_query"),
                        text_final_answer=chain_emb.get("text_final_answer"),
                    )

                    def convert_scores(obj):
                        if isinstance(obj, torch.Tensor):
                            return (
                                obj.item()
                                if obj.numel() == 1
                                else obj.cpu().numpy().tolist()
                            )
                        elif isinstance(obj, dict):
                            return {k: convert_scores(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_scores(item) for item in obj]
                        else:
                            return obj

                    sample_scores.append(convert_scores(scores_result))
                scores.append(sample_scores)
        else:
            logger.error("Must extract embeddings or provide --load_embeddings")
            return

    else:
        logger.info("Using batch processing mode")
        if not args.skip_embedding_extraction:
            text_encoder, multimodal_encoder, audio_encoder, omnimodal_encoder = (
                setup_encoders(args, device, logger)
            )
            embeddings = extract_embeddings(
                samples,
                cots,
                text_encoder,
                multimodal_encoder,
                audio_encoder,
                omnimodal_encoder,
                device,
                logger,
                encode_from_url=args.encode_from_url,
            )
        elif args.load_embeddings:
            embeddings = load_embeddings_from_path(
                args.load_embeddings, samples, device, cots, logger
            )
            if args.num_chains_for_scoring is not None:
                embeddings = [
                    sample_embs[: args.num_chains_for_scoring]
                    for sample_embs in embeddings
                ]
        else:
            logger.error("Must extract embeddings or provide --load_embeddings path")
            return

        scores = compute_coherence_scores(embeddings, confidence_scorer, logger)

    logger.info("Saving results...")
    save_results(
        cots=cots,
        embeddings=embeddings if args.save_embeddings else None,
        scores=scores,
        save_embeddings=args.save_embeddings,
        save_scores=args.save_scores,
        samples=samples,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
