"""
Embedding extraction module for experiments.
Handles extraction and loading of embeddings from CoT chains.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Union
import logging
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import TextEncoder, MultimodalEncoder, AudioEncoder, OmnimodalEncoder
from src.dataset.cot_generator import CoTChain


def setup_encoders(
    text_encoder: str = "sentence-transformers/all-mpnet-base-v2",
    multimodal_encoder: Optional[str] = None,
    audio_encoder: Optional[str] = None,
    omnimodal_encoder: Optional[str] = None,
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
) -> tuple:
    """
    Setup text, multimodal, audio, and omnimodal encoders.

    Args:
        text_encoder: Name of text encoder model
        multimodal_encoder: Name of multimodal (image) encoder model
        audio_encoder: Name of audio encoder model
        omnimodal_encoder: Name of omnimodal encoder model (for unified encoding)
        device: Device to run on
        logger: Logger instance

    Returns:
        Tuple of (text_encoder, multimodal_encoder, audio_encoder, omnimodal_encoder)
    """
    if logger:
        logger.info("Setting up encoders...")

    text_enc = None
    multimodal_enc = None
    audio_enc = None
    omnimodal_enc = None

    if audio_encoder and "clap" in audio_encoder.lower():
        text_encoder_name = "clap"
    else:
        text_enc = TextEncoder(model_name=text_encoder, device=device)
        text_encoder_name = text_encoder

    if logger:
        logger.info(f"Text encoder loaded: {text_encoder_name}")

    if multimodal_encoder:
        multimodal_enc = MultimodalEncoder(
            model_name=multimodal_encoder, device=device
        )
        if logger:
            logger.info(f"Multimodal encoder loaded: {multimodal_encoder}")

    if audio_encoder:
        use_clap = "clap" in audio_encoder.lower()
        audio_enc = AudioEncoder(
            model_name=audio_encoder, device=device, use_clap=use_clap
        )
        if logger:
            logger.info(f"Audio encoder loaded: {audio_encoder}")
        if use_clap:
            text_encoder_name = "clap"

    if omnimodal_encoder:
        omnimodal_enc = OmnimodalEncoder(
            model_name=omnimodal_encoder, device=device
        )
        if logger:
            logger.info(f"Omnimodal encoder loaded: {omnimodal_encoder}")

    return text_encoder_name if isinstance(text_enc, str) or text_enc is None else text_enc, multimodal_enc, audio_enc, omnimodal_enc


def extract_embeddings_batch(
    samples: List,
    cot_chains: List[List[CoTChain]],
    text_encoder: Union[str, TextEncoder],
    multimodal_encoder: Optional[MultimodalEncoder],
    audio_encoder: Optional[AudioEncoder],
    omnimodal_encoder: Optional[OmnimodalEncoder] = None,
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
) -> List[List[Dict[str, torch.Tensor]]]:
    """
    Extract embeddings for all samples and chains in batch mode.

    Args:
        samples: List of samples
        cot_chains: List of CoT chains for each sample
        text_encoder: Text encoder (or "clap" string)
        multimodal_encoder: Multimodal encoder
        audio_encoder: Audio encoder
        omnimodal_encoder: Omnimodal encoder (optional)
        device: Device to run on
        logger: Logger instance

    Returns:
        List of embeddings for each sample and chain
    """
    if logger:
        logger.info("Extracting embeddings...")

    all_embeddings = []

    for idx, (sample, chains) in enumerate(zip(samples, cot_chains)):
        if logger and (idx + 1) % 10 == 0:
            logger.info(f"Processing sample {idx + 1}/{len(samples)}")

        sample_embeddings = []

        for chain in chains:
            if text_encoder != "clap":
                step_embeddings = text_encoder.encode_cot_steps(
                    chain.steps, question=sample.question
                )
            else:
                step_embeddings = None

            modal_embeddings = None

            if sample.images and multimodal_encoder:
                modal_embeddings = multimodal_encoder.encode_images(sample.images)

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
                    if modal_embeddings is None:
                        embedding_dim = (
                            audio_encoder.get_embedding_dim()
                            if audio_encoder
                            else multimodal_encoder.get_embedding_dim()
                        )
                        modal_embeddings = torch.zeros(embedding_dim).to(device)

            if modal_embeddings is None:
                embedding_dim = (
                    multimodal_encoder.get_embedding_dim()
                    if multimodal_encoder
                    else (
                        audio_encoder.get_embedding_dim() if audio_encoder else 512
                    )
                )
                modal_embeddings = torch.zeros(embedding_dim).to(device)

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

            omnimodal_embeddings = None
            if omnimodal_encoder:
                try:
                    omni_emb = omnimodal_encoder.encode_from_sample(sample)
                    omnimodal_embeddings = {
                        "audio": omni_emb.get("audio"),
                        "image": omni_emb.get("image"),
                        "video": omni_emb.get("video"),
                        "omnimodal": omni_emb.get("omnimodal"),
                    }
                except Exception as e:
                    if logger:
                        logger.warning(
                            f"Failed to encode with omnimodal encoder for sample {sample.id}: {e}"
                        )

            chain_embedding_data = {
                "step_embeddings": step_embeddings.cpu() if step_embeddings is not None else None,
                "modal_embeddings": modal_embeddings.cpu(),
                "question_embedding": question_embedding.cpu(),
                "answer_embedding": answer_embedding.cpu(),
                "modality_type": sample.modality,
                "text_steps": chain.steps,
                "text_query": sample.question,
                "text_final_answer": chain.final_answer,
            }

            if omnimodal_embeddings:
                chain_embedding_data["omnimodal_embeddings"] = omnimodal_embeddings

            sample_embeddings.append(chain_embedding_data)

        all_embeddings.append(sample_embeddings)

    if logger:
        logger.info("Embedding extraction complete")

    return all_embeddings


def load_embeddings(
    embeddings_path: str,
    samples: List,
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
) -> List[List[Dict[str, torch.Tensor]]]:
    """
    Load embeddings from file or directory.

    Args:
        embeddings_path: Path to embeddings file or directory
        samples: List of samples (for directory-based loading)
        device: Device to load embeddings to
        logger: Logger instance

    Returns:
        List of embeddings for each sample
    """
    path = Path(embeddings_path)
    text_fields = ["text_steps", "text_query", "text_final_answer", "modality_type"]

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
                chain_dict = {
                    k: (
                        torch.tensor(v).to(device)
                        if isinstance(v, list) and k not in text_fields
                        else v
                    )
                    for k, v in chain_emb.items()
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
                chain_dict = {
                    k: (
                        torch.tensor(v).to(device)
                        if isinstance(v, list) and k not in text_fields
                        else v
                    )
                    for k, v in chain_emb.items()
                }
                sample_list.append(chain_dict)
            embeddings.append(sample_list)

        if logger:
            logger.info(f"Loaded embeddings for {len(embeddings)} samples")

    return embeddings


def save_embeddings(
    embeddings: List[List[Dict[str, torch.Tensor]]],
    save_path: str,
    samples: Optional[List] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Save embeddings to file or directory.

    Args:
        embeddings: List of embeddings for each sample
        save_path: Path to save embeddings to
        samples: List of samples (required for directory-based saving)
        logger: Logger instance
    """
    embeddings_path = Path(save_path)
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
                f"Saved {len(embeddings)} embedding files to directory {save_path}"
            )
    else:
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

        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(embeddings_serializable, f, indent=2)

        if logger:
            logger.info(f"Saved embeddings to {save_path}")
