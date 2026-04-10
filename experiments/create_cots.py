"""
Chain-of-Thought (CoT) generation module.
Handles generation and loading of CoT chains for experiments.
"""

import json
from pathlib import Path
from typing import List, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.cot_generator import CoTGenerator, CoTChain, OpenAICoTGenerator


def generate_cots(
    samples: List,
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
    model_type: str = "qwen2_audio",
    num_chains: int = 20,
    max_new_tokens: Optional[int] = None,
    use_openai_api: bool = False,
    openai_api_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
) -> List[List[CoTChain]]:
    """
    Generate CoT chains for all samples.

    Args:
        samples: List of samples to generate CoTs for
        model_name: Name or path of the model to use
        model_type: Type of model (e.g., "qwen2_audio")
        num_chains: Number of chains to generate per sample
        max_new_tokens: Maximum number of new tokens to generate
        use_openai_api: Whether to use OpenAI API
        openai_api_key: OpenAI API key (if using OpenAI API)
        logger: Logger instance
        verbose: Whether to print verbose output

    Returns:
        List of CoT chains for each sample
    """
    if logger:
        logger.info("Generating CoT chains...")

    if use_openai_api:
        generator = OpenAICoTGenerator(
            model_name=model_name,
            api_key=openai_api_key
        )
    else:
        generator = CoTGenerator(
            model_name=model_name,
            model_type=model_type
        )

    cots = []
    for idx, sample in enumerate(samples):
        try:
            chains = generator.generate_cot_from_sample(
                sample,
                max_new_tokens=max_new_tokens,
                num_chains=num_chains
            )
            cots.append(chains)

            if logger and (idx + 1) % 10 == 0:
                logger.info(f"Generated CoTs for {idx + 1}/{len(samples)} samples")

        except Exception as e:
            if verbose or logger:
                msg = f"Could not process sample at index {idx}. Exception: {e}"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
            continue

    if logger:
        logger.info(f"CoT generation complete. Generated {len(cots)} samples.")

    return cots


def load_cots(
    cots_path: str,
    logger: Optional[logging.Logger] = None,
) -> List[List[CoTChain]]:
    """
    Load CoT chains from file.

    Args:
        cots_path: Path to CoT chains file (JSON)
        logger: Logger instance

    Returns:
        List of CoT chains for each sample
    """
    if logger:
        logger.info(f"Loading CoTs from {cots_path}...")

    with open(cots_path, "r") as f:
        cots_data = json.load(f)

    cots = [[CoTChain(**chain) for chain in cot] for cot in cots_data]

    if logger:
        logger.info(f"Loaded {len(cots)} CoT samples")

    return cots


def convert_to_dict(chain: CoTChain) -> dict:
    """
    Convert a CoTChain object to a dictionary for serialization.

    Args:
        chain: CoTChain object

    Returns:
        Dictionary representation
    """
    return {
        "metadata": chain.metadata,
        "final_answer": chain.final_answer,
        "text": chain.text,
        "steps": chain.steps,
        "log_probs": chain.log_probs,
    }


def save_cots(
    cots: List[List[CoTChain]],
    save_path: str,
    logger: Optional[logging.Logger] = None,
):
    """
    Save CoT chains to file.

    Args:
        cots: List of CoT chains for each sample
        save_path: Path to save CoTs to
        logger: Logger instance
    """
    if not cots:
        raise ValueError("CoTs list is empty! All generation efforts have failed!")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(
            [[convert_to_dict(chain) for chain in chains] for chains in cots],
            f,
            indent=2,
        )

    if logger:
        logger.info(f"Saved CoTs to {save_path}")


def filter_cots_by_count(
    cots: List[List[CoTChain]],
    num_chains: int,
    logger: Optional[logging.Logger] = None,
) -> List[List[CoTChain]]:
    """
    Filter CoTs to only keep first N chains per sample.

    Args:
        cots: List of CoT chains for each sample
        num_chains: Number of chains to keep per sample
        logger: Logger instance

    Returns:
        Filtered list of CoT chains
    """
    filtered = [chains[:num_chains] for chains in cots]

    if logger:
        logger.info(f"Filtered CoTs to {num_chains} chains per sample")

    return filtered
