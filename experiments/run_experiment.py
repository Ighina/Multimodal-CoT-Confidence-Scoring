"""
Main experiment runner for multimodal CoT confidence scoring.
"""

import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import UNOBenchLoader, CoTGenerator, DataProcessor, CoTDataset
from src.embeddings import TextEncoder, MultimodalEncoder, EmbeddingCache
from src.coherence import (
    InternalCoherenceMetric,
    CrossModalCoherenceMetric,
    ChainConfidenceScorer
)
from src.coherence_models import ConfidenceHead, KDEDensityModel, GMMDensityModel
from src.baselines import (
    CoTLengthBaseline,
    LogProbabilityBaseline,
    MajorityVoteBaseline,
    SemanticEntropyBaseline
)
from src.evaluation import ConfidenceEvaluator
from src.utils import setup_logger, ExperimentLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_encoders(config: dict, device: str):
    """Setup text and multimodal encoders."""
    text_encoder = TextEncoder(
        model_name=config['model']['text_encoder']['name'],
        device=device
    )

    multimodal_encoder = MultimodalEncoder(
        model_name=config['model']['multimodal_encoder']['name'],
        device=device
    )

    return text_encoder, multimodal_encoder


def setup_coherence_metrics(config: dict):
    """Setup coherence metrics."""
    internal_metric = InternalCoherenceMetric(
        similarity_metric=config['coherence']['internal']['similarity_metric'],
        aggregation=config['coherence']['internal']['aggregation'],
        goal_directedness_weight=config['coherence']['internal']['goal_directedness_weight'],
        smoothness_weight=config['coherence']['internal']['smoothness_weight']
    )

    cross_modal_metric = CrossModalCoherenceMetric(
        similarity_metric=config['coherence']['cross_modal']['similarity_metric'],
        contrastive_margin=config['coherence']['cross_modal']['contrastive_margin'],
        use_attention=config['coherence']['cross_modal']['attention_weighted']
    )

    return internal_metric, cross_modal_metric


def setup_confidence_scorer(config: dict, internal_metric, cross_modal_metric):
    """Setup chain confidence scorer."""
    # Setup density model if enabled
    density_model = None
    if config['coherence']['chain_confidence']['use_density_model']:
        density_type = config['coherence']['chain_confidence']['density_model_type']

        if density_type == 'kde':
            density_model = KDEDensityModel()
        elif density_type == 'gmm':
            density_model = GMMDensityModel()

    # Create confidence scorer
    scorer = ChainConfidenceScorer(
        internal_metric=internal_metric,
        cross_modal_metric=cross_modal_metric,
        density_model=density_model,
        internal_weight=config['coherence']['chain_confidence']['internal_weight'],
        cross_modal_weight=config['coherence']['chain_confidence']['cross_modal_weight'],
        density_weight=config['coherence']['chain_confidence']['density_weight']
    )

    return scorer


def extract_embeddings(
    samples,
    cot_chains,
    text_encoder,
    multimodal_encoder,
    cache=None
):
    """Extract embeddings for all samples and chains."""
    all_embeddings = []

    for sample, chains in zip(samples, cot_chains):
        sample_embeddings = []

        for chain in chains:
            # Extract step embeddings
            step_embeddings = text_encoder.encode_cot_steps(
                chain.steps,
                question=sample.question
            )

            # Extract image embeddings
            if sample.images:
                image_embeddings = multimodal_encoder.encode_images(sample.images)
            else:
                # No images - use zero embeddings
                image_embeddings = torch.zeros(
                    multimodal_encoder.get_embedding_dim()
                )

            # Extract question and answer embeddings
            question_embedding = text_encoder(sample.question)
            answer_embedding = text_encoder(chain.final_answer)

            chain_embedding_data = {
                'step_embeddings': step_embeddings,
                'image_embeddings': image_embeddings,
                'question_embedding': question_embedding,
                'answer_embedding': answer_embedding
            }

            sample_embeddings.append(chain_embedding_data)

        all_embeddings.append(sample_embeddings)

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Run multimodal CoT confidence scoring experiments"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Name of experiment'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Set random seed
    torch.manual_seed(config['seed'])

    # Setup logging
    experiment_name = args.experiment_name or 'multimodal_cot_experiment'
    exp_logger = ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=config['logging']['log_dir'],
        use_tensorboard=config['logging']['use_tensorboard'],
        use_wandb=config['logging']['use_wandb'],
        wandb_project=config['logging'].get('wandb_project')
    )

    exp_logger.log_config(config)
    exp_logger.logger.info(f"Using device: {device}")

    # Load data
    exp_logger.logger.info("Loading UNO-Bench dataset...")
    uno_loader = UNOBenchLoader(
        data_path=config['data']['data_path'],
        split=config['data']['split']
    )

    stats = uno_loader.get_statistics()
    exp_logger.logger.info(f"Dataset statistics: {stats}")

    # Generate CoT chains (or load pre-generated)
    exp_logger.logger.info("Generating CoT chains...")
    cot_generator = CoTGenerator(
        model_name=config['model']['lvlm']['name'],
        device=device,
        cot_prompt_template=config['cot_generation']['cot_prompt_template']
    )

    # For demo purposes, we'll use mock chains
    # In practice, you would generate real chains here
    exp_logger.logger.info("Note: Using mock CoT generation for demo")

    # Setup encoders
    exp_logger.logger.info("Setting up encoders...")
    text_encoder, multimodal_encoder = setup_encoders(config, device)

    # Setup coherence metrics
    exp_logger.logger.info("Setting up coherence metrics...")
    internal_metric, cross_modal_metric = setup_coherence_metrics(config)

    # Setup confidence scorer
    exp_logger.logger.info("Setting up confidence scorer...")
    confidence_scorer = setup_confidence_scorer(
        config,
        internal_metric,
        cross_modal_metric
    )

    # Setup baselines
    baselines = {
        'cot_length': CoTLengthBaseline(),
        'semantic_entropy': SemanticEntropyBaseline(),
    }

    # Setup evaluator
    evaluator = ConfidenceEvaluator(
        scorer=confidence_scorer,
        baseline_methods=baselines,
        device=device
    )

    exp_logger.logger.info("Experiment setup complete!")
    exp_logger.logger.info("To run the full pipeline:")
    exp_logger.logger.info("1. Generate CoT chains for all samples")
    exp_logger.logger.info("2. Extract embeddings")
    exp_logger.logger.info("3. Compute coherence scores")
    exp_logger.logger.info("4. Train confidence head (if using learned model)")
    exp_logger.logger.info("5. Evaluate on test set")
    exp_logger.logger.info("6. Run ablations and comparisons")

    # Close logger
    exp_logger.close()


if __name__ == '__main__':
    main()
