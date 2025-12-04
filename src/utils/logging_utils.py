"""
Logging utilities for experiments.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


def setup_logger(
    name: str = "multimodal_cot",
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_metrics(
    metrics: Dict,
    logger: Optional[logging.Logger] = None,
    prefix: str = ""
):
    """
    Log metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        logger: Logger instance
        prefix: Prefix for log messages
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"{prefix}Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    logger.info(f"    {sub_key}: {sub_value:.4f}")
                else:
                    logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def save_results(
    results: Dict,
    save_path: str,
    logger: Optional[logging.Logger] = None
):
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        save_path: Path to save file
        logger: Logger instance
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any non-serializable objects
    serializable_results = make_serializable(results)

    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    if logger:
        logger.info(f"Results saved to {save_path}")


def make_serializable(obj):
    """
    Convert objects to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        Serializable version
    """
    import torch
    import numpy as np

    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj


class ExperimentLogger:
    """
    Comprehensive logger for experiments with support for multiple backends.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of experiment
            log_dir: Directory for logs
            use_tensorboard: Use TensorBoard logging
            use_wandb: Use Weights & Biases logging
            wandb_project: W&B project name
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup standard logger
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=str(self.log_dir)
        )

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(
                log_dir=str(self.log_dir / "tensorboard")
            )
        else:
            self.tb_writer = None

        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(
                project=wandb_project or "multimodal-cot-confidence",
                name=experiment_name,
                dir=str(self.log_dir)
            )

        self.step = 0

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log scalar metric."""
        if step is None:
            step = self.step

        self.logger.info(f"Step {step} - {name}: {value:.4f}")

        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)

        if self.use_wandb:
            import wandb
            wandb.log({name: value}, step=step)

    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log multiple metrics."""
        if step is None:
            step = self.step

        log_metrics(metrics, self.logger, prefix=f"Step {step} - ")

        if self.tb_writer:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(name, value, step)

        if self.use_wandb:
            import wandb
            # Flatten nested metrics
            flat_metrics = {}
            for name, value in metrics.items():
                if isinstance(value, dict):
                    for sub_name, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            flat_metrics[f"{name}/{sub_name}"] = sub_value
                elif isinstance(value, (int, float)):
                    flat_metrics[name] = value
            wandb.log(flat_metrics, step=step)

    def log_config(self, config: Dict):
        """Log experiment configuration."""
        self.logger.info("Experiment Configuration:")
        log_metrics(config, self.logger)

        # Save config to file
        config_path = self.log_dir / "config.json"
        save_results(config, str(config_path), self.logger)

        if self.use_wandb:
            import wandb
            wandb.config.update(config)

    def increment_step(self):
        """Increment global step counter."""
        self.step += 1

    def close(self):
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb:
            import wandb
            wandb.finish()
