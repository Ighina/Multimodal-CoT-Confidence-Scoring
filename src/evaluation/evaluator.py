"""
Comprehensive evaluator for confidence scoring methods.
"""

from typing import Dict, List, Optional, Callable
import numpy as np
import torch
from tqdm import tqdm

from .metrics import evaluate_confidence_scores, compare_methods
from ..coherence.chain_confidence import ChainConfidenceScorer


class ConfidenceEvaluator:
    """
    Evaluate confidence scoring methods on test data.
    """

    def __init__(
        self,
        scorer: Optional[ChainConfidenceScorer] = None,
        baseline_methods: Optional[Dict[str, Callable]] = None,
        device: str = "cuda"
    ):
        """
        Initialize evaluator.

        Args:
            scorer: Main confidence scorer to evaluate
            baseline_methods: Dictionary of baseline methods
            device: Device for computation
        """
        self.scorer = scorer
        self.baseline_methods = baseline_methods or {}
        self.device = device

    def evaluate_scorer(
        self,
        test_data: List[Dict],
        n_bins: int = 10,
        abstention_thresholds: Optional[List[float]] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Evaluate the main scorer on test data.

        Args:
            test_data: List of test samples with embeddings and labels
            n_bins: Number of calibration bins
            abstention_thresholds: Thresholds for abstention analysis
            show_progress: Show progress bar

        Returns:
            Evaluation results
        """
        confidences = []
        labels = []

        iterator = tqdm(test_data) if show_progress else test_data

        for sample in iterator:
            # Extract data
            step_embeddings = sample['step_embeddings']
            image_embeddings = sample['image_embeddings']
            label = sample['label']

            # Compute confidence
            with torch.no_grad():
                result = self.scorer(
                    step_embeddings=step_embeddings,
                    image_embeddings=image_embeddings
                )
                confidence = result['confidence'].item()

            confidences.append(confidence)
            labels.append(label)

        # Convert to numpy
        confidences = np.array(confidences)
        labels = np.array(labels)

        # Evaluate
        results = evaluate_confidence_scores(
            confidences=confidences,
            labels=labels,
            n_bins=n_bins,
            abstention_thresholds=abstention_thresholds
        )

        return results

    def evaluate_baselines(
        self,
        test_data: List[Dict],
        n_bins: int = 10,
        abstention_thresholds: Optional[List[float]] = None,
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Evaluate all baseline methods.

        Args:
            test_data: Test data
            n_bins: Number of calibration bins
            abstention_thresholds: Thresholds for abstention
            show_progress: Show progress bar

        Returns:
            Dictionary mapping method names to results
        """
        baseline_results = {}

        for method_name, method_fn in self.baseline_methods.items():
            print(f"Evaluating baseline: {method_name}")

            confidences = []
            labels = []

            iterator = tqdm(test_data, desc=method_name) if show_progress else test_data

            for sample in iterator:
                # Compute confidence using baseline
                try:
                    confidence = method_fn(sample)
                    confidences.append(confidence)
                    labels.append(sample['label'])
                except Exception as e:
                    print(f"Error in {method_name}: {e}")
                    continue

            if confidences:
                confidences = np.array(confidences)
                labels = np.array(labels)

                results = evaluate_confidence_scores(
                    confidences=confidences,
                    labels=labels,
                    n_bins=n_bins,
                    abstention_thresholds=abstention_thresholds
                )

                baseline_results[method_name] = results

        return baseline_results

    def compare_all_methods(
        self,
        test_data: List[Dict],
        include_baselines: bool = True,
        **eval_kwargs
    ) -> Dict:
        """
        Compare main scorer and all baselines.

        Args:
            test_data: Test data
            include_baselines: Whether to evaluate baselines
            **eval_kwargs: Additional evaluation arguments

        Returns:
            Comparison results
        """
        all_results = {}

        # Evaluate main scorer
        if self.scorer is not None:
            print("Evaluating main scorer...")
            scorer_results = self.evaluate_scorer(test_data, **eval_kwargs)
            all_results['coherence_scorer'] = scorer_results

        # Evaluate baselines
        if include_baselines and self.baseline_methods:
            baseline_results = self.evaluate_baselines(test_data, **eval_kwargs)
            all_results.update(baseline_results)

        # Compare
        comparison = compare_methods(all_results)

        return {
            'individual_results': all_results,
            'comparison': comparison
        }

    def ablation_study(
        self,
        test_data: List[Dict],
        ablations: List[str]
    ) -> Dict[str, Dict]:
        """
        Conduct ablation study on scorer components.

        Args:
            test_data: Test data
            ablations: List of ablation names
                ('internal_only', 'cross_modal_only', 'no_density')

        Returns:
            Ablation results
        """
        results = {}

        for ablation in ablations:
            print(f"Running ablation: {ablation}")

            # Modify scorer for ablation
            modified_scorer = self._create_ablated_scorer(ablation)

            # Evaluate
            confidences = []
            labels = []

            for sample in tqdm(test_data, desc=ablation):
                step_embeddings = sample['step_embeddings']
                image_embeddings = sample['image_embeddings']
                label = sample['label']

                with torch.no_grad():
                    result = modified_scorer(
                        step_embeddings=step_embeddings,
                        image_embeddings=image_embeddings
                    )
                    confidence = result['confidence'].item()

                confidences.append(confidence)
                labels.append(label)

            confidences = np.array(confidences)
            labels = np.array(labels)

            results[ablation] = evaluate_confidence_scores(confidences, labels)

        return results

    def _create_ablated_scorer(self, ablation: str) -> ChainConfidenceScorer:
        """
        Create modified scorer for ablation study.

        Args:
            ablation: Ablation type

        Returns:
            Modified scorer
        """
        # Clone scorer and modify weights
        from copy import deepcopy
        scorer = deepcopy(self.scorer)

        if ablation == 'internal_only':
            # Only use internal coherence
            scorer.weights = torch.tensor([1.0, 0.0, 0.0])
        elif ablation == 'cross_modal_only':
            # Only use cross-modal coherence
            scorer.weights = torch.tensor([0.0, 1.0, 0.0])
        elif ablation == 'no_density':
            # Remove density component
            scorer.weights = torch.tensor([0.6, 0.4, 0.0])
            scorer.density_model = None

        return scorer

    def reranking_evaluation(
        self,
        test_data_with_multiple_chains: List[Dict]
    ) -> Dict:
        """
        Evaluate chain reranking using confidence scores.

        For each problem with multiple chains, select the highest-confidence
        chain and check if it's correct.

        Args:
            test_data_with_multiple_chains: Test data where each sample
                has multiple chains

        Returns:
            Reranking results
        """
        correct_by_confidence = 0
        correct_by_random = 0
        total = 0

        for sample in test_data_with_multiple_chains:
            chains = sample['chains']  # List of chain data
            true_label = sample['label']

            # Compute confidence for each chain
            chain_confidences = []
            chain_labels = []

            for chain in chains:
                with torch.no_grad():
                    result = self.scorer(
                        step_embeddings=chain['step_embeddings'],
                        image_embeddings=chain['image_embeddings']
                    )
                    confidence = result['confidence'].item()

                chain_confidences.append(confidence)
                chain_labels.append(chain.get('is_correct', 0))

            # Select highest-confidence chain
            best_idx = np.argmax(chain_confidences)
            selected_correct = chain_labels[best_idx]

            # Random selection baseline
            random_correct = np.mean(chain_labels)

            correct_by_confidence += selected_correct
            correct_by_random += random_correct
            total += 1

        return {
            'accuracy_by_confidence': correct_by_confidence / total,
            'accuracy_by_random': correct_by_random / total,
            'improvement': (correct_by_confidence - correct_by_random) / total
        }
