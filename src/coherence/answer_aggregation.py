"""
Answer aggregation using confidence-weighted voting.

Aggregates multiple reasoning chain outputs by grouping similar/same answers
and weighting them by their confidence scores.
"""

from typing import List, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from ..embeddings.embedding_utils import compute_similarity


class WeightedFrequency:
    """
    Aggregate answers using confidence-weighted frequency.

    For multiple-choice questions: Uses exact string matching.
    For open-ended questions: Uses semantic similarity and clustering.

    The confidence scores are used as weights, and the final aggregated
    score for each answer is normalized by the total sum of all confidence scores.
    """

    def __init__(
        self,
        mode: str = "exact",  # "exact" or "semantic"
        similarity_threshold: float = 0.85,
        clustering_method: str = "dbscan",  # "dbscan" or "agglomerative"
        min_cluster_size: int = 1,
        embedding_model: Optional[nn.Module] = None,
        normalize: bool = True
    ):
        """
        Initialize weighted frequency aggregator.

        Args:
            mode: Aggregation mode - "exact" for multiple-choice, "semantic" for open-ended
            similarity_threshold: Threshold for clustering similar answers (semantic mode)
            clustering_method: Method for clustering ("dbscan" or "agglomerative")
            min_cluster_size: Minimum cluster size for DBSCAN
            embedding_model: Optional model to compute embeddings for answers (if not pre-computed)
            normalize: Whether to normalize weights by total sum
        """
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size
        self.embedding_model = embedding_model
        self.normalize = normalize

    def aggregate_exact(
        self,
        answers: List[str],
        confidence_scores: List[float]
    ) -> Dict[str, float]:
        """
        Aggregate answers using exact string matching.

        Args:
            answers: List of answer strings
            confidence_scores: List of confidence scores (one per answer)

        Returns:
            Dictionary mapping each unique answer to its aggregated confidence score
        """
        if len(answers) != len(confidence_scores):
            raise ValueError("Number of answers must match number of confidence scores")

        # Group by exact answer
        answer_weights = defaultdict(float)

        for answer, confidence in zip(answers, confidence_scores):
            answer_weights[answer] += confidence

        # Normalize by total sum
        if self.normalize:
            total_weight = sum(answer_weights.values())
            if total_weight > 0:
                answer_weights = {
                    ans: weight / total_weight
                    for ans, weight in answer_weights.items()
                }

        return dict(answer_weights)

    def aggregate_semantic(
        self,
        answers: List[str],
        confidence_scores: List[float],
        answer_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Aggregate answers using semantic similarity and clustering.

        Args:
            answers: List of answer strings
            confidence_scores: List of confidence scores
            answer_embeddings: Optional pre-computed embeddings (num_answers, embed_dim)
                             If None, will compute using embedding_model

        Returns:
            Dictionary mapping representative answers to aggregated confidence scores
        """
        if len(answers) != len(confidence_scores):
            raise ValueError("Number of answers must match number of confidence scores")

        if len(answers) == 0:
            return {}

        # Get embeddings
        if answer_embeddings is None:
            if self.embedding_model is None:
                raise ValueError("Must provide either answer_embeddings or embedding_model")
            answer_embeddings = self._compute_embeddings(answers)

        # Cluster similar answers
        clusters = self._cluster_answers(answer_embeddings)

        # Aggregate confidence scores by cluster
        cluster_weights = defaultdict(float)
        cluster_answers = defaultdict(list)
        cluster_confidences = defaultdict(list)

        for answer, confidence, cluster_id in zip(answers, confidence_scores, clusters):
            cluster_weights[cluster_id] += confidence
            cluster_answers[cluster_id].append(answer)
            cluster_confidences[cluster_id].append(confidence)

        # Select representative answer for each cluster (highest confidence)
        result = {}
        for cluster_id, total_weight in cluster_weights.items():
            # Find answer with highest individual confidence in this cluster
            cluster_ans = cluster_answers[cluster_id]
            cluster_conf = cluster_confidences[cluster_id]
            max_idx = np.argmax(cluster_conf)
            representative_answer = cluster_ans[max_idx]

            result[representative_answer] = total_weight

        # Normalize by total sum
        if self.normalize:
            total_weight = sum(result.values())
            if total_weight > 0:
                result = {
                    ans: weight / total_weight
                    for ans, weight in result.items()
                }

        return result

    def _compute_embeddings(self, answers: List[str]) -> torch.Tensor:
        """
        Compute embeddings for answers using the embedding model.

        Args:
            answers: List of answer strings

        Returns:
            Tensor of embeddings (num_answers, embed_dim)
        """
        # This is a placeholder - actual implementation depends on embedding model
        embeddings = []

        with torch.no_grad():
            for answer in answers:
                embedding = self.embedding_model(answer)
                embeddings.append(embedding)

        return torch.stack(embeddings)

    def _cluster_answers(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Cluster answer embeddings based on similarity.

        Args:
            embeddings: Answer embeddings (num_answers, embed_dim)

        Returns:
            Array of cluster labels (one per answer)
        """
        if len(embeddings) == 1:
            return np.array([0])

        # Convert to numpy for sklearn
        embeddings_np = embeddings.cpu().numpy()

        if self.clustering_method == "dbscan":
            # DBSCAN clustering with cosine distance
            # eps controls the similarity threshold
            eps = 1.0 - self.similarity_threshold
            clustering = DBSCAN(
                eps=eps,
                min_samples=self.min_cluster_size,
                metric='cosine'
            )
            labels = clustering.fit_predict(embeddings_np)

        elif self.clustering_method == "agglomerative":
            # Agglomerative clustering with cosine affinity
            n_clusters = None
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                distance_threshold=1.0 - self.similarity_threshold,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings_np)

        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        return labels

    def aggregate(
        self,
        answers: List[str],
        confidence_scores: Union[List[float], torch.Tensor],
        answer_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Aggregate answers based on the configured mode.

        Args:
            answers: List of answer strings
            confidence_scores: List or tensor of confidence scores
            answer_embeddings: Optional pre-computed embeddings (semantic mode only)

        Returns:
            Dictionary mapping answers to aggregated confidence scores
        """
        # Convert confidence scores to list if tensor
        if isinstance(confidence_scores, torch.Tensor):
            confidence_scores = confidence_scores.cpu().tolist()

        if self.mode == "exact":
            return self.aggregate_exact(answers, confidence_scores)
        elif self.mode == "semantic":
            return self.aggregate_semantic(answers, confidence_scores, answer_embeddings)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_best_answer(
        self,
        answers: List[str],
        confidence_scores: Union[List[float], torch.Tensor],
        answer_embeddings: Optional[torch.Tensor] = None,
        return_confidence: bool = True
    ) -> Union[str, Tuple[str, float]]:
        """
        Get the best answer based on aggregated confidence.

        Args:
            answers: List of answer strings
            confidence_scores: List or tensor of confidence scores
            answer_embeddings: Optional pre-computed embeddings
            return_confidence: Whether to return confidence score along with answer

        Returns:
            Best answer string, or tuple of (answer, confidence) if return_confidence=True
        """
        aggregated = self.aggregate(answers, confidence_scores, answer_embeddings)

        if not aggregated:
            return ("", 0.0) if return_confidence else ""

        # Get answer with highest aggregated confidence
        best_answer = max(aggregated.items(), key=lambda x: x[1])

        if return_confidence:
            return best_answer  # (answer, confidence)
        else:
            return best_answer[0]  # just answer


class SelfConsistencyAggregator(WeightedFrequency):
    """
    Self-consistency aggregator that extends WeightedFrequency.

    This version can optionally weight all answers equally (uniform weighting)
    which corresponds to standard self-consistency, or use confidence weighting.
    """

    def __init__(
        self,
        use_confidence_weighting: bool = True,
        **kwargs
    ):
        """
        Initialize self-consistency aggregator.

        Args:
            use_confidence_weighting: If True, use confidence scores as weights.
                                     If False, use uniform weighting (standard self-consistency).
            **kwargs: Additional arguments passed to WeightedFrequency
        """
        super().__init__(**kwargs)
        self.use_confidence_weighting = use_confidence_weighting

    def aggregate(
        self,
        answers: List[str],
        confidence_scores: Optional[Union[List[float], torch.Tensor]] = None,
        answer_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Aggregate answers with optional confidence weighting.

        Args:
            answers: List of answer strings
            confidence_scores: Optional confidence scores (ignored if use_confidence_weighting=False)
            answer_embeddings: Optional pre-computed embeddings

        Returns:
            Dictionary mapping answers to aggregated scores
        """
        if not self.use_confidence_weighting or confidence_scores is None:
            # Uniform weighting (standard self-consistency)
            confidence_scores = [1.0] * len(answers)
        elif isinstance(confidence_scores, torch.Tensor):
            confidence_scores = confidence_scores.cpu().tolist()

        return super().aggregate(answers, confidence_scores, answer_embeddings)


class EnsembleAggregator:
    """
    Ensemble multiple aggregation strategies.

    Combines results from different aggregation methods (e.g., exact + semantic)
    to produce a more robust final answer.
    """

    def __init__(
        self,
        aggregators: List[WeightedFrequency],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble aggregator.

        Args:
            aggregators: List of aggregator instances
            weights: Optional weights for each aggregator (uniform if None)
        """
        self.aggregators = aggregators

        if weights is None:
            self.weights = [1.0 / len(aggregators)] * len(aggregators)
        else:
            if len(weights) != len(aggregators):
                raise ValueError("Number of weights must match number of aggregators")
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def aggregate(
        self,
        answers: List[str],
        confidence_scores: Union[List[float], torch.Tensor],
        answer_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Aggregate using ensemble of methods.

        Args:
            answers: List of answer strings
            confidence_scores: Confidence scores
            answer_embeddings: Optional pre-computed embeddings

        Returns:
            Combined aggregated scores
        """
        # Get results from each aggregator
        all_results = []
        for aggregator in self.aggregators:
            result = aggregator.aggregate(answers, confidence_scores, answer_embeddings)
            all_results.append(result)

        # Combine results with weighted averaging
        combined = defaultdict(float)

        for result, weight in zip(all_results, self.weights):
            for answer, score in result.items():
                combined[answer] += weight * score

        return dict(combined)

    def get_best_answer(
        self,
        answers: List[str],
        confidence_scores: Union[List[float], torch.Tensor],
        answer_embeddings: Optional[torch.Tensor] = None,
        return_confidence: bool = True
    ) -> Union[str, Tuple[str, float]]:
        """
        Get the best answer from ensemble.

        Args:
            answers: List of answer strings
            confidence_scores: Confidence scores
            answer_embeddings: Optional pre-computed embeddings
            return_confidence: Whether to return confidence score

        Returns:
            Best answer and optionally its confidence
        """
        aggregated = self.aggregate(answers, confidence_scores, answer_embeddings)

        if not aggregated:
            return ("", 0.0) if return_confidence else ""

        best_answer = max(aggregated.items(), key=lambda x: x[1])

        if return_confidence:
            return best_answer
        else:
            return best_answer[0]
