"""
Data processing utilities for preparing CoT chains for coherence analysis.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from .uno_bench_loader import UNOBenchSample
from .cot_generator import CoTChain


class CoTDataset(Dataset):
    """
    Dataset of CoT chains with labels for training confidence models.
    """

    def __init__(
        self,
        samples: List[UNOBenchSample],
        cot_chains: List[List[CoTChain]],
        labels: List[int],
        embeddings: Optional[Dict] = None
    ):
        """
        Initialize dataset.

        Args:
            samples: Original UNO-Bench samples
            cot_chains: Generated CoT chains (list of lists, multiple chains per sample)
            labels: Binary labels (1 = correct, 0 = incorrect)
            embeddings: Pre-computed embeddings (optional)
        """
        self.samples = samples
        self.cot_chains = cot_chains
        self.labels = labels
        self.embeddings = embeddings or {}

        # Flatten to individual chains
        self.flat_data = []
        for sample, chains, label in zip(samples, cot_chains, labels):
            for chain in chains:
                self.flat_data.append({
                    'sample': sample,
                    'chain': chain,
                    'label': label
                })

    def __len__(self) -> int:
        return len(self.flat_data)

    def __getitem__(self, idx: int) -> Dict:
        data = self.flat_data[idx]

        item = {
            'sample_id': data['sample'].id,
            'question': data['sample'].question,
            'images': data['sample'].images,
            'audio_paths': data['sample'].audio_paths,
            'audio_data': data['sample'].audio_data,
            'chain_text': data['chain'].text,
            'chain_steps': data['chain'].steps,
            'final_answer': data['chain'].final_answer,
            'gold_answer': data['sample'].answer,
            'label': data['label']
        }

        # Add embeddings if available
        chain_id = f"{data['sample'].id}_{idx}"
        if chain_id in self.embeddings:
            item['embeddings'] = self.embeddings[chain_id]

        return item


class DataProcessor:
    """
    Process raw samples and CoT chains into training-ready format.
    """

    def __init__(self, scoring_model=None):
        """
        Initialize processor.

        Args:
            scoring_model: UNO-Bench scoring model for automatic labeling
        """
        self.scoring_model = scoring_model

    def process_samples(
        self,
        samples: List[UNOBenchSample],
        cot_chains: List[List[CoTChain]],
        use_scoring_model: bool = True
    ) -> Tuple[List[UNOBenchSample], List[List[CoTChain]], List[int]]:
        """
        Process samples and assign labels.

        Args:
            samples: UNO-Bench samples
            cot_chains: Generated CoT chains
            use_scoring_model: Whether to use automatic scoring model

        Returns:
            Tuple of (samples, cot_chains, labels)
        """
        labels = []

        for sample, chains in zip(samples, cot_chains):
            # Score each chain
            chain_labels = []
            for chain in chains:
                if use_scoring_model and self.scoring_model:
                    label = self._score_with_model(sample, chain)
                else:
                    label = self._exact_match_score(sample, chain)
                chain_labels.append(label)

            # For now, use the same label for all chains from the same sample
            # In practice, each chain might have different correctness
            labels.append(chain_labels[0] if chain_labels else 0)

        return samples, cot_chains, labels

    def _score_with_model(self, sample: UNOBenchSample, chain: CoTChain) -> int:
        """Score using UNO-Bench scoring model."""
        if self.scoring_model is None:
            return self._exact_match_score(sample, chain)

        # Placeholder for actual scoring model call
        # The UNO-Bench paper mentions an automatic scoring model
        # Implementation depends on the specific model provided
        score = self.scoring_model.score(
            question=sample.question,
            predicted_answer=chain.final_answer,
            gold_answer=sample.answer,
            reasoning=chain.text
        )

        # Convert score to binary label
        return 1 if score > 0.5 else 0

    def _exact_match_score(self, sample: UNOBenchSample, chain: CoTChain) -> int:
        """Simple exact match scoring."""
        # Normalize answers for comparison
        pred = self._normalize_answer(chain.final_answer)
        gold = self._normalize_answer(sample.answer)

        return 1 if pred == gold else 0

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Normalize answer for comparison."""
        import re

        # Convert to lowercase
        answer = answer.lower()

        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)

        # Remove extra whitespace
        answer = ' '.join(answer.split())

        return answer

    def create_train_val_split(
        self,
        dataset: CoTDataset,
        val_split: float = 0.2,
        seed: int = 42
    ) -> Tuple[CoTDataset, CoTDataset]:
        """
        Split dataset into train and validation sets.

        Args:
            dataset: Full dataset
            val_split: Fraction for validation
            seed: Random seed

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        np.random.seed(seed)

        n = len(dataset)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - val_split))

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_data = [dataset.flat_data[i] for i in train_indices]
        val_data = [dataset.flat_data[i] for i in val_indices]

        # Create new datasets
        train_dataset = CoTDataset(
            samples=[d['sample'] for d in train_data],
            cot_chains=[[d['chain']] for d in train_data],
            labels=[d['label'] for d in train_data],
            embeddings=dataset.embeddings
        )

        val_dataset = CoTDataset(
            samples=[d['sample'] for d in val_data],
            cot_chains=[[d['chain']] for d in val_data],
            labels=[d['label'] for d in val_data],
            embeddings=dataset.embeddings
        )

        return train_dataset, val_dataset

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate function for DataLoader.

        Args:
            batch: List of dataset items

        Returns:
            Batched dictionary
        """
        return {
            'sample_ids': [item['sample_id'] for item in batch],
            'questions': [item['question'] for item in batch],
            'images': [item['images'] for item in batch],
            'audio_paths': [item['audio_paths'] for item in batch],
            'audio_data': [item['audio_data'] for item in batch],
            'chain_texts': [item['chain_text'] for item in batch],
            'chain_steps': [item['chain_steps'] for item in batch],
            'final_answers': [item['final_answer'] for item in batch],
            'gold_answers': [item['gold_answer'] for item in batch],
            'labels': torch.tensor([item['label'] for item in batch], dtype=torch.float32)
        }
