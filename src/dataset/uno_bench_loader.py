"""
UNO-Bench data loader for multimodal reasoning tasks.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


@dataclass
class UNOBenchSample:
    """Single sample from UNO-Bench dataset."""

    id: str
    question: str
    answer: str
    images: List[Image.Image]
    audio_paths: Optional[List[str]] = None  # Paths to audio files
    audio_data: Optional[List[np.ndarray]] = None  # Loaded audio waveforms
    modality: str = 'omni-modal'  # 'uni-modal', 'omni-modal', 'audio', etc.
    reasoning_type: str = 'unknown'  # e.g., 'logical', 'mathematical', 'spatial', 'auditory'
    metadata: Optional[Dict] = None


class UNOBenchLoader:
    """
    Loader for UNO-Bench dataset with support for both uni-modal and omni-modal tasks.

    UNO-Bench provides human-curated multimodal reasoning tasks with diverse
    reasoning types and automatic scoring capability.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        modality_filter: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize UNO-Bench loader.

        Args:
            data_path: Path to UNO-Bench dataset
            split: Dataset split ('train', 'val', 'test')
            modality_filter: Filter by modality ('uni-modal', 'omni-modal', None for all)
            cache_dir: Directory for caching processed data
        """
        self.data_path = Path(data_path)
        self.split = split
        self.modality_filter = modality_filter
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.samples = self._load_samples()

    def _load_samples(self) -> List[UNOBenchSample]:
        """Load samples from dataset."""
        json_path = self.data_path / f"{self.split}.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"UNO-Bench {self.split} split not found at {json_path}. "
                "Please download the dataset from the official repository."
            )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for item in data:
            # Apply modality filter
            if self.modality_filter and item.get('modality') != self.modality_filter:
                continue

            # Load images
            images = []
            for img_path in item.get('image_paths', []):
                full_path = self.data_path / img_path
                if full_path.exists():
                    images.append(Image.open(full_path).convert('RGB'))

            # Load audio paths (not loading audio data yet for efficiency)
            audio_paths = []
            for audio_path in item.get('audio_paths', []):
                full_path = self.data_path / audio_path
                if full_path.exists():
                    audio_paths.append(str(full_path))

            sample = UNOBenchSample(
                id=item['id'],
                question=item['question'],
                answer=item['answer'],
                images=images,
                audio_paths=audio_paths if audio_paths else None,
                audio_data=None,  # Lazy loading - load when needed
                modality=item.get('modality', 'omni-modal'),
                reasoning_type=item.get('reasoning_type', 'unknown'),
                metadata=item.get('metadata', {})
            )
            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UNOBenchSample:
        return self.samples[idx]

    def get_by_reasoning_type(self, reasoning_type: str) -> List[UNOBenchSample]:
        """Get all samples of a specific reasoning type."""
        return [s for s in self.samples if s.reasoning_type == reasoning_type]

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.samples),
            'modality_distribution': {},
            'reasoning_type_distribution': {},
            'images_per_sample': []
        }

        for sample in self.samples:
            # Modality distribution
            stats['modality_distribution'][sample.modality] = \
                stats['modality_distribution'].get(sample.modality, 0) + 1

            # Reasoning type distribution
            stats['reasoning_type_distribution'][sample.reasoning_type] = \
                stats['reasoning_type_distribution'].get(sample.reasoning_type, 0) + 1

            # Images per sample
            stats['images_per_sample'].append(len(sample.images))

        return stats


class UNOBenchDataset(Dataset):
    """PyTorch Dataset wrapper for UNO-Bench."""

    def __init__(
        self,
        loader: UNOBenchLoader,
        transform=None
    ):
        """
        Initialize dataset.

        Args:
            loader: UNOBenchLoader instance
            transform: Optional image transforms
        """
        self.loader = loader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.loader[idx]

        # Apply transforms to images
        images = sample.images
        if self.transform:
            images = [self.transform(img) for img in images]

        return {
            'id': sample.id,
            'question': sample.question,
            'answer': sample.answer,
            'images': images,
            'modality': sample.modality,
            'reasoning_type': sample.reasoning_type,
            'metadata': sample.metadata
        }
