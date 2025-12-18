import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd


@dataclass
class UNOBenchSample:
    """Single sample from UNO-Bench dataset."""

    id: str
    question: str
    answer: str
    images: Optional[List[Image.Image]] = None  # PIL images
    image_paths: Optional[List[str]] = None  # Paths to image files
    audio_paths: Optional[List[str]] = None  # Paths to audio files
    audio_data: Optional[List[np.ndarray]] = None  # Loaded audio waveforms
    modality: str = "omni-modal"  # 'uni-modal', 'omni-modal', 'audio', etc.
    reasoning_type: str = (
        "unknown"  # e.g., 'logical', 'mathematical', 'spatial', 'auditory'
    )
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
        split: str = "validation",
        modality_filter: Optional[str] = None,
        cache_dir: Optional[str] = None,
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
        json_path = self.data_path / f"{self.split}.parquet"

        if not json_path.exists():
            raise FileNotFoundError(
                f"UNO-Bench {self.split} split not found at {json_path}. "
                "Please download the dataset from the official repository."
            )

        data = pd.read_parquet(json_path)

        # Apply modality filter
        if self.modality_filter:
            data = data[data.subset_name == self.modality_filter]
            assert len(
                data
            ), f"Modality filter {self.modality_filter} is not a valid subset name in UNOBench"

        samples = []
        for idx in range(len(data)):
            item = data.iloc[idx]
            # Load images
            images = []
            image_paths = []
            for img_path in item.get("images", []):
                if item["images"][img_path] is None:
                    continue
                full_path = self.data_path / item["images"][img_path]
                image_paths.append(full_path)
                if full_path.exists():
                    images.append(Image.open(full_path).convert("RGB"))

            # Load audio paths (not loading audio data yet for efficiency)
            audio_paths = []
            for audio_path in item.get("audios", []):
                if item["audios"][audio_path] is None:
                    continue
                full_path = self.data_path / item["audios"][audio_path]
                if full_path.exists():
                    audio_paths.append(str(full_path))

            sample = UNOBenchSample(
                id=item["qid"],
                question=item["question"],
                answer=item["answer"],
                images=images,
                image_paths=image_paths,
                audio_paths=audio_paths if audio_paths else None,
                audio_data=None,  # Lazy loading - load when needed
                modality=item.get("subset_name", "omni-modal"),
                reasoning_type=item.get("task", "unknown"),
                metadata={
                    "source": item.get("source", "unknown"),
                    "score_type": item.get("score_type", "unknown"),
                    "audio_types": item.get("audio_types", "unknown"),
                    "ability": item.get("ability", "unknown"),
                },
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
            "total_samples": len(self.samples),
            "modality_distribution": {},
            "reasoning_type_distribution": {},
            "images_per_sample": [],
        }

        for sample in self.samples:
            # Modality distribution
            stats["modality_distribution"][sample.modality] = (
                stats["modality_distribution"].get(sample.modality, 0) + 1
            )

            # Reasoning type distribution
            stats["reasoning_type_distribution"][sample.reasoning_type] = (
                stats["reasoning_type_distribution"].get(sample.reasoning_type, 0) + 1
            )

            # Images per sample
            stats["images_per_sample"].append(len(sample.images))

        return stats


class UNOBenchDataset(Dataset):
    """PyTorch Dataset wrapper for UNO-Bench."""

    def __init__(self, loader: UNOBenchLoader, transform=None):
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
            "id": sample.id,
            "question": sample.question,
            "answer": sample.answer,
            "image": images,
            "image_paths": sample.image_paths,
            "audio_paths": sample.audio_paths,
            "modality": sample.modality,
            "reasoning_type": sample.reasoning_type,
            "metadata": sample.metadata,
        }
