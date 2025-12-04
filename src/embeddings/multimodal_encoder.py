"""
Multimodal encoder for joint text-image-audio embeddings.
"""

from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, Wav2Vec2Model, Wav2Vec2Processor
import open_clip
import numpy as np


class MultimodalEncoder(nn.Module):
    """
    Encode text and images into aligned multimodal embeddings.

    Supports:
    - CLIP (OpenAI and open_clip variants)
    - LVLM internal multimodal projectors
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        use_open_clip: bool = False
    ):
        """
        Initialize multimodal encoder.

        Args:
            model_name: Model name or path
            device: Device to run on
            use_open_clip: Use open_clip library instead of HuggingFace
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.use_open_clip = use_open_clip

        if use_open_clip:
            self._load_open_clip()
        else:
            self._load_huggingface_clip()

    def _load_huggingface_clip(self):
        """Load CLIP from HuggingFace."""
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.embedding_dim = self.model.config.projection_dim

    def _load_open_clip(self):
        """Load CLIP from open_clip."""
        model_name = self.model_name.split('/')[-1]
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            model_name,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embedding_dim = self.model.visual.output_dim

    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings

        Returns:
            Text embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False

        if self.use_open_clip:
            tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
        else:
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

        if normalize:
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        if return_single:
            return text_features[0]

        return text_features

    def encode_images(
        self,
        images: Union[Image.Image, List[Image.Image]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: Single image or list of images
            normalize: Whether to normalize embeddings

        Returns:
            Image embeddings
        """
        if isinstance(images, Image.Image):
            images = [images]
            return_single = True
        else:
            return_single = False

        if self.use_open_clip:
            image_tensors = torch.stack([
                self.processor(img).to(self.device) for img in images
            ])
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensors)
        else:
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

        if normalize:
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

        if return_single:
            return image_features[0]

        return image_features

    def encode_multimodal(
        self,
        texts: List[str],
        images: List[Image.Image],
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both text and images.

        Args:
            texts: List of texts
            images: List of images
            normalize: Whether to normalize

        Returns:
            Tuple of (text_embeddings, image_embeddings)
        """
        text_features = self.encode_text(texts, normalize)
        image_features = self.encode_images(images, normalize)

        return text_features, image_features

    def compute_similarity(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between text and image embeddings.

        Args:
            text_embeds: Text embeddings (batch_size, embed_dim)
            image_embeds: Image embeddings (batch_size, embed_dim)

        Returns:
            Similarity scores
        """
        # Cosine similarity (embeddings should be normalized)
        similarity = torch.matmul(text_embeds, image_embeds.T)

        return similarity

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim

    def forward(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Image.Image]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for encoding.

        Args:
            texts: Optional list of texts
            images: Optional list of images

        Returns:
            Embeddings based on what's provided
        """
        if texts is not None and images is not None:
            return self.encode_multimodal(texts, images)
        elif texts is not None:
            return self.encode_text(texts)
        elif images is not None:
            return self.encode_images(images)
        else:
            raise ValueError("Must provide either texts or images")


class AudioEncoder(nn.Module):
    """
    Encode audio into embeddings aligned with text.

    Supports:
    - Wav2Vec2 for audio feature extraction
    - CLAP (Contrastive Language-Audio Pretraining) for text-audio alignment
    - Custom audio transformers
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "cuda",
        use_clap: bool = False,
        target_sample_rate: int = 16000
    ):
        """
        Initialize audio encoder.

        Args:
            model_name: Model name or path (Wav2Vec2 or CLAP model)
            device: Device to run on
            use_clap: Use CLAP for text-audio alignment instead of Wav2Vec2
            target_sample_rate: Target sample rate for audio processing
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.use_clap = use_clap
        self.target_sample_rate = target_sample_rate

        if use_clap:
            self._load_clap()
        else:
            self._load_wav2vec2()

    def _load_wav2vec2(self):
        """Load Wav2Vec2 model from HuggingFace."""
        self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)

        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size

    def _load_clap(self):
        """Load CLAP model for audio-text alignment."""
        try:
            # Try to load CLAP from laion_clap library
            import laion_clap
            self.model = laion_clap.CLAP_Module(enable_fusion=False, device=self.device)
            self.model.load_ckpt(self.model_name)
            self.embedding_dim = 512  # CLAP default embedding dimension
            self.processor = None
        except ImportError:
            raise ImportError(
                "CLAP support requires laion_clap library. "
                "Install with: pip install laion-clap"
            )

    def encode_audio(
        self,
        audio_data: Union[np.ndarray, List[np.ndarray]],
        sample_rate: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode audio to embeddings.

        Args:
            audio_data: Audio waveform(s) as numpy array(s)
            sample_rate: Sample rate of input audio (will resample if different from target)
            normalize: Whether to normalize embeddings

        Returns:
            Audio embeddings
        """
        if isinstance(audio_data, np.ndarray) or isinstance(audio_data, torch.Tensor):
            audio_data = [audio_data]
            return_single = True
        else:
            return_single = False

        # Resample if necessary
        if sample_rate is not None and sample_rate != self.target_sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, self.target_sample_rate)

        if self.use_clap:
            # CLAP encoding
            if isinstance(audio_data[0], np.ndarray):
                audio_data = [torch.from_numpy(x) for x in audio_data]
            audio_features = self.model.get_audio_embedding_from_data(
                x=audio_data,
                use_tensor=True
            )
            audio_features = audio_features.to(self.device)
        else:
            # Wav2Vec2 encoding
            inputs = self.processor(
                audio_data,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over sequence dimension
                audio_features = outputs.last_hidden_state.mean(dim=1)

        if normalize:
            audio_features = torch.nn.functional.normalize(audio_features, p=2, dim=-1)

        if return_single:
            return audio_features[0]

        return audio_features

    def encode_audio_from_file(
        self,
        audio_paths: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode audio from file path(s).

        Args:
            audio_paths: Single audio file path or list of paths
            normalize: Whether to normalize embeddings

        Returns:
            Audio embeddings
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
            return_single = True
        else:
            return_single = False

        # Load audio files
        audio_data = []
        for path in audio_paths:
            audio, sr = self._load_audio_file(path)
            audio_data.append(audio)

        # Get sample rate from first file
        sample_rate = sr

        # Encode
        audio_features = self.encode_audio(audio_data, sample_rate, normalize)

        if return_single:
            return audio_features[0] if not return_single else audio_features

        return audio_features

    def _load_audio_file(self, path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file from path.

        Args:
            path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            import librosa
            audio, sr = librosa.load(path, sr=None)
            return audio, sr
        except ImportError:
            raise ImportError(
                "Audio file loading requires librosa. "
                "Install with: pip install librosa"
            )

    def _resample_audio(
        self,
        audio_data: List[np.ndarray],
        orig_sr: int,
        target_sr: int
    ) -> List[np.ndarray]:
        """
        Resample audio to target sample rate.

        Args:
            audio_data: List of audio arrays
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio data
        """
        try:
            import librosa
            resampled = [
                librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
                for audio in audio_data
            ]
            return resampled
        except ImportError:
            raise ImportError(
                "Audio resampling requires librosa. "
                "Install with: pip install librosa"
            )

    def encode_text_for_audio_alignment(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text for alignment with audio (only works with CLAP).

        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings

        Returns:
            Text embeddings aligned with audio space
        """
        if not self.use_clap:
            raise ValueError(
                "Text encoding for audio alignment requires CLAP model. "
                "Set use_clap=True when initializing AudioEncoder."
            )

        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False

        # Encode text with CLAP
        text_features = self.model.get_text_embedding(texts, use_tensor=True)
        text_features = text_features.to(self.device)

        if normalize:
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        if return_single:
            return text_features[0]

        return text_features

    def compute_audio_text_similarity(
        self,
        audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between audio and text embeddings.

        Args:
            audio_embeds: Audio embeddings (batch_size, embed_dim)
            text_embeds: Text embeddings (batch_size, embed_dim)

        Returns:
            Similarity scores
        """
        # Cosine similarity (embeddings should be normalized)
        similarity = torch.matmul(audio_embeds, text_embeds.T)
        return similarity

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim

    def forward(
        self,
        audio_data: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        texts: Optional[Union[str, List[str]]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for encoding.

        Args:
            audio_data: Optional audio waveform(s)
            texts: Optional text(s) (only works with CLAP)

        Returns:
            Embeddings based on what's provided
        """
        if audio_data is not None and texts is not None:
            if not self.use_clap:
                raise ValueError("Joint audio-text encoding requires CLAP model")
            audio_embeds = self.encode_audio(audio_data)
            text_embeds = self.encode_text_for_audio_alignment(texts)
            return audio_embeds, text_embeds
        elif audio_data is not None:
            return self.encode_audio(audio_data)
        elif texts is not None:
            return self.encode_text_for_audio_alignment(texts)
        else:
            raise ValueError("Must provide either audio_data or texts")
