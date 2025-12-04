"""Embedding extraction modules for text and multimodal inputs."""

from .text_encoder import TextEncoder
from .multimodal_encoder import MultimodalEncoder, AudioEncoder
from .embedding_utils import EmbeddingCache, compute_similarity

__all__ = [
    "TextEncoder",
    "MultimodalEncoder",
    "AudioEncoder",
    "EmbeddingCache",
    "compute_similarity",
]
