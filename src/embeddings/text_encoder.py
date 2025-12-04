"""
Text encoder for extracting embeddings from CoT steps and questions.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    Encode text (questions, CoT steps) into dense embeddings.

    Supports multiple encoder backends:
    - Sentence Transformers (recommended for general use)
    - BERT-style models
    - Custom LVLM hidden states
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda",
        use_sentence_transformers: bool = True
    ):
        """
        Initialize text encoder.

        Args:
            model_name: Model name or path
            device: Device to run on
            use_sentence_transformers: Use SentenceTransformer library
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.use_sentence_transformers = use_sentence_transformers

        if use_sentence_transformers:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_dim = self.model.config.hidden_size

    def forward(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize embeddings

        Returns:
            Tensor of shape (batch_size, embedding_dim) or (embedding_dim,)
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False

        if self.use_sentence_transformers:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
        else:
            embeddings = self._encode_with_transformer(texts, normalize)

        if return_single:
            return embeddings[0]

        return embeddings

    def _encode_with_transformer(
        self,
        texts: List[str],
        normalize: bool
    ) -> torch.Tensor:
        """Encode using standard transformer model."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use [CLS] token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def encode_cot_steps(
        self,
        steps: List[str],
        question: Optional[str] = None,
        prepend_question: bool = False
    ) -> torch.Tensor:
        """
        Encode CoT steps, optionally prepending question context.

        Args:
            steps: List of reasoning steps
            question: Original question (optional)
            prepend_question: Whether to prepend question to each step

        Returns:
            Tensor of shape (num_steps, embedding_dim)
        """
        if prepend_question and question:
            # Prepend question to give context
            texts = [f"{question} {step}" for step in steps]
        else:
            texts = steps

        return self.forward(texts)

    def encode_with_pooling(
        self,
        texts: List[str],
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Encode texts and apply pooling to get single vector.

        Args:
            texts: List of texts
            pooling: Pooling method ('mean', 'max', 'cls')

        Returns:
            Single pooled embedding
        """
        embeddings = self.forward(texts)

        if pooling == "mean":
            return embeddings.mean(dim=0)
        elif pooling == "max":
            return embeddings.max(dim=0)[0]
        elif pooling == "first" or pooling == "cls":
            return embeddings[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim
