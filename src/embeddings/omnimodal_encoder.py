"""
Omnimodal encoder for joint text-image-video-audio embeddings.

Uses Qwen2.5-Omni as the backbone, which natively processes all modalities
in a single forward pass — no separate projectors or alignment losses needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Type aliases (mirror conventions from the rest of the codebase)
# ---------------------------------------------------------------------------

# Raw audio: 1-D float32 numpy array at any sample rate
AudioArray = np.ndarray

# A single multimodal "document": a list of content dicts that can mix
# text, image, video, and audio entries in any order.
#
# Example:
#   [
#       {"type": "text",  "text": "Describe this clip."},
#       {"type": "video", "video": "/path/to/clip.mp4"},
#       {"type": "audio", "audio": "/path/to/sound.wav"},
#   ]
ContentList = List[Dict]


class OmnimodalEncoder(nn.Module):
    """
    Encode arbitrary combinations of text, images, video, and audio into a
    single embedding per input document.

    Backbone: Qwen2.5-Omni-Thinker (LCO-Embedding/LCO-Embedding-Omni-7B or
    any compatible checkpoint).  All modalities are processed jointly by the
    same transformer — there is no late-fusion step.

    The embedding is the hidden state of the *last token* of the last layer,
    following the usage shown in the reference notebook.

    Typical usage
    -------------
    >>> encoder = OmnimodalEncoder()
    >>>
    >>> # Pure text
    >>> emb = encoder.encode([{"type": "text", "text": "passage: hello world"}])
    >>>
    >>> # Text + video
    >>> emb = encoder.encode([
    ...     {"type": "text",  "text": "passage: a person drawing"},
    ...     {"type": "video", "video": "draw.mp4"},
    ... ])
    >>>
    >>> # Batch of documents (each document is its own content list)
    >>> embs = encoder.encode_batch([
    ...     [{"type": "text", "text": "passage: doc one"}],
    ...     [{"type": "text", "text": "passage: doc two"},
    ...      {"type": "image", "image": "/path/img.jpg"}],
    ... ])
    """

    # Prefix that the reference model was trained with for passage encoding.
    # Override via the `prefix` argument to __init__ if needed.
    DEFAULT_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str = "LCO-Embedding/LCO-Embedding-Omni-7B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_pixels: Optional[int] = None,
        use_audio_in_video: bool = True,
        prefix: str = DEFAULT_PREFIX,
        device_map: Optional[str] = "auto",
    ):
        """
        Initialise the omnimodal encoder.

        Args:
            model_name:        HuggingFace repo or local path for the
                               Qwen2.5-Omni-compatible checkpoint.
            device:            Target device when *not* using device_map
                               (e.g. "cuda", "cpu").  Ignored when
                               device_map is set.
            torch_dtype:       Weight dtype. bfloat16 is the default used
                               in the reference notebook and recommended for
                               modern GPUs.
            max_pixels:        Optional pixel cap for image/video inputs
                               (e.g. 1280*28*28).  Passed straight to the
                               processor.  Reduces VRAM at the cost of
                               resolution.
            use_audio_in_video: Whether to extract and encode the audio
                               track embedded in video files.  Mirrors the
                               `use_audio_in_video` flag in process_mm_info.
            prefix:            Text prefix prepended to the *first* text
                               chunk of every document (default "passage: ").
                               Pass an empty string to disable.
            device_map:        Passed to from_pretrained for multi-GPU /
                               CPU-offload layouts.  Set to None to use
                               the `device` argument instead.
        """
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_pixels = max_pixels
        self.use_audio_in_video = use_audio_in_video
        self.prefix = prefix
        self.device_map = device_map

        self._load_model()

    # ------------------------------------------------------------------
    # Private: model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Qwen2.5-Omni processor and model weights."""
        try:
            from transformers import (
                Qwen2_5OmniProcessor,
                Qwen2_5OmniThinkerForConditionalGeneration,
            )
        except ImportError:
            raise ImportError(
                "OmnimodalEncoder requires the Qwen2.5-Omni transformers "
                "integration.  Install with:\n"
                "  pip install transformers>=4.51.0"
            )

        try:
            from qwen_omni_utils import process_mm_info  # noqa: F401 — validate early
        except ImportError:
            raise ImportError(
                "OmnimodalEncoder requires qwen-omni-utils.  Install with:\n"
                "  pip install qwen-omni-utils"
            )

        # Processor -------------------------------------------------------
        processor_kwargs = {}
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_name, **processor_kwargs
        )

        # Model -----------------------------------------------------------
        model_kwargs: Dict = {"torch_dtype": self.torch_dtype}
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        else:
            model_kwargs["device_map"] = None

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_name, **model_kwargs
        )

        if self.device_map is None:
            self.model = self.model.to(self.device)

        # Derive embedding dimension from model config --------------------
        self.embedding_dim: int = self.model.config.hidden_size

    # ------------------------------------------------------------------
    # Private: document preparation
    # ------------------------------------------------------------------

    def _apply_prefix(self, content: ContentList) -> ContentList:
        """
        Prepend self.prefix to the first text entry in a content list.

        If there is no text entry and a prefix is set, a text entry is
        inserted at position 0.  This mirrors how the reference notebook
        prepends "passage: " to every document.
        """
        if not self.prefix:
            return content

        content = list(content)  # shallow copy — don't mutate caller's list

        for i, item in enumerate(content):
            if item.get("type") == "text":
                content[i] = {**item, "text": self.prefix + item["text"]}
                return content

        # No text entry found — insert one at the front
        content.insert(0, {"type": "text", "text": self.prefix})
        return content

    def _build_conversation(self, content: ContentList) -> List[Dict]:
        """
        Wrap a content list in the single-turn user-message format that
        apply_chat_template expects.
        """
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Private: forward through the model
    # ------------------------------------------------------------------

    def _forward_conversations(
        self,
        conversations: List[List[Dict]],
        normalize: bool,
    ) -> torch.Tensor:
        """
        Run a batch of conversations through the model and return embeddings.

        Args:
            conversations: List of single-turn conversation dicts, one per
                           document.
            normalize:     Whether to L2-normalise the output embeddings.

        Returns:
            Tensor of shape (batch_size, embedding_dim).
        """
        from qwen_omni_utils import process_mm_info

        # 1. Render chat templates ----------------------------------------
        texts = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2. Extract multimodal side-inputs --------------------------------
        audio_inputs, image_inputs, video_inputs = process_mm_info(
            conversations,
            use_audio_in_video=self.use_audio_in_video,
        )

        # 3. Tokenise + build pixel/audio tensors --------------------------
        inputs = self.processor(
            text=texts,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        target_device = (
            next(self.model.parameters()).device
            if self.device_map is not None
            else torch.device(self.device)
        )
        inputs = inputs.to(target_device)

        # 4. Model forward — extract last-layer last-token hidden state ----
        with torch.no_grad():
            hidden_states = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[
                -1
            ]  # (batch, seq_len, hidden)

        # Last token of each sequence in the batch
        embeddings = hidden_states[:, -1, :]  # (batch, hidden)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    # ------------------------------------------------------------------
    # Public: single-document encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        content: ContentList,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode a single multimodal document.

        A document is a flat list of content dicts.  Each dict must have a
        "type" key set to one of "text", "image", "video", or "audio", plus
        the corresponding payload key:

            {"type": "text",  "text": "some string"}
            {"type": "image", "image": "/path/to/image.jpg"}
            {"type": "video", "video": "/path/to/clip.mp4"}
            {"type": "audio", "audio": "/path/to/sound.wav"}

        Video URLs are also accepted wherever local paths are.

        Args:
            content:   List of content dicts (the multimodal document).
            normalize: Whether to L2-normalise the embedding.

        Returns:
            1-D tensor of shape (embedding_dim,).
        """
        content = self._apply_prefix(content)
        conversation = self._build_conversation(content)
        embeddings = self._forward_conversations([conversation], normalize)
        return embeddings[0]  # (D,)

    # ------------------------------------------------------------------
    # Public: batch encoding
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        contents: List[ContentList],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode a batch of multimodal documents in a single forward pass.

        Args:
            contents:  List of content lists, one per document.
            normalize: Whether to L2-normalise the embeddings.

        Returns:
            Tensor of shape (batch_size, embedding_dim).
        """
        conversations = [
            self._build_conversation(self._apply_prefix(c)) for c in contents
        ]
        return self._forward_conversations(conversations, normalize)

    # ------------------------------------------------------------------
    # Public: modality-specific convenience helpers
    # ------------------------------------------------------------------

    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode plain text string(s).

        Args:
            texts:     Single string or list of strings.
            normalize: Whether to L2-normalise the embeddings.

        Returns:
            Shape (D,) for a single string, (N, D) for a list.
        """
        if isinstance(texts, str):
            return self.encode([{"type": "text", "text": texts}], normalize)

        return self.encode_batch(
            [[{"type": "text", "text": t}] for t in texts],
            normalize,
        )

    def encode_images(
        self,
        image_paths: Union[str, List[str]],
        text_prompt: str = "",
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode image file(s), optionally accompanied by a text prompt.

        Args:
            image_paths:  Single path/URL or list of paths/URLs.
            text_prompt:  Optional text to include alongside each image.
                          The prefix is still applied on top of this.
            normalize:    Whether to L2-normalise the embeddings.

        Returns:
            Shape (D,) for a single path, (N, D) for a list.
        """

        def _content(path: str) -> ContentList:
            content: ContentList = [{"type": "image", "image": path}]
            if text_prompt:
                content.insert(0, {"type": "text", "text": text_prompt})
            return content

        if isinstance(image_paths, str):
            return self.encode(_content(image_paths), normalize)

        return self.encode_batch([_content(p) for p in image_paths], normalize)

    def encode_videos(
        self,
        video_paths: Union[str, List[str]],
        text_prompt: str = "",
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode video file(s), optionally accompanied by a text prompt.

        Audio embedded in the video is included when use_audio_in_video=True
        (the default), matching the reference notebook behaviour.

        Args:
            video_paths:  Single path/URL or list of paths/URLs.
            text_prompt:  Optional text to include alongside each video.
            normalize:    Whether to L2-normalise the embeddings.

        Returns:
            Shape (D,) for a single path, (N, D) for a list.
        """

        def _content(path: str) -> ContentList:
            content: ContentList = [{"type": "video", "video": path}]
            if text_prompt:
                content.insert(0, {"type": "text", "text": text_prompt})
            return content

        if isinstance(video_paths, str):
            return self.encode(_content(video_paths), normalize)

        return self.encode_batch([_content(p) for p in video_paths], normalize)

    def encode_audio(
        self,
        audio_paths: Union[str, List[str]],
        text_prompt: str = "",
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode audio file(s), optionally accompanied by a text prompt.

        Args:
            audio_paths:  Single path/URL or list of paths/URLs.
            text_prompt:  Optional text to include alongside each audio file.
            normalize:    Whether to L2-normalise the embeddings.

        Returns:
            Shape (D,) for a single path, (N, D) for a list.
        """

        def _content(path: str) -> ContentList:
            content: ContentList = [{"type": "audio", "audio": path}]
            if text_prompt:
                content.insert(0, {"type": "text", "text": text_prompt})
            return content

        if isinstance(audio_paths, str):
            return self.encode(_content(audio_paths), normalize)

        return self.encode_batch([_content(p) for p in audio_paths], normalize)

    # ------------------------------------------------------------------
    # Public: similarity
    # ------------------------------------------------------------------

    def compute_similarity(
        self,
        query_embeds: torch.Tensor,
        doc_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between two sets of embeddings.

        Assumes embeddings are already L2-normalised (normalize=True, the
        default).  If you passed normalize=False, normalise manually first.

        Args:
            query_embeds: (Q, D) or (D,) tensor.
            doc_embeds:   (K, D) or (D,) tensor.

        Returns:
            (Q, K) similarity matrix, or scalar if both inputs are 1-D.
        """
        q_2d = query_embeds.unsqueeze(0) if query_embeds.dim() == 1 else query_embeds
        d_2d = doc_embeds.unsqueeze(0) if doc_embeds.dim() == 1 else doc_embeds

        sim = torch.matmul(q_2d, d_2d.T)  # (Q, K)

        if query_embeds.dim() == 1 and doc_embeds.dim() == 1:
            return sim[0, 0]
        return sim

    # ------------------------------------------------------------------
    # Public: misc
    # ------------------------------------------------------------------

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the output embeddings."""
        return self.embedding_dim

    # ------------------------------------------------------------------
    # nn.Module forward
    # ------------------------------------------------------------------

    def forward(
        self,
        contents: Optional[Union[ContentList, List[ContentList]]] = None,
        texts: Optional[Union[str, List[str]]] = None,
        image_paths: Optional[Union[str, List[str]]] = None,
        video_paths: Optional[Union[str, List[str]]] = None,
        audio_paths: Optional[Union[str, List[str]]] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Unified forward pass.

        Priority order (first non-None wins):
          1. `contents`     — pre-built content list(s), most flexible
          2. `texts`        — plain text shortcut
          3. `image_paths`  — image-only shortcut
          4. `video_paths`  — video-only shortcut
          5. `audio_paths`  — audio-only shortcut

        Args:
            contents:     A single ContentList or a list of ContentLists.
            texts:        Plain string(s).
            image_paths:  Image file path(s) or URL(s).
            video_paths:  Video file path(s) or URL(s).
            audio_paths:  Audio file path(s) or URL(s).
            normalize:    Whether to L2-normalise the output.

        Returns:
            Embedding tensor — shape (D,) for a single input, (N, D) for a
            batch.
        """
        if contents is not None:
            # Distinguish single ContentList from List[ContentList] by
            # checking whether the first element is a dict.
            if contents and isinstance(contents[0], dict):
                return self.encode(contents, normalize)  # type: ignore[arg-type]
            return self.encode_batch(contents, normalize)  # type: ignore[arg-type]

        if texts is not None:
            return self.encode_text(texts, normalize)

        if image_paths is not None:
            return self.encode_images(image_paths, normalize=normalize)

        if video_paths is not None:
            return self.encode_videos(video_paths, normalize=normalize)

        if audio_paths is not None:
            return self.encode_audio(audio_paths, normalize=normalize)

        raise ValueError(
            "Must provide at least one of: contents, texts, image_paths, "
            "video_paths, audio_paths."
        )
