"""
Omnimodal encoder for joint text-image-audio-video embeddings.
Based on LCO-Embedding and NVIDIA Omni-Embed models.
"""

from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from qwen_omni_utils import process_mm_info

class OmnimodalEncoder(nn.Module):
    """
    Encode multimodal content (text, images, audio, video) into unified embeddings.

    Supports:
    - LCO-Embedding/LCO-Embedding-Omni-7B
    - nvidia/omni-embed-nemotron-3b
    """

    def __init__(
        self,
        model_name: str = "LCO-Embedding/LCO-Embedding-Omni-7B",
        device: str = "cuda",
        base_url: Optional[str] = None,
        torch_dtype = torch.bfloat16,
        use_audio_in_video: bool = True,
    ):
        """
        Initialize omnimodal encoder.

        Args:
            model_name: Model name or path (LCO or NVIDIA model)
            device: Device to run on
            base_url: Base URL for loading media files
            torch_dtype: Torch dtype for model
            use_audio_in_video: Whether to extract audio from video
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.base_url = base_url or "https://huggingface.co/datasets/meituan-longcat/UNO-Bench/resolve/main/"
        self.use_audio_in_video = use_audio_in_video
        self.torch_dtype = torch_dtype

        self._load_model()

    def _load_model(self):
        """Load model and processor."""
        if "nvidia" in self.model_name.lower():
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                attn_implementation="eager",
                trust_remote_code=True,
                device_map="auto"
            )
            self.model_type = "nvidia"
        else:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto"
            )
            self.model_type = "lco"

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()

        self.embedding_dim = self._get_embedding_dim()

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension from model."""
        if self.model_type == "nvidia":
            return 2048
        else:
            return 3584

    def create_text_messages(self, 
                             question: str, 
                             steps: str):
      messages_question = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }]

      message_answer = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": steps[-1]
                        }
                    ]}]

      messages_steps = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": steps[0]
                        }
                    ]
                }]
      for step in steps[1:-1]:
        messages_steps.append(
            {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": step
                        }
                    ]
                }
        )

      return {"question": messages_question,
              "answer": message_answer,
              "steps": messages_steps}

    def create_messages(
        self,
        question: str,
        audios: Optional[Dict[str, str]] = None,
        images: Optional[Dict[str, str]] = None,
        videos: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List]:
        """
        Create message formats for different modality combinations.

        Args:
            question: Question text
            audios: Dictionary of audio paths
            images: Dictionary of image paths
            videos: Dictionary of video paths

        Returns:
            Dictionary with message formats for all, audio-only, image-only, video-only
        """
        messages = {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }
        messages_audio = {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }
        messages_video = {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }
        messages_image = {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }

        if audios:
            for key, audio in audios.items():
                if audio is not None:
                    audio_url = self.base_url + audio if not audio.startswith("http") else audio
                    messages["content"].append({"type": "audio", "audio": audio_url})
                    messages_audio["content"].append({"type": "audio", "audio": audio_url})

        if images:
            for key, image in images.items():
                if image is not None:
                    image_url = self.base_url + image if not image.startswith("http") else image
                    messages["content"].append({"type": "image", "image": image_url})
                    messages_image["content"].append({"type": "image", "image": image_url})

        if videos:
            for key, video in videos.items():
                if video is not None:
                    video_url = self.base_url + video if not video.startswith("http") else video
                    messages["content"].append({"type": "video", "video": video_url})
                    messages_video["content"].append({"type": "video", "video": video_url})

        return {
            "all": [[messages]],
            "image": [[messages_image]],
            "audio": [[messages_audio]],
            "video": [[messages_video]]
        }

    def _run_model(self, inputs: Dict) -> torch.Tensor:
        """
        Run model inference and extract embeddings.

        Args:
            inputs: Processed inputs dictionary

        Returns:
            Embedding tensor
        """
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.inference_mode():
            if self.model_type == "nvidia":
                last_hidden_states = self.model(
                    **inputs,
                    output_hidden_states=True
                ).hidden_states[-1]

                attention_mask = inputs["attention_mask"]
                last_hidden_states_masked = last_hidden_states.masked_fill(
                    ~attention_mask[..., None].bool(), 0.0
                )
                embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                embedding = F.normalize(embedding, dim=-1)
            else:
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                embedding = outputs.hidden_states[-1][:, -1, :]

        embedding = embedding.detach().cpu()
        torch.cuda.empty_cache()

        return embedding
    
    def encode_text(self,
                    question: str,
                    steps: str):
      with torch.no_grad():
          messages = self.create_text_messages(question, steps)

          text = self.processor.apply_chat_template(
                  messages["question"], tokenize=False, add_generation_prompt=True
            )
          inputs = self.processor(
              text=text,
              return_tensors="pt",
              padding=True
          )
          question_embedding = self._run_model(inputs)

          text = self.processor.apply_chat_template(
                  messages["answer"], tokenize=False, add_generation_prompt=True
            )
          inputs = self.processor(
              text=text,
              return_tensors="pt",
              padding=True
          )
          answer_embedding = self._run_model(inputs)

          text = [self.processor.apply_chat_template(
                  [step], tokenize=False, add_generation_prompt=True
            ) for step in messages["steps"]]

          inputs = self.processor(
              text=text,
              return_tensors="pt",
              padding=True
          )
          step_embeddings = self._run_model(inputs)
      return question_embedding, answer_embedding, step_embeddings

    def encode(
        self,
        messages: Dict[str, List],
        audio_inputs: Optional[List] = None,
        image_inputs: Optional[List] = None,
        video_inputs: Optional[List] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Encode multimodal content into embeddings.

        Args:
            messages: Message formats for different modalities
            audio_inputs: Processed audio inputs
            image_inputs: Processed image inputs
            video_inputs: Processed video inputs

        Returns:
            Dictionary with embeddings for each modality combination
        """
        multimodal_output = {
            "audio": None,
            "image": None,
            "video": None,
            "omnimodal": None
        }

        with torch.no_grad():
            if self.model_type == "nvidia":
                text_kwargs = {
                    "truncation": True,
                    "padding": True,
                    "max_length": 204800,
                }
                videos_kwargs = {
                    "min_pixels": 32*14*14,
                    "max_pixels": 64*28*28,
                    "use_audio_in_video": self.use_audio_in_video,
                }
                audio_kwargs = {"max_length": 2048000}
            else:
                text_kwargs = {}
                videos_kwargs = {}
                audio_kwargs = {}

            if audio_inputs is not None and len(audio_inputs) > 0:
                text = self.processor.apply_chat_template(
                    messages["audio"],
                    tokenize=False,
                    add_generation_prompt=self.model_type == "lco"
                )
                inputs = self.processor(
                    text=text,
                    audio=audio_inputs,
                    images=None,
                    videos=None,
                    return_tensors="pt",
                    padding=True,
                    text_kwargs=text_kwargs if text_kwargs else None,
                    audio_kwargs=audio_kwargs if audio_kwargs else None,
                )
                multimodal_output["audio"] = self._run_model(inputs)

            if image_inputs is not None and len(image_inputs) > 0:
                text = self.processor.apply_chat_template(
                    messages["image"],
                    tokenize=False,
                    add_generation_prompt=self.model_type == "lco"
                )
                inputs = self.processor(
                    text=text,
                    audio=None,
                    images=image_inputs,
                    videos=None,
                    return_tensors="pt",
                    padding=True,
                    text_kwargs=text_kwargs if text_kwargs else None,
                )
                multimodal_output["image"] = self._run_model(inputs)

            if video_inputs is not None and len(video_inputs) > 0:
                text = self.processor.apply_chat_template(
                    messages["video"],
                    tokenize=False,
                    add_generation_prompt=self.model_type == "lco"
                )
                inputs = self.processor(
                    text=text,
                    audio=None,
                    images=None,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                    text_kwargs=text_kwargs if text_kwargs else None,
                    videos_kwargs=videos_kwargs if videos_kwargs else None,
                )
                multimodal_output["video"] = self._run_model(inputs)

            text = self.processor.apply_chat_template(
                messages["all"],
                tokenize=False,
                add_generation_prompt=self.model_type == "lco"
            )
            inputs = self.processor(
                text=text,
                audio=audio_inputs,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                text_kwargs=text_kwargs if text_kwargs else None,
                videos_kwargs=videos_kwargs if videos_kwargs else None,
                audio_kwargs=audio_kwargs if audio_kwargs else None,
            )
            multimodal_output["omnimodal"] = self._run_model(inputs)

        return multimodal_output

    def encode_from_sample(
        self,
        sample: Dict,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Encode from a sample dictionary (UNO-Bench format).

        Args:
            sample: Sample dictionary with question, audios, images, videos

        Returns:
            Dictionary with embeddings for each modality combination
        """
        messages = self.create_messages(
            question=sample.get("question", ""),
            audios=sample.get("audios", {}),
            images=sample.get("images", {}),
            videos=sample.get("videos", {})
        )

        try:
            audio_inputs, image_inputs, video_inputs = process_mm_info(
                messages["all"],
                use_audio_in_video=self.use_audio_in_video
            )
        except Exception:
            audio_inputs, image_inputs, video_inputs = process_mm_info(
                messages["all"],
                use_audio_in_video=False
            )

        return self.encode(messages, audio_inputs, image_inputs, video_inputs)

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim

    def forward(
        self,
        question: Optional[str] = None,
        audios: Optional[Dict[str, str]] = None,
        images: Optional[Dict[str, str]] = None,
        videos: Optional[Dict[str, str]] = None,
        sample: Optional[Dict] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass for encoding.

        Args:
            question: Question text
            audios: Dictionary of audio paths
            images: Dictionary of image paths
            videos: Dictionary of video paths
            sample: Sample dictionary (alternative to providing individual components)

        Returns:
            Dictionary with embeddings for each modality combination
        """
        if sample is not None:
            return self.encode_from_sample(sample)

        messages = self.create_messages(question, audios, images, videos)

        try:
            audio_inputs, image_inputs, video_inputs = process_mm_info(
                messages["all"],
                use_audio_in_video=self.use_audio_in_video
            )
        except Exception:
            audio_inputs, image_inputs, video_inputs = process_mm_info(
                messages["all"],
                use_audio_in_video=False
            )

        return self.encode(messages, audio_inputs, image_inputs, video_inputs)
