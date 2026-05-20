import re
import base64
import io
from typing import Any, Optional, List, Dict, Union
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
from PIL import Image


class OpenAICoTGenerator:
    """
    Generate Chain-of-Thought reasoning using OpenAI API.

    Supports multimodal models (GPT-4o, GPT-4o-mini, GPT-4 Turbo) with
    vision and audio capabilities (where available).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        cot_prompt_template: Optional[str] = None,
        batch_size: int = 10,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        Initialize OpenAI CoT generator.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini", "gpt-4-turbo")
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env variable)
            cot_prompt_template: Template for CoT prompting
            batch_size: Number of requests to process in parallel (for rate limiting)
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        # Default CoT prompt template
        self.cot_prompt_template = cot_prompt_template or (
            "{question}\n"
            "Let's solve this step by step, showing clear reasoning.\n"
            "answer one of the following options: A, B, C, D.\n"
        )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

        # Validate model supports multimodal
        self._validate_model()

        print(f"Initialized OpenAI CoT Generator with model: {self.model_name}")

    def _validate_model(self):
        """Validate that the model supports required features."""
        # Models that support vision
        vision_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-vision-preview",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano"
        ]

        # Models that support audio (currently limited)
        audio_models = ["gpt-audio", 
                        "gpt-audio-mini"]

        if not any(vm in self.model_name for vm in vision_models):
            print(
                f"Warning: Model {self.model_name} may not support vision inputs. "
                f"Recommended models: {', '.join(vision_models)}"
            )

    def _encode_image(self, image: Union[Image.Image, str]) -> str:
        """
        Encode image to base64 string for OpenAI API.

        Args:
            image: PIL Image or path to image file

        Returns:
            Base64 encoded image string
        """
        if isinstance(image, str):
            # Load image from path
            image = Image.open(image)

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Encode to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_str}"

    def _format_messages(
        self,
        question: str,
        images: Optional[List[Union[Image.Image, str]]] = None,
        audio_paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format messages for OpenAI API.

        Args:
            question: Question text
            images: List of images (PIL Images or paths)
            audio_paths: List of audio file paths (limited support)

        Returns:
            List of message dictionaries for OpenAI API
        """
        # Format question with CoT template
        formatted_question = self.cot_prompt_template.format(question=question)

        # Build content array
        content = []

        # Add text
        content.append({"type": "text", "text": formatted_question})

        # Add images if present
        if images:
            for img in images:
                img_base64 = self._encode_image(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": img_base64, "detail": "auto"},
                    }
                )

        # Note: Audio support in OpenAI is limited and may require special handling
        # For now, we'll add a warning if audio is provided
        if audio_paths:
            for path in audio_paths:
                with open(path, "rb") as wav:
                    wav_data = wav.read()
                encoded_string = base64.b64encode(wav_data).decode('utf-8')
                content.append(
                    {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                        }
                    }
                )

        return [{"role": "user", 
                 "content": content}]