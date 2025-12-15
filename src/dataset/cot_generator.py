import re
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


@dataclass
class CoTChain:
    """Container for a single Chain-of-Thought reasoning chain."""

    text: str
    steps: List[str]
    final_answer: str
    log_probs: Optional[List[float]]
    metadata: Dict[str, Any]


class CoTGenerator:
    """
    Generate Chain-of-Thought reasoning from vision-language models.

    Supports multiple LVLM architectures with optimized VLLM batching
    and prefix caching for maximum throughput.
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        cot_prompt_template: Optional[str] = None,
        batch_size: int = 32,
        enable_prefix_caching: bool = True,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        lazy_model_loading: bool = False,
    ):
        """
        Initialize CoT generator with VLLM optimization.

        Args:
            model_name: Name/path of the LVLM model
            device: Device to run model on
            cot_prompt_template: Template for CoT prompting
            batch_size: Batch size for processing multiple requests
            enable_prefix_caching: Enable automatic prefix caching in VLLM
            gpu_memory_utilization: Fraction of GPU memory to use (0-1)
            max_model_len: Maximum sequence length (None for model default)
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.enable_prefix_caching = enable_prefix_caching
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size

        # Default CoT prompt template
        self.cot_prompt_template = cot_prompt_template or (
            "Question: {question}\n"
            "Let's solve this step by step, showing clear reasoning:\n"
        )

        # Model will be loaded lazily
        self.llm = None
        self.tokenizer = None

        if not self.lazy_model_loading:
            self._load_model()

    def _load_model(self):
        """Lazy loading of VLLM model with optimizations."""
        if self.llm is not None:
            return

        print(f"Loading VLLM model: {self.model_name}")
        print(
            f"Batch size: {self.batch_size}, Prefix caching: {self.enable_prefix_caching}"
        )

        # Initialize VLLM with optimization flags
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=self.enable_prefix_caching,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            # Additional optimizations
            enforce_eager=False,  # Use CUDA graphs for better performance
            disable_log_stats=False,  # Keep stats for debugging
        )

        # Get tokenizer for potential preprocessing
        self.tokenizer = self.llm.get_tokenizer()
        print(f"Model loaded successfully. Vocab size: {len(self.tokenizer)}")

    def generate_cot_chains(
        self,
        question: str,
        images: List[Image.Image],
        num_chains: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        return_log_probs: bool = True,
    ) -> List[CoTChain]:
        # TODO: Add audio to this and next function
        """
        Generate multiple CoT chains for a single question and images.

        Args:
            question: Input question
            images: List of input images
            num_chains: Number of chains to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            return_log_probs: Whether to return log probabilities

        Returns:
            List of CoTChain objects
        """
        # Generate multiple chains for the same input
        return self.generate_cot_chains_batch(
            questions=[question] * num_chains,
            images_list=[images] * num_chains,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_log_probs=return_log_probs,
        )

    def generate_cot_chains_batch(
        self,
        questions: List[str],
        images_list: List[str],
        audio_list: List[str],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        return_log_probs: bool = True,
        use_tqdm: bool = True,
    ) -> List[CoTChain]:
        """
        Generate CoT chains for multiple questions in optimized batches.

        This is the recommended method for processing multiple samples as it
        leverages VLLM's batching and prefix caching for maximum throughput.

        Args:
            questions: List of input questions
            images_list: List of image paths (one per question)
            audio_list: List of audio paths (one per question)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            return_log_probs: Whether to return log probabilities
            use_tqdm: Show progress bar

        Returns:
            List of CoTChain objects (one per question)
        """

        # TODO: different models accept different type of processed multimodal data
        # Create another file to handle multiple models and preprocess multimodal paths according to model type

        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            logprobs=1 if return_log_probs else None,
            n=1,  # Number of completions per prompt
        )

        # Format all prompts with the template
        prompts = [
            self._format_prompt(q, imgs) for q, imgs in zip(questions, images_list)
        ]

        all_chains = []

        # Process in batches for memory efficiency
        num_batches = (len(prompts) + self.batch_size - 1) // self.batch_size

        if use_tqdm:
            from tqdm import tqdm

            batch_iterator = tqdm(
                range(num_batches), desc="Generating CoT chains", unit="batch"
            )
        else:
            batch_iterator = range(num_batches)

        for batch_idx in batch_iterator:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(prompts))

            batch_prompts = prompts[start_idx:end_idx]
            batch_questions = questions[start_idx:end_idx]
            batch_images = images_list[start_idx:end_idx]

            # Generate for entire batch at once - VLLM handles batching internally
            outputs = self.llm.generate(
                batch_prompts,
                sampling_params,
                use_tqdm=False,  # We have our own progress bar
            )

            # Process outputs
            for output, question, images in zip(outputs, batch_questions, batch_images):
                generated_text = output.outputs[0].text

                # Extract log probabilities if requested
                log_probs = None
                if return_log_probs and output.outputs[0].logprobs:
                    log_probs = [
                        logprob_dict[token_id].logprob
                        for logprob_dict, token_id in zip(
                            output.outputs[0].logprobs, output.outputs[0].token_ids
                        )
                    ]

                # Parse into structured chain
                chain = self._create_chain(
                    generated_text=generated_text,
                    log_probs=log_probs,
                    metadata={
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_images": len(images),
                        "question": question,
                        "finish_reason": output.outputs[0].finish_reason,
                    },
                )
                all_chains.append(chain)

        return all_chains

    def _format_prompt(
        self,
        question: str,
        images: List[Image.Image] = [],
        audios: List[str] = [],
        prompt_style: str = "llava",
    ) -> str:
        """
        Format the prompt for the model.

        For vision-language models, you may need to include special tokens
        or formatting. This is a basic text-only version.

        Args:
            question: The question to answer
            images: Associated images (handling depends on model)
            audios: Associated audios (handling depends on model)
            prompt_style: Style of prompt (e.g., "llava")

        Returns:
            Formatted prompt string
        """
        # Basic text formatting with template
        prompt = self.cot_prompt_template.format(question=question)

        # For VLMs, you might need to add image tokens or special formatting
        # Example for LLaVA-style models:
        if images and not audios:
            if prompt_style == "llava":
                image_tokens = " ".join(["<image>"] * len(images))
                prompt = f"{image_tokens}\n{prompt}"

                prompt = {"prompt": prompt, "multi_modal_data": {"image": images}}
        elif images:
            if prompt_style == "llava":
                image_tokens = " ".join(["<image>"] * len(images))
                audio_tokens = " ".join(["<audio>"] * len(audios))
                prompt = f"{image_tokens}\n{audio_tokens}\n{prompt}"

                prompt = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": images, "audio": audios},
                }
        elif audios:
            if prompt_style == "llava":
                # PLACEHOLDER: check audio-language model format
                audio_tokens = " ".join(["<audio>"] * len(audios))
                prompt = f"{audio_tokens}\n{prompt}"

                prompt = {"prompt": prompt, "multi_modal_data": {"audio": audios}}

        return prompt

    def _create_chain(
        self,
        generated_text: str,
        log_probs: Optional[List[float]],
        metadata: Dict[str, Any],
    ) -> CoTChain:
        """Create a CoTChain object from generated text."""
        steps = self._parse_steps(generated_text)
        final_answer = self._extract_final_answer(generated_text)

        return CoTChain(
            text=generated_text,
            steps=steps,
            final_answer=final_answer,
            log_probs=log_probs,
            metadata=metadata,
        )

    def _parse_steps(self, text: str) -> List[str]:
        """
        Parse generated text into individual reasoning steps.

        Looks for patterns like:
        - "Step 1:", "Step 2:", etc.
        - Numbered lists: "1.", "2.", etc.
        - Sentence boundaries for implicit steps
        """
        steps = []

        # Try explicit step markers first
        step_pattern = r"(?:Step\s+\d+:|^\d+\.)\s*(.+?)(?=(?:Step\s+\d+:|\d+\.|Therefore|Thus|So|Answer|$))"
        matches = re.finditer(step_pattern, text, re.MULTILINE | re.DOTALL)

        for match in matches:
            step_text = match.group(1).strip()
            if step_text:
                steps.append(step_text)

        # If no explicit steps found, split by sentences
        if not steps:
            sentences = re.split(r"[.!?]+", text)
            steps = [s.strip() for s in sentences if len(s.strip()) > 10]

        return steps

    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from the generated text."""
        # Look for common answer patterns
        patterns = [
            r"(?:Therefore|Thus|So),?\s+(?:the answer is|the final answer is)[:;]?\s*(.+)",
            r"(?:Answer|Final answer)[:;]\s*(.+)",
            r"(?:The result is|The solution is)[:;]?\s*(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no pattern found, return last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines[-1] if lines else ""

    def update_batch_size(self, new_batch_size: int):
        """Update the batch size for future generations."""
        self.batch_size = new_batch_size
        print(f"Batch size updated to: {self.batch_size}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.llm is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "prefix_caching_enabled": self.enable_prefix_caching,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "vocab_size": len(self.tokenizer) if self.tokenizer else None,
        }
