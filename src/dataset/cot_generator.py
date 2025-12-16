import re
import asyncio
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from PIL import Image
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Import multimodal model formatter
from .multimodal_models import format_multimodal_prompt


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
        model_type: Optional[str] = None,
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
            model_type: Model type identifier for multimodal formatting (e.g., "llava", "qwen2_audio")
                       If None, will attempt to infer from model_name
            device: Device to run model on
            cot_prompt_template: Template for CoT prompting
            batch_size: Batch size for processing multiple requests
            enable_prefix_caching: Enable automatic prefix caching in VLLM
            gpu_memory_utilization: Fraction of GPU memory to use (0-1)
            max_model_len: Maximum sequence length (None for model default)
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.model_name = model_name
        self.model_type = model_type or self._infer_model_type(model_name)
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

        if not lazy_model_loading:
            self._load_model()

    def _infer_model_type(self, model_name: str) -> str:
        """
        Infer model type from model name/path.

        Args:
            model_name: Model name or path

        Returns:
            Inferred model type string
        """
        model_name_lower = model_name.lower()

        # Common model type patterns
        if "llava" in model_name_lower:
            if "onevision" in model_name_lower:
                return "llava-onevision"
            elif "next" in model_name_lower:
                if "video" in model_name_lower:
                    return "llava-next-video"
                return "llava-next"
            return "llava"
        elif "qwen2-audio" in model_name_lower or "qwen2_audio" in model_name_lower:
            return "qwen2_audio"
        elif "qwen2.5-omni" in model_name_lower or "qwen2_5_omni" in model_name_lower:
            return "qwen2_5_omni"
        elif "qwen2-vl" in model_name_lower or "qwen2_vl" in model_name_lower:
            return "qwen2_vl"
        elif "qwen2.5-vl" in model_name_lower or "qwen2_5_vl" in model_name_lower:
            return "qwen2_5_vl"
        elif "qwen3-vl" in model_name_lower or "qwen3_vl" in model_name_lower:
            if "moe" in model_name_lower:
                return "qwen3_vl_moe"
            return "qwen3_vl"
        elif "phi-4" in model_name_lower or "phi4" in model_name_lower:
            if "multimodal" in model_name_lower:
                return "phi4_multimodal"
            return "phi4_mm"
        elif "phi-3" in model_name_lower or "phi3" in model_name_lower:
            return "phi3_v"
        elif "gemma-3" in model_name_lower or "gemma3" in model_name_lower:
            if "n" in model_name_lower:
                return "gemma3n"
            return "gemma3"
        elif "internvl" in model_name_lower:
            return "internvl_chat"
        elif "minicpm" in model_name_lower:
            if "o" in model_name_lower:
                return "minicpmo"
            return "minicpmv"
        elif "molmo" in model_name_lower:
            return "molmo"
        elif "ultravox" in model_name_lower:
            return "ultravox"
        elif "whisper" in model_name_lower:
            return "whisper"
        elif "aria" in model_name_lower:
            return "aria"
        elif "glm-4v" in model_name_lower or "glm4v" in model_name_lower:
            if "5" in model_name_lower:
                if "fp8" in model_name_lower:
                    return "glm4_5v_fp8"
                return "glm4_5v"
            elif "1" in model_name_lower:
                return "glm4_1v"
            return "glm4v"
        elif "deepseek" in model_name_lower:
            if "v2" in model_name_lower:
                return "deepseek_vl_v2"
            if "ocr" in model_name_lower:
                return "deepseek_ocr"
            return "deepseek_vl_v2"

        # Default to llava if can't infer
        print(f"Warning: Could not infer model type from '{model_name}', defaulting to 'llava'")
        return "llava"

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

    def generate_cot_from_sample(
        self,
        sample,
        num_chains: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        return_log_probs: bool = True,
    ) -> List[CoTChain]:
        """
        Generate CoT chains from a UNOBenchSample.

        This is a convenience method that extracts the question, images, and
        audio paths from a UNOBenchSample and generates CoT chains.

        Args:
            sample: UNOBenchSample instance
            num_chains: Number of chains to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            return_log_probs: Whether to return log probabilities

        Returns:
            List of CoTChain objects
        """
        return self.generate_cot_chains(
            question=sample.question,
            images=sample.images if sample.images else None,
            audio_paths=sample.audio_paths if sample.audio_paths else None,
            num_chains=num_chains,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_log_probs=return_log_probs,
        )

    def generate_cot_from_samples_batch(
        self,
        samples: List,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        return_log_probs: bool = True,
        use_tqdm: bool = True,
    ) -> List[CoTChain]:
        """
        Generate CoT chains from multiple UNOBenchSamples in optimized batches.

        This is the most efficient way to process multiple UNO-Bench samples.

        Args:
            samples: List of UNOBenchSample instances
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            return_log_probs: Whether to return log probabilities
            use_tqdm: Show progress bar

        Returns:
            List of CoTChain objects (one per sample)
        """
        questions = [s.question for s in samples]
        images_list = [s.images if s.images else None for s in samples]
        audio_list = [s.audio_paths if s.audio_paths else None for s in samples]

        return self.generate_cot_chains_batch(
            questions=questions,
            images_list=images_list,
            audio_list=audio_list,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_log_probs=return_log_probs,
            use_tqdm=use_tqdm,
        )

    def generate_cot_chains(
        self,
        question: str,
        images: Optional[List[Union[Image.Image, str]]] = None,
        audio_paths: Optional[List[str]] = None,
        num_chains: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        return_log_probs: bool = True,
    ) -> List[CoTChain]:
        """
        Generate multiple CoT chains for a single question with multimodal inputs.

        Args:
            question: Input question
            images: List of input images (PIL Images or paths)
            audio_paths: List of audio file paths
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
            audio_list=[audio_paths] * num_chains,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_log_probs=return_log_probs,
        )

    def generate_cot_chains_batch(
        self,
        questions: List[str],
        images_list: Optional[List[Optional[List[Union[Image.Image, str]]]]] = None,
        audio_list: Optional[List[Optional[List[str]]]] = None,
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
        Uses the unified multimodal_models interface for model-specific formatting.

        Args:
            questions: List of input questions
            images_list: List of image lists (PIL Images or paths), one per question
            audio_list: List of audio path lists, one per question
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            return_log_probs: Whether to return log probabilities
            use_tqdm: Show progress bar

        Returns:
            List of CoTChain objects (one per question)
        """
        # Ensure lists have same length
        if images_list is None:
            images_list = [None] * len(questions)
        if audio_list is None:
            audio_list = [None] * len(questions)

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
            self._format_prompt(q, imgs, audios)
            for q, imgs, audios in zip(questions, images_list, audio_list)
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
            batch_audios = audio_list[start_idx:end_idx]
            for output, question, images, audios in zip(
                outputs, batch_questions, batch_images, batch_audios
            ):
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
                        "num_images": len(images) if images else 0,
                        "num_audios": len(audios) if audios else 0,
                        "question": question,
                        "finish_reason": output.outputs[0].finish_reason,
                    },
                )
                all_chains.append(chain)

        return all_chains

    def _format_prompt(
        self,
        question: str,
        images: Optional[Union[List[Image.Image], List[str]]] = None,
        audio_paths: Optional[List[str]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Format the prompt for the model using model-specific formatting.

        This method uses the unified multimodal_models interface to format
        prompts correctly for different model architectures.

        Args:
            question: The question to answer
            images: Associated images (PIL Images or paths)
            audio_paths: Associated audio file paths

        Returns:
            Formatted prompt (string or dict with prompt and multi_modal_data)
        """
        # Format question with CoT template
        formatted_question = self.cot_prompt_template.format(question=question)

        # Use the unified multimodal formatter
        try:
            prompt_data = format_multimodal_prompt(
                model_type=self.model_type,
                question=formatted_question,
                images=images if images else None,
                audio_paths=audio_paths if audio_paths else None,
                modality="auto"
            )

            # Return format expected by VLLM
            if prompt_data.get("multi_modal_data"):
                return {
                    "prompt": prompt_data["prompt"],
                    "multi_modal_data": prompt_data["multi_modal_data"]
                }
            else:
                return prompt_data["prompt"]

        except Exception as e:
            print(f"Warning: Failed to format prompt with model-specific formatter: {e}")
            print(f"Falling back to basic formatting for model type: {self.model_type}")
            # Fallback to basic text prompt
            return formatted_question

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
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "prefix_caching_enabled": self.enable_prefix_caching,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "vocab_size": len(self.tokenizer) if self.tokenizer else None,
        }


# ============================================================================
# Usage Examples - Integration with UNO-Bench Loader and Multimodal Models
# ============================================================================

"""
EXAMPLE 1: Basic usage with vision-language model (LLaVA)
-----------------------------------------------------------

from dataset.uno_bench_loader import UNOBenchLoader
from dataset.cot_generator import CoTGenerator

# Load UNO-Bench dataset
loader = UNOBenchLoader(
    data_path="path/to/uno_bench",
    split="validation",
    modality_filter="uni-modal"  # or "omni-modal" for multimodal
)

# Initialize CoT generator with LLaVA
generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_type="llava",  # Explicit model type
    batch_size=8,
    enable_prefix_caching=True
)

# Generate CoT for a single sample
sample = loader[0]
chains = generator.generate_cot_from_sample(
    sample=sample,
    num_chains=5,
    temperature=0.7
)

for i, chain in enumerate(chains):
    print(f"Chain {i+1}:")
    print(f"  Steps: {len(chain.steps)}")
    print(f"  Final answer: {chain.final_answer}")
    print(f"  Metadata: {chain.metadata}")


EXAMPLE 2: Batch processing with audio-language model (Qwen2-Audio)
--------------------------------------------------------------------

# Initialize generator with audio model
generator = CoTGenerator(
    model_name="Qwen/Qwen2-Audio-7B",
    model_type="qwen2_audio",
    batch_size=16
)

# Load omni-modal or audio samples
loader = UNOBenchLoader(
    data_path="path/to/uno_bench",
    split="validation",
    modality_filter="omni-modal"
)

# Process multiple samples in batch
samples = loader.samples[:10]  # First 10 samples
chains = generator.generate_cot_from_samples_batch(
    samples=samples,
    temperature=0.8,
    max_new_tokens=512,
    use_tqdm=True
)

# chains[i] corresponds to samples[i]
for sample, chain in zip(samples, chains):
    print(f"Question: {sample.question}")
    print(f"Answer: {chain.final_answer}")
    print(f"Modality: {sample.modality}")


EXAMPLE 3: Using different models for different modalities
-----------------------------------------------------------

from dataset.multimodal_models import get_supported_models

# See all supported models
models = get_supported_models()
print("Vision models:", models["vision_models"])
print("Audio models:", models["audio_models"])
print("Omni models:", models["omni_models"])

# Process vision samples with Qwen2-VL
vision_loader = UNOBenchLoader(
    data_path="path/to/uno_bench",
    split="validation",
    modality_filter="uni-modal"
)

vision_generator = CoTGenerator(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    model_type="qwen2_vl"
)

vision_chains = vision_generator.generate_cot_from_samples_batch(
    samples=vision_loader.samples
)

# Process omni-modal samples with Qwen2.5-Omni
omni_loader = UNOBenchLoader(
    data_path="path/to/uno_bench",
    split="validation",
    modality_filter="omni-modal"
)

omni_generator = CoTGenerator(
    model_name="Qwen/Qwen2.5-Omni-7B",
    model_type="qwen2_5_omni"
)

omni_chains = omni_generator.generate_cot_from_samples_batch(
    samples=omni_loader.samples
)


EXAMPLE 4: Custom prompt template and advanced usage
-----------------------------------------------------

# Custom CoT prompt template
custom_template = (
    "Context: You are an expert reasoning assistant.\\n"
    "Question: {question}\\n"
    "Please provide a detailed step-by-step analysis:\\n"
)

generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_type="llava",
    cot_prompt_template=custom_template,
    batch_size=4,
    gpu_memory_utilization=0.85
)

# Manual data preparation (without UNOBenchLoader)
from PIL import Image

question = "What is the total count of objects in these images?"
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
audio_paths = None  # No audio for vision-only model

chains = generator.generate_cot_chains(
    question=question,
    images=images,
    audio_paths=audio_paths,
    num_chains=3
)


EXAMPLE 5: Model type inference (automatic)
--------------------------------------------

# No need to specify model_type - it will be inferred from model_name
generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-7b-hf",  # Will infer "llava"
    # model_type is automatically set to "llava"
)

generator2 = CoTGenerator(
    model_name="Qwen/Qwen2-Audio-7B",  # Will infer "qwen2_audio"
)

# Check inferred model type
print(generator.model_type)  # Output: "llava"
print(generator2.model_type)  # Output: "qwen2_audio"
"""
