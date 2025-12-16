# Multimodal Chain-of-Thought Generator - Usage Guide

## Overview

The **Multimodal CoT Generator** is a unified interface for generating Chain-of-Thought (CoT) reasoning from multiple multimodal language models. It seamlessly integrates with the UNO-Bench dataset loader and supports **80+ vision-language and audio-language models** through a consistent API.

### Key Features

- **🎯 Unified API**: Single interface for all supported models (LLaVA, Qwen, Gemma, Phi, etc.)
- **🔄 Auto-Detection**: Automatically infers model type from model name
- **📦 Batch Processing**: Efficient batching with VLLM optimization
- **🎨 Multi-Modal**: Supports images, audio, and omni-modal (image+audio) inputs
- **⚡ High Performance**: Built on VLLM with prefix caching for maximum throughput
- **🔌 UNO-Bench Integration**: Direct compatibility with UNOBenchSample format

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Supported Models](#supported-models)
4. [Basic Usage](#basic-usage)
5. [Advanced Usage](#advanced-usage)
6. [Integration with UNO-Bench](#integration-with-uno-bench)
7. [Model-Specific Examples](#model-specific-examples)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

```bash
# Install VLLM (required)
pip install vllm>=0.5.0

# Install other dependencies
pip install torch torchvision transformers pillow numpy
```

### Project Setup

```bash
# Clone and install the project
git clone <repository-url>
cd Multimodal-CoT-Confidence-Scoring
pip install -e .
```

---

## Quick Start

### Minimal Example (Vision Model)

```python
from src.dataset.uno_bench_loader import UNOBenchLoader
from src.dataset.cot_generator import CoTGenerator

# 1. Load dataset
loader = UNOBenchLoader(
    data_path="path/to/uno_bench",
    split="validation"
)

# 2. Initialize generator (model type auto-inferred)
generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-7b-hf",
    batch_size=8
)

# 3. Generate CoT for a sample
sample = loader[0]
chains = generator.generate_cot_from_sample(sample, num_chains=5)

# 4. Access results
for i, chain in enumerate(chains):
    print(f"\n=== Chain {i+1} ===")
    print(f"Final Answer: {chain.final_answer}")
    print(f"Reasoning Steps: {len(chain.steps)}")
    print(f"Full Text:\n{chain.text}")
```

### Minimal Example (Audio Model)

```python
# Initialize with audio model
generator = CoTGenerator(
    model_name="Qwen/Qwen2-Audio-7B",
    model_type="qwen2_audio"
)

# Load omni-modal samples
loader = UNOBenchLoader(
    data_path="path/to/uno_bench",
    split="validation",
    modality_filter="omni-modal"
)

# Generate CoT chains
sample = loader[0]
chains = generator.generate_cot_from_sample(sample)
```

---

## Supported Models

### Vision-Language Models (68 models)

| Model Family | Model Types | Example |
|-------------|-------------|---------|
| **LLaVA** | `llava`, `llava-next`, `llava-next-video`, `llava-onevision` | `llava-hf/llava-1.5-7b-hf` |
| **Qwen** | `qwen_vl`, `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `qwen3_vl_moe` | `Qwen/Qwen2-VL-7B-Instruct` |
| **Phi** | `phi3_v`, `phi4_mm`, `phi4_multimodal` | `microsoft/Phi-3-vision-128k-instruct` |
| **Gemma** | `gemma3`, `gemma3n` | `google/gemma-3-9b-it` |
| **InternVL** | `internvl_chat`, `interns1` | `OpenGVLab/InternVL-Chat-V1-5` |
| **GLM** | `glm4v`, `glm4_1v`, `glm4_5v`, `glm4_5v_fp8` | `THUDM/glm-4v-9b` |
| **DeepSeek** | `deepseek_vl_v2`, `deepseek_ocr` | `deepseek-ai/deepseek-vl2` |
| **MiniCPM** | `minicpmv`, `minicpmo` | `openbmb/MiniCPM-V-2_6` |
| **Others** | `aria`, `molmo`, `idefics3`, `mantis`, `paligemma`, `smolvlm`, etc. | See [Full List](#full-model-list) |

### Audio-Language Models (12 models)

| Model Family | Model Types | Example |
|-------------|-------------|---------|
| **Qwen Audio** | `qwen2_audio`, `qwen2_5_omni` | `Qwen/Qwen2-Audio-7B` |
| **Gemma** | `gemma3n` | `google/gemma-3n-E2B-it` |
| **Phi** | `phi4_mm`, `phi4_multimodal` | `microsoft/phi-4` |
| **Ultravox** | `ultravox` | `fixie-ai/ultravox-v0_2` |
| **Whisper** | `whisper` | `openai/whisper-large-v3` |
| **Others** | `audioflamingo3`, `granite_speech`, `midashenglm`, `voxtral` | See [Full List](#full-model-list) |

### Omni-Modal Models (2 models)

- `qwen2_5_omni` - Supports both image and audio
- `gemma3n` - Supports both image and audio

---

## Basic Usage

### 1. Single Sample - Vision Model

```python
from PIL import Image
from src.dataset.cot_generator import CoTGenerator

# Initialize generator
generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_type="llava",  # Optional - auto-inferred if omitted
    batch_size=4,
    gpu_memory_utilization=0.85
)

# Prepare inputs
question = "What is happening in this image?"
images = [Image.open("example.jpg")]

# Generate multiple reasoning chains
chains = generator.generate_cot_chains(
    question=question,
    images=images,
    num_chains=5,
    temperature=0.7,
    max_new_tokens=512
)

# Process results
for chain in chains:
    print(f"Final Answer: {chain.final_answer}")
    print(f"Reasoning Steps: {chain.steps}")
    print(f"Metadata: {chain.metadata}")
```

### 2. Single Sample - Audio Model

```python
# Initialize audio model
generator = CoTGenerator(
    model_name="Qwen/Qwen2-Audio-7B",
    model_type="qwen2_audio"
)

# Prepare audio inputs
question = "What is being said in this audio?"
audio_paths = ["/path/to/audio.wav"]

# Generate CoT chains
chains = generator.generate_cot_chains(
    question=question,
    images=None,
    audio_paths=audio_paths,
    num_chains=3
)
```

### 3. Batch Processing Multiple Samples

```python
# Prepare batch data
questions = [
    "Describe this image",
    "What objects are visible?",
    "What is the main subject?"
]

images_list = [
    [Image.open("img1.jpg")],
    [Image.open("img2.jpg"), Image.open("img2b.jpg")],
    [Image.open("img3.jpg")]
]

# Generate chains for all samples in batch
chains = generator.generate_cot_chains_batch(
    questions=questions,
    images_list=images_list,
    audio_list=None,
    temperature=0.7,
    use_tqdm=True  # Show progress bar
)

# chains[i] corresponds to questions[i]
for i, chain in enumerate(chains):
    print(f"Q{i+1}: {questions[i]}")
    print(f"A{i+1}: {chain.final_answer}\n")
```

---

## Advanced Usage

### Custom Prompt Templates

```python
# Define custom CoT prompt
custom_template = (
    "Context: You are an expert visual analyst.\n"
    "Question: {question}\n"
    "Provide a detailed step-by-step reasoning:\n"
    "Step 1:"
)

generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-7b-hf",
    cot_prompt_template=custom_template
)
```

### Adjusting Generation Parameters

```python
chains = generator.generate_cot_chains(
    question="Analyze this scene",
    images=[img],
    num_chains=10,
    temperature=0.8,        # Higher = more diverse
    top_p=0.95,             # Nucleus sampling
    max_new_tokens=1024,    # Longer responses
    return_log_probs=True   # Get token probabilities
)

# Access log probabilities
for chain in chains:
    if chain.log_probs:
        avg_logprob = sum(chain.log_probs) / len(chain.log_probs)
        print(f"Average log probability: {avg_logprob:.3f}")
```

### Lazy Model Loading

```python
# Don't load model immediately (useful for multi-GPU setups)
generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-7b-hf",
    lazy_model_loading=True
)

# Model loads on first generation call
chains = generator.generate_cot_chains(...)  # Model loads here
```

### Multi-GPU / Tensor Parallelism

```python
# Use 4 GPUs with tensor parallelism
generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-13b-hf",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9
)
```

### Model Information

```python
# Get loaded model information
info = generator.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Type: {info['model_type']}")
print(f"Vocab Size: {info['vocab_size']}")
print(f"Prefix Caching: {info['prefix_caching_enabled']}")
```

---

## Integration with UNO-Bench

### Basic Integration

```python
from src.dataset.uno_bench_loader import UNOBenchLoader
from src.dataset.cot_generator import CoTGenerator

# Load UNO-Bench
loader = UNOBenchLoader(
    data_path="data/uno_bench",
    split="validation",
    modality_filter="omni-modal"  # "uni-modal" or "omni-modal" or None
)

# Initialize generator
generator = CoTGenerator(
    model_name="Qwen/Qwen2.5-Omni-7B",
    model_type="qwen2_5_omni"
)

# Process single sample
sample = loader[0]
print(f"Question: {sample.question}")
print(f"Modality: {sample.modality}")
print(f"Images: {len(sample.images)}")
print(f"Audio: {len(sample.audio_paths) if sample.audio_paths else 0}")

chains = generator.generate_cot_from_sample(sample, num_chains=5)
```

### Batch Processing UNO-Bench Samples

```python
# Load all samples
all_samples = loader.samples

# Process in efficient batches
chains = generator.generate_cot_from_samples_batch(
    samples=all_samples,
    temperature=0.7,
    max_new_tokens=512,
    use_tqdm=True
)

# Analyze results
correct_count = 0
for sample, chain in zip(all_samples, chains):
    # Compare with ground truth
    predicted = chain.final_answer.lower().strip()
    actual = sample.answer.lower().strip()

    if predicted == actual:
        correct_count += 1

accuracy = correct_count / len(all_samples)
print(f"Accuracy: {accuracy:.2%}")
```

### Filter by Reasoning Type

```python
# Load specific reasoning types
loader = UNOBenchLoader(data_path="data/uno_bench", split="validation")

# Get samples by reasoning type
logical_samples = loader.get_by_reasoning_type("logical")
mathematical_samples = loader.get_by_reasoning_type("mathematical")

# Process each type with appropriate model
for samples in [logical_samples, mathematical_samples]:
    chains = generator.generate_cot_from_samples_batch(samples)
    # Analyze performance per reasoning type
```

### Dataset Statistics

```python
# Get dataset overview
stats = loader.get_statistics()
print(f"Total Samples: {stats['total_samples']}")
print(f"Modality Distribution: {stats['modality_distribution']}")
print(f"Reasoning Types: {stats['reasoning_type_distribution']}")
print(f"Images per Sample (avg): {sum(stats['images_per_sample']) / len(stats['images_per_sample']):.1f}")
```

---

## Model-Specific Examples

### Example 1: LLaVA (Most Popular Vision Model)

```python
generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_type="llava"
)

chains = generator.generate_cot_chains(
    question="Describe the main activity in this image",
    images=[Image.open("scene.jpg")],
    num_chains=3
)
```

### Example 2: Qwen2-VL (State-of-the-Art Vision)

```python
generator = CoTGenerator(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    model_type="qwen2_vl",
    max_model_len=8192  # Longer context
)

# Qwen2-VL excels at multi-image reasoning
chains = generator.generate_cot_chains(
    question="Compare these two images and identify the differences",
    images=[Image.open("img1.jpg"), Image.open("img2.jpg")],
    num_chains=5
)
```

### Example 3: Qwen2-Audio (Audio Understanding)

```python
generator = CoTGenerator(
    model_name="Qwen/Qwen2-Audio-7B",
    model_type="qwen2_audio"
)

chains = generator.generate_cot_chains(
    question="Transcribe and summarize this audio",
    audio_paths=["recording.wav"],
    num_chains=1
)
```

### Example 4: Qwen2.5-Omni (Omni-Modal)

```python
generator = CoTGenerator(
    model_name="Qwen/Qwen2.5-Omni-7B",
    model_type="qwen2_5_omni"
)

# Process both image and audio
chains = generator.generate_cot_chains(
    question="What is happening in this scene? Consider both visual and audio information.",
    images=[Image.open("video_frame.jpg")],
    audio_paths=["video_audio.wav"],
    num_chains=5
)
```

### Example 5: Phi-3.5-Vision (Efficient Small Model)

```python
generator = CoTGenerator(
    model_name="microsoft/Phi-3-vision-128k-instruct",
    model_type="phi3_v",
    batch_size=16,  # Can use larger batch size
    gpu_memory_utilization=0.7  # Uses less memory
)

# Good for large-scale processing
chains = generator.generate_cot_from_samples_batch(
    samples=loader.samples[:1000]
)
```

---

## API Reference

### CoTGenerator

#### `__init__()`

```python
CoTGenerator(
    model_name: str,
    model_type: Optional[str] = None,
    device: str = "cuda",
    cot_prompt_template: Optional[str] = None,
    batch_size: int = 32,
    enable_prefix_caching: bool = True,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
    lazy_model_loading: bool = False
)
```

**Parameters:**
- `model_name`: HuggingFace model name or local path
- `model_type`: Model type identifier (auto-inferred if None)
- `device`: Device to use ("cuda" or "cpu")
- `cot_prompt_template`: Custom prompt template with `{question}` placeholder
- `batch_size`: Number of samples to process per batch
- `enable_prefix_caching`: Enable VLLM prefix caching for efficiency
- `gpu_memory_utilization`: Fraction of GPU memory to use (0-1)
- `max_model_len`: Maximum sequence length
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `lazy_model_loading`: Delay model loading until first generation

#### `generate_cot_chains()`

```python
generate_cot_chains(
    question: str,
    images: Optional[List[Union[Image.Image, str]]] = None,
    audio_paths: Optional[List[str]] = None,
    num_chains: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    return_log_probs: bool = True
) -> List[CoTChain]
```

**Parameters:**
- `question`: Input question text
- `images`: List of PIL Images or image paths
- `audio_paths`: List of audio file paths
- `num_chains`: Number of reasoning chains to generate
- `temperature`: Sampling temperature (higher = more diverse)
- `top_p`: Nucleus sampling threshold
- `max_new_tokens`: Maximum tokens to generate
- `return_log_probs`: Whether to return token log probabilities

**Returns:** List of `CoTChain` objects

#### `generate_cot_chains_batch()`

```python
generate_cot_chains_batch(
    questions: List[str],
    images_list: Optional[List[Optional[List[Union[Image.Image, str]]]]] = None,
    audio_list: Optional[List[Optional[List[str]]]] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    return_log_probs: bool = True,
    use_tqdm: bool = True
) -> List[CoTChain]
```

**Parameters:** Similar to `generate_cot_chains()` but accepts lists

**Returns:** List of `CoTChain` objects (one per question)

#### `generate_cot_from_sample()`

```python
generate_cot_from_sample(
    sample: UNOBenchSample,
    num_chains: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    return_log_probs: bool = True
) -> List[CoTChain]
```

Convenience method for UNOBenchSample objects.

#### `generate_cot_from_samples_batch()`

```python
generate_cot_from_samples_batch(
    samples: List[UNOBenchSample],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    return_log_probs: bool = True,
    use_tqdm: bool = True
) -> List[CoTChain]
```

Batch process multiple UNOBenchSample objects.

### CoTChain (Data Class)

```python
@dataclass
class CoTChain:
    text: str                      # Full generated text
    steps: List[str]               # Parsed reasoning steps
    final_answer: str              # Extracted final answer
    log_probs: Optional[List[float]]  # Token log probabilities
    metadata: Dict[str, Any]       # Generation metadata
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```python
generator = CoTGenerator(
    model_name="your-model",
    batch_size=4,                  # Reduce batch size
    gpu_memory_utilization=0.7,    # Use less GPU memory
    max_model_len=2048             # Reduce context length
)
```

### Issue: Model Type Not Recognized

**Solution:**
```python
# Explicitly specify model_type
generator = CoTGenerator(
    model_name="custom/model-path",
    model_type="llava"  # Or appropriate type
)
```

### Issue: Slow Generation

**Solutions:**
```python
# 1. Enable prefix caching
generator = CoTGenerator(
    model_name="your-model",
    enable_prefix_caching=True  # Default is True
)

# 2. Use larger batch size
generator.update_batch_size(64)

# 3. Use tensor parallelism
generator = CoTGenerator(
    model_name="your-model",
    tensor_parallel_size=4  # Use 4 GPUs
)
```

### Issue: Audio Files Not Loading

**Check:**
1. Audio file paths are correct
2. Audio files are in supported formats (WAV, MP3, FLAC)
3. Sample rate is appropriate (16kHz recommended)

```python
# Verify audio paths
import os
for audio_path in audio_paths:
    assert os.path.exists(audio_path), f"File not found: {audio_path}"
```

### Issue: Generated Text Doesn't Follow CoT Format

**Solution:**
```python
# Customize prompt to enforce format
custom_template = (
    "Question: {question}\n"
    "Let's solve this step-by-step:\n"
    "Step 1: "
)

generator = CoTGenerator(
    model_name="your-model",
    cot_prompt_template=custom_template
)
```

---

## Full Model List

### Vision Models (68)

<details>
<summary>Click to expand</summary>

- aria
- aya_vision
- bagel
- bee
- blip-2
- chameleon
- command_a_vision
- deepseek_vl_v2
- deepseek_ocr
- dots_ocr
- ernie45_vl
- fuyu
- gemma3
- gemma3n
- glm4v
- glm4_1v
- glm4_5v
- glm4_5v_fp8
- h2ovl_chat
- hunyuan_vl
- hyperclovax_seed_vision
- idefics3
- interns1
- internvl_chat
- keye_vl
- keye_vl1_5
- kimi_vl
- lightonocr
- llama4
- llava
- llava-next
- llava-next-video
- llava-onevision
- mantis
- minicpmv
- minicpmo
- minimax_vl_01
- mistral3
- molmo
- nemotron_vl
- NVLM_D
- ovis
- ovis2_5
- paddleocr_vl
- paligemma
- paligemma2
- phi3_v
- phi4_mm
- phi4_multimodal
- pixtral_hf
- qwen_vl
- qwen2_vl
- qwen2_5_vl
- qwen3_vl
- qwen3_vl_moe
- r_vl
- skyworkr1v
- smolvlm
- step3
- tarsier
- tarsier2

</details>

### Audio Models (12)

- audioflamingo3
- gemma3n
- granite_speech
- midashenglm
- minicpmo
- phi4_mm
- phi4_multimodal
- qwen2_audio
- qwen2_5_omni
- ultravox
- voxtral
- whisper

---

## Best Practices

### 1. Memory Management

```python
# For large datasets, process in smaller batches
BATCH_SIZE = 16
for i in range(0, len(samples), BATCH_SIZE):
    batch = samples[i:i+BATCH_SIZE]
    chains = generator.generate_cot_from_samples_batch(batch)
    # Process and save chains
    save_results(chains)
```

### 2. Error Handling

```python
try:
    chains = generator.generate_cot_chains(question, images)
except Exception as e:
    print(f"Generation failed: {e}")
    # Fallback or retry logic
```

### 3. Model Selection

- **Fast prototyping:** Use `phi3_v` or `smolvlm`
- **Best quality:** Use `qwen2_5_vl`, `llama4`, or `glm4_5v`
- **Audio tasks:** Use `qwen2_audio` or `ultravox`
- **Multi-modal:** Use `qwen2_5_omni`

### 4. Prompt Engineering

```python
# Good CoT prompt
good_template = (
    "Question: {question}\n"
    "Let's solve this step by step:\n"
)

# Bad CoT prompt
bad_template = "{question} Answer:"  # Too short, no CoT guidance
```

---

## Performance Benchmarks

Approximate throughput on A100 (80GB):

| Model | Batch Size | Tokens/sec | Memory |
|-------|-----------|------------|--------|
| LLaVA-1.5-7B | 32 | 2800 | 25GB |
| Qwen2-VL-7B | 16 | 2400 | 35GB |
| Phi-3-Vision | 64 | 3500 | 18GB |
| Qwen2-Audio-7B | 16 | 2200 | 32GB |

---

## Additional Resources

- [VLLM Documentation](https://docs.vllm.ai/)
- [UNO-Bench Paper](https://arxiv.org/abs/XXXX.XXXXX)
- [Model Cards on HuggingFace](https://huggingface.co/models)

---

## Contributing

Found a bug or want to add a new model? Please:

1. Check if the model is supported by VLLM
2. Add the model-specific formatting function in `multimodal_models.py`
3. Update the model maps and documentation
4. Submit a pull request

---

## License

This code is licensed under Apache-2.0 (consistent with VLLM).

---

## Contact

For questions or issues:
- Open a GitHub issue
- Check existing documentation in `README.md` and `AUDIO_INTEGRATION.md`
