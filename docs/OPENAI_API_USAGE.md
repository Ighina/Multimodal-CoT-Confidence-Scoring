# Using OpenAI API for Chain-of-Thought Generation

This guide explains how to use OpenAI's API (GPT-4o, GPT-4o-mini, GPT-4 Turbo) for generating Chain-of-Thought (CoT) reasoning instead of the default VLLM implementation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Running Experiments with OpenAI API](#running-experiments-with-openai-api)
- [Multimodal Capabilities](#multimodal-capabilities)
- [Log Probabilities](#log-probabilities)
- [Cost Considerations](#cost-considerations)
- [API Key Setup](#api-key-setup)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. An OpenAI API account with available credits
2. An API key from OpenAI ([Get one here](https://platform.openai.com/api-keys))
3. Python packages: `openai>=1.0.0`

## Supported Models

The following OpenAI models support multimodal inputs (text + images):

- **gpt-4o** (recommended) - Latest multimodal model with vision and audio support
- **gpt-4o-mini** - Faster and cheaper version of GPT-4o
- **gpt-4-turbo** - Previous generation multimodal model
- **gpt-4-turbo-2024-04-09** - Specific version of GPT-4 Turbo
- **gpt-4-vision-preview** - Preview version (older)

### Modality Support

| Model | Text | Images | Audio | Log Probs |
|-------|------|--------|-------|-----------|
| gpt-4o | ✅ | ✅ | ✅ (beta) | ✅ |
| gpt-4o-mini | ✅ | ✅ | ❌ | ✅ |
| gpt-4-turbo | ✅ | ✅ | ❌ | ✅ |

**Note:** Audio support in OpenAI API is currently limited and in beta. For audio-based tasks, VLLM with models like Qwen2-Audio is recommended.

## Installation

Install the OpenAI Python package:

```bash
pip install openai
```

Or if you have a requirements file:

```bash
pip install -r requirements.txt  # Make sure openai>=1.0.0 is included
```

## API Key Setup

You can provide your API key in two ways:

### Option 1: Environment Variable (Recommended)

Set the `OPENAI_API_KEY` environment variable:

**Linux/Mac:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY='your-api-key-here'
```

### Option 2: Command Line Argument

Pass the API key directly via command line:

```bash
python experiments/run_experiments_temp.py \
    --use_openai_api \
    --openai_api_key "your-api-key-here" \
    --cot_model_name "gpt-4o"
```

## Basic Usage

### Using OpenAICoTGenerator Directly

```python
from src.dataset.cot_generator import OpenAICoTGenerator
from src.dataset.uno_bench_loader import UNOBenchLoader

# Initialize the generator
generator = OpenAICoTGenerator(
    model_name="gpt-4o",
    api_key="your-api-key-here",  # Optional if OPENAI_API_KEY is set
    batch_size=10,
    max_retries=3,
    timeout=60.0
)

# Load dataset
loader = UNOBenchLoader(
    data_path="uno-bench",
    split="validation",
    modality_filter="UNOBench-Image"  # For image tasks
)

# Generate CoT for a single sample
sample = loader[0]
chains = generator.generate_cot_from_sample(
    sample=sample,
    num_chains=5,
    temperature=0.7,
    max_new_tokens=512,
    return_log_probs=True
)

# Access results
for i, chain in enumerate(chains):
    print(f"Chain {i+1}:")
    print(f"  Steps: {chain.steps}")
    print(f"  Final Answer: {chain.final_answer}")
    print(f"  Log Probs: {chain.log_probs[:5]}...")  # First 5 tokens
    print(f"  Tokens Used: {chain.metadata['total_tokens']}")
```

### Batch Processing

```python
# Process multiple samples
samples = loader.samples[:10]  # First 10 samples

chains = generator.generate_cot_from_samples_batch(
    samples=samples,
    temperature=0.7,
    max_new_tokens=512,
    use_tqdm=True  # Show progress bar
)

# Each chain corresponds to one sample
for sample, chain in zip(samples, chains):
    print(f"Question: {sample.question}")
    print(f"Answer: {chain.final_answer}")
```

**Efficiency Note:** The generator automatically groups identical prompts and uses OpenAI's `n` parameter to generate multiple completions in a single API call. This significantly reduces costs and latency when generating multiple chains for the same question.

## Running Experiments with OpenAI API

The `run_experiments_temp.py` script has been updated to support OpenAI API.

### Basic Command

```bash
python experiments/run_experiments_temp.py \
    --use_openai_api \
    --cot_model_name "gpt-4o" \
    --data_path "uno-bench" \
    --split "validation" \
    --modality_filter "UNOBench-Image" \
    --max_samples 10 \
    --num_chains 5 \
    --save_cots "results/cots_gpt4o.json" \
    --save_scores "results/scores_gpt4o.json"
```

### Complete Example with All Options

```bash
python experiments/run_experiments_temp.py \
    --use_openai_api \
    --openai_api_key "sk-..." \
    --cot_model_name "gpt-4o-mini" \
    --data_path "uno-bench" \
    --split "validation" \
    --modality_filter "UNOBench-Image" \
    --max_samples 50 \
    --num_chains 10 \
    --max_new_tokens 512 \
    --text_encoder "sentence-transformers/all-mpnet-base-v2" \
    --multimodal_encoder "openai/clip-vit-large-patch14" \
    --similarity_metric "cosine" \
    --use_density_model \
    --density_model_type "kde" \
    --save_cots "results/cots_gpt4o_mini.json" \
    --save_embeddings "results/embeddings/" \
    --save_scores "results/scores_gpt4o_mini.json" \
    --log_file "logs/experiment_gpt4o.log"
```

## Multimodal Capabilities

### Vision (Images)

GPT-4o and GPT-4 Turbo support image inputs:

```python
from PIL import Image

# With images
question = "What objects are in this image?"
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]

chains = generator.generate_cot_chains(
    question=question,
    images=images,
    num_chains=3
)
```

When using the experiments script:

```bash
python experiments/run_experiments_temp.py \
    --use_openai_api \
    --cot_model_name "gpt-4o" \
    --modality_filter "UNOBench-Image" \
    --multimodal_encoder "openai/clip-vit-large-patch14"
```

### Audio (Limited Support)

⚠️ **Warning:** Audio support in OpenAI API is currently limited and in beta. For better audio support, use VLLM with audio-specialized models like Qwen2-Audio.

If you need to work with audio data:

```bash
# Use VLLM instead
python experiments/run_experiments_temp.py \
    --cot_model_name "Qwen/Qwen2-Audio-7B-Instruct" \
    --cot_model_type "qwen2_audio" \
    --modality_filter "UNOBench-Audio" \
    --audio_encoder "laion/clap-htsat-unfused"
```

## Log Probabilities

The OpenAI API supports log probabilities with the following characteristics:

- **Availability:** All GPT-4o and GPT-4 Turbo models support logprobs
- **Top-K:** OpenAI returns up to 20 top log probabilities per token
- **Format:** Similar to VLLM, returned as a list of floats

Example:

```python
chains = generator.generate_cot_chains(
    question="What is 2+2?",
    return_log_probs=True
)

chain = chains[0]
print(f"Log probabilities: {chain.log_probs}")
print(f"Number of tokens: {len(chain.log_probs)}")
```

The log probabilities can be used for:
- Uncertainty estimation
- Confidence scoring
- Detecting hallucinations
- Model calibration

## API Efficiency Optimizations

The `OpenAICoTGenerator` includes several optimizations to minimize API costs and latency:

### 1. Automatic Prompt Grouping

When generating multiple chains for the same question, the generator automatically groups identical prompts and uses OpenAI's `n` parameter to generate multiple completions in a single API call.

**Example:**
```python
# Generate 5 chains for the same question
chains = generator.generate_cot_chains(
    question="What is in this image?",
    images=[img],
    num_chains=5  # Only 1 API call is made, not 5!
)
```

This reduces:
- **API calls:** 5x fewer calls
- **Input token costs:** Input tokens charged only once
- **Latency:** Faster overall processing

### 2. Retry Logic with Exponential Backoff

Automatic retry handling for transient failures:
- Up to 3 retries (configurable)
- Exponential backoff: 2s, 4s, 8s
- Prevents failed experiments due to temporary API issues

### 3. Batch Processing

Process multiple different samples efficiently:
```python
# Processes samples in optimized batches
chains = generator.generate_cot_from_samples_batch(
    samples=samples,
    use_tqdm=True
)
```

## Cost Considerations

Using OpenAI API incurs costs based on token usage. Approximate pricing (as of 2025):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4-turbo | $10.00 | $30.00 |

**Cost Estimation Example:**

- Processing 100 samples
- Each with 200 input tokens (question + image)
- Generating 500 output tokens per chain
- 5 chains per sample

**Without optimization (naive approach):**
```
Total input tokens:  100 × 200 × 5 = 100,000
Total output tokens: 100 × 500 × 5 = 250,000

For gpt-4o:
Input cost:  100,000 / 1,000,000 × $2.50  = $0.25
Output cost: 250,000 / 1,000,000 × $10.00 = $2.50
Total:                                       $2.75
```

**With our optimization (using n parameter):**
```
Total input tokens:  100 × 200 × 1 = 20,000  (5x reduction!)
Total output tokens: 100 × 500 × 5 = 250,000 (same)

For gpt-4o:
Input cost:  20,000 / 1,000,000 × $2.50  = $0.05  (saved $0.20)
Output cost: 250,000 / 1,000,000 × $10.00 = $2.50
Total:                                      $2.55  (7% cheaper)

For gpt-4o-mini:
Input cost:  20,000 / 1,000,000 × $0.15  = $0.003  (saved $0.012)
Output cost: 250,000 / 1,000,000 × $0.60  = $0.15
Total:                                      $0.153  (7% cheaper)
```

The optimization automatically saves money by charging input tokens only once per unique prompt!

**Tips to Reduce Costs:**

1. Use `gpt-4o-mini` for experimentation and prototyping
2. Limit `--max_samples` during development
3. Reduce `--num_chains` (fewer chains per sample)
4. Set `--max_new_tokens` to a reasonable value (avoid overly long generations)
5. Cache results using `--save_cots` to avoid regenerating

## Examples

### Example 1: Quick Test with GPT-4o-mini

```bash
export OPENAI_API_KEY='your-key'

python experiments/run_experiments_temp.py \
    --use_openai_api \
    --cot_model_name "gpt-4o-mini" \
    --max_samples 5 \
    --num_chains 3 \
    --save_cots "test_cots.json"
```

### Example 2: Vision Task with Full Pipeline

```bash
python experiments/run_experiments_temp.py \
    --use_openai_api \
    --cot_model_name "gpt-4o" \
    --modality_filter "UNOBench-Image" \
    --max_samples 100 \
    --num_chains 10 \
    --multimodal_encoder "openai/clip-vit-large-patch14" \
    --text_encoder "sentence-transformers/all-mpnet-base-v2" \
    --use_density_model \
    --save_cots "results/gpt4o_image_cots.json" \
    --save_scores "results/gpt4o_image_scores.json" \
    --save_embeddings "results/gpt4o_embeddings/"
```

### Example 3: Reusing Cached CoTs

Generate once:

```bash
python experiments/run_experiments_temp.py \
    --use_openai_api \
    --cot_model_name "gpt-4o" \
    --max_samples 100 \
    --save_cots "cached_cots.json"
```

Reuse for different scoring parameters:

```bash
python experiments/run_experiments_temp.py \
    --skip_cot_generation \
    --load_cots "cached_cots.json" \
    --use_density_model \
    --internal_weight 0.5 \
    --cross_modal_weight 0.5 \
    --save_scores "scores_v1.json"
```

## Troubleshooting

### Error: "openai module not found"

```bash
pip install openai
```

### Error: "Invalid API key"

- Check that your API key is correct
- Ensure you have credits in your OpenAI account
- Verify the key is set correctly:
  ```python
  import os
  print(os.getenv("OPENAI_API_KEY"))
  ```

### Error: "Rate limit exceeded"

- OpenAI has rate limits. Reduce `--batch_size` or add delays
- Consider upgrading your OpenAI plan for higher limits

### Warning: "Model may not support vision inputs"

- You're using a text-only model
- Switch to: `gpt-4o`, `gpt-4o-mini`, or `gpt-4-turbo`

### Error: Timeout

- Increase the timeout: Modify the `OpenAICoTGenerator` initialization
  ```python
  generator = OpenAICoTGenerator(..., timeout=120.0)
  ```

### High costs / Unexpected billing

- Check your token usage in the metadata:
  ```python
  print(chain.metadata['total_tokens'])
  ```
- Use `gpt-4o-mini` instead of `gpt-4o` for testing
- Reduce `--num_chains` and `--max_samples`

## Comparison: OpenAI API vs VLLM

| Feature | OpenAI API | VLLM |
|---------|-----------|------|
| **Cost** | Pay per token | Free (self-hosted) |
| **Setup** | No setup needed | Requires GPU setup |
| **Vision** | ✅ Excellent | ✅ Model-dependent |
| **Audio** | ⚠️ Limited (beta) | ✅ Excellent (Qwen2-Audio) |
| **Speed** | Fast API | Very fast (local) |
| **Models** | GPT-4 family | Any HuggingFace model |
| **Customization** | Limited | Full control |
| **Log Probs** | ✅ Up to 20 tokens | ✅ All tokens |
| **Best For** | Quick experiments, prototyping | Production, audio tasks, custom models |

## Recommendations

- **For image-based tasks:** Use `gpt-4o` or `gpt-4o-mini` with OpenAI API
- **For audio-based tasks:** Use VLLM with `Qwen2-Audio` or similar
- **For prototyping:** Use `gpt-4o-mini` to reduce costs
- **For production:** Consider VLLM for cost efficiency and control
- **For best results:** Experiment with both and compare

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [GPT-4o Vision Guide](https://platform.openai.com/docs/guides/vision)
- [OpenAI Pricing](https://openai.com/pricing)
- [UNO-Bench Dataset](https://github.com/TIGER-AI-Lab/UNO-Bench)
