# Multimodal CoT Confidence Scoring - Integrated Experiment Runner

This script (`run_experiments_temp.py`) provides a comprehensive pipeline for running multimodal Chain-of-Thought (CoT) confidence scoring experiments with support for both images and audio.

## Features

- **Multi-modal Support**: Works with images, audio, or both modalities
- **Flexible CoT Generation**: Support for various models (Qwen2-Audio, vision-language models, etc.)
- **Comprehensive Encoders**: Text, multimodal (CLIP for images), and audio encoders
- **Full Coherence Pipeline**: Internal coherence, cross-modal coherence, and confidence scoring
- **Flexible Saving/Loading**: Save and load CoTs, embeddings, and scores independently
- **Iterative Experimentation**: Generate chains once, then iterate with different subsets using `--num_chains_for_scoring`
- **Extensive Configuration**: Command-line arguments for all major parameters

## Typical Workflow

### Option 1: Full Pipeline (Generate Everything)
```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --modality_filter UNOBench-Audio \
    --num_chains 20 \
    --audio_encoder facebook/wav2vec2-base-960h \
    --save_cots results/cots.json \
    --save_embeddings results/embeddings.json \
    --save_scores results/scores.json
```

### Option 2: Staged Workflow (Recommended for Large Experiments)
```bash
# Stage 1: Generate many CoTs once (expensive)
python experiments/run_experiments_temp.py \
    --num_chains 50 \
    --audio_encoder facebook/wav2vec2-base-960h \
    --save_cots results/cots_50.json \
    --save_embeddings results/embeddings_50.json

# Stage 2: Quickly iterate on scoring with different chain counts (fast)
python experiments/run_experiments_temp.py \
    --skip_embedding_extraction \
    --load_embeddings results/embeddings_50.json \
    --num_chains_for_scoring 5 \
    --save_scores results/scores_5.json

python experiments/run_experiments_temp.py \
    --skip_embedding_extraction \
    --load_embeddings results/embeddings_50.json \
    --num_chains_for_scoring 20 \
    --save_scores results/scores_20.json
```

This staged approach is **much faster** for iterating on experiments because:
- CoT generation is slow (requires running LLM inference)
- Embedding extraction is moderately slow (requires encoder inference)
- Scoring is very fast (just similarity computations)

## Quick Start

### Basic Usage - Audio Modality

Generate CoTs for audio samples and save results:

```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --modality_filter UNOBench-Audio \
    --cot_model_name Qwen/Qwen2-Audio-7B-Instruct \
    --cot_model_type qwen2_audio \
    --audio_encoder facebook/wav2vec2-base-960h \
    --num_chains 20 \
    --max_samples 10 \
    --save_cots results/audio_cots.json \
    --save_scores results/audio_scores.json \
    --log_file logs/audio_experiment.log
```

### Image Modality

Process image samples with CLIP encoder:

```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --modality_filter UNOBench-Image \
    --cot_model_name llava-hf/llava-1.5-7b-hf \
    --cot_model_type llava \
    --multimodal_encoder openai/clip-vit-large-patch14 \
    --num_chains 15 \
    --save_cots results/image_cots.json \
    --save_embeddings results/image_embeddings.json \
    --save_scores results/image_scores.json
```

### Mixed Modalities

Process all samples regardless of modality:

```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --multimodal_encoder openai/clip-vit-large-patch14 \
    --audio_encoder facebook/wav2vec2-base-960h \
    --num_chains 10 \
    --save_cots results/mixed_cots.json \
    --save_scores results/mixed_scores.json
```

## Command-Line Arguments

### Dataset Arguments

- `--data_path`: Path to UNO-Bench dataset (default: `uno-bench`)
- `--split`: Dataset split to use - `train`, `validation`, `test` (default: `validation`)
- `--modality_filter`: Filter by modality (e.g., `UNOBench-Audio`, `UNOBench-Image`, or `None` for all)
- `--max_samples`: Maximum number of samples to process (default: `None` for all)

### CoT Generation Arguments

- `--cot_model_name`: Model name for CoT generation (default: `Qwen/Qwen2-Audio-7B-Instruct`)
- `--cot_model_type`: Model type - `qwen2_audio`, `llava`, etc. (default: `qwen2_audio`)
- `--num_chains`: Number of CoT chains to generate per sample (default: `20`)
- `--num_chains_for_scoring`: Number of chains to use for scoring (subset of generated/loaded chains). If `None`, uses all chains. Useful for iterating experiments with different chain counts.
- `--max_new_tokens`: Maximum tokens for generation (default: `None`)

### Encoder Arguments

- `--text_encoder`: Text encoder for CoT steps (default: `sentence-transformers/all-mpnet-base-v2`)
- `--multimodal_encoder`: Image encoder (e.g., `openai/clip-vit-large-patch14`)
- `--audio_encoder`: Audio encoder (e.g., `facebook/wav2vec2-base-960h` or CLAP models)

### Coherence Metric Arguments

- `--similarity_metric`: Similarity metric - `cosine`, `euclidean`, `dot` (default: `cosine`)
- `--aggregation`: Aggregation method - `mean`, `min`, `max` (default: `mean`)
- `--goal_directedness_weight`: Weight for goal directedness (default: `0.6`)
- `--smoothness_weight`: Weight for step smoothness (default: `0.4`)
- `--contrastive_margin`: Margin for contrastive loss (default: `0.2`)
- `--use_attention`: Use attention weighting in cross-modal coherence

### Confidence Scoring Arguments

- `--use_density_model`: Enable density model for confidence scoring
- `--density_model_type`: Type of density model - `kde`, `gmm` (default: `kde`)
- `--internal_weight`: Weight for internal coherence (default: `0.4`)
- `--cross_modal_weight`: Weight for cross-modal coherence (default: `0.4`)
- `--density_weight`: Weight for density score (default: `0.2`)

### Save Arguments

- `--save_cots`: Path to save generated CoTs (JSON format)
- `--save_embeddings`: Path to save embeddings (JSON format)
- `--save_scores`: Path to save coherence scores (JSON format)
- `--log_file`: Path to log file

### General Arguments

- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--seed`: Random seed (default: `42`)
- `--skip_cot_generation`: Skip CoT generation and load from file
- `--load_cots`: Path to load pre-generated CoTs from
- `--skip_embedding_extraction`: Skip embedding extraction and load from file
- `--load_embeddings`: Path to load pre-computed embeddings from

## Advanced Examples

### With CLAP Audio Encoder

Use CLAP for better audio-text alignment:

```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --modality_filter UNOBench-Audio \
    --audio_encoder laion/clap-htsat-unfused \
    --use_attention \
    --save_scores results/clap_scores.json
```

### With Density Model

Enable density-based confidence scoring:

```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --use_density_model \
    --density_model_type kde \
    --internal_weight 0.3 \
    --cross_modal_weight 0.4 \
    --density_weight 0.3 \
    --save_scores results/density_scores.json
```

### Load Pre-generated CoTs

Skip CoT generation and load from file:

```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --skip_cot_generation \
    --load_cots results/audio_cots.json \
    --audio_encoder facebook/wav2vec2-base-960h \
    --save_scores results/scores_from_loaded_cots.json
```

### Load Pre-computed Embeddings

Skip both CoT generation and embedding extraction:

```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --skip_cot_generation \
    --load_cots results/audio_cots.json \
    --skip_embedding_extraction \
    --load_embeddings results/audio_embeddings.json \
    --save_scores results/scores_from_loaded_embeddings.json
```

### Iterative Chain Experiments

Generate many chains once, then iterate with different subsets:

```bash
# Step 1: Generate 50 chains and save everything
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --modality_filter UNOBench-Audio \
    --num_chains 50 \
    --audio_encoder facebook/wav2vec2-base-960h \
    --save_cots results/audio_50_cots.json \
    --save_embeddings results/audio_50_embeddings.json

# Step 2: Experiment with 5 chains
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --skip_embedding_extraction \
    --load_embeddings results/audio_50_embeddings.json \
    --num_chains_for_scoring 5 \
    --save_scores results/scores_5_chains.json

# Step 3: Experiment with 10 chains
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --skip_embedding_extraction \
    --load_embeddings results/audio_50_embeddings.json \
    --num_chains_for_scoring 10 \
    --save_scores results/scores_10_chains.json

# Step 4: Experiment with 20 chains
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --skip_embedding_extraction \
    --load_embeddings results/audio_50_embeddings.json \
    --num_chains_for_scoring 20 \
    --save_scores results/scores_20_chains.json
```

### Custom Coherence Weights

Fine-tune coherence metric weights:

```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --goal_directedness_weight 0.7 \
    --smoothness_weight 0.3 \
    --contrastive_margin 0.3 \
    --use_attention \
    --save_scores results/custom_weights_scores.json
```

## Output Format

### CoTs JSON Structure

```json
[
  [  // Sample 1
    {  // Chain 1
      "metadata": {...},
      "answer": "final answer text",
      "content": "full CoT text",
      "steps": ["step 1", "step 2", ...],
      "log_probs": [...]
    },
    ...  // More chains
  ],
  ...  // More samples
]
```

### Scores JSON Structure

```json
[
  [  // Sample 1
    {  // Chain 1
      "confidence": 0.85,
      "internal": {
        "overall": 0.82,
        "smoothness": 0.78,
        "goal_directedness": 0.85,
        ...
      },
      "cross_modal": {
        "overall": 0.88,
        "alignment": 0.90,
        ...
      },
      "density": 0.45,
      "weights": [0.4, 0.4, 0.2]
    },
    ...  // More chains
  ],
  ...  // More samples
]
```

## Notes

- **Device Selection**: The script automatically falls back to CPU if CUDA is not available
- **Memory Management**: For large datasets, consider using `--max_samples` to process in batches
- **Encoder Selection**:
  - Use CLIP-based models for images
  - Use Wav2Vec2 or CLAP for audio
  - CLAP provides better audio-text alignment than Wav2Vec2
- **Modality Priority**: If a sample has both images and audio, audio takes precedence when both encoders are provided
- **Iterative Experimentation Tips**:
  - Generate more chains than you need (e.g., 50) and save embeddings
  - Use `--num_chains_for_scoring` to quickly test with 5, 10, 15, 20, etc. chains
  - This avoids re-running expensive CoT generation and embedding extraction
  - Perfect for ablation studies on the effect of chain count
- **Loading Pre-computed Results**:
  - You can skip CoT generation with `--skip_cot_generation --load_cots`
  - You can skip embedding extraction with `--skip_embedding_extraction --load_embeddings`
  - When loading embeddings, encoders are not needed

## Troubleshooting

1. **Out of Memory**: Reduce `--num_chains` or `--max_samples`
2. **Audio Loading Errors**: Ensure `librosa` is installed: `pip install librosa`
3. **CLAP Errors**: Install CLAP: `pip install laion-clap`
4. **Missing Dataset**: Download UNO-Bench and place in the path specified by `--data_path`

## Integration with Original run_experiment.py

This script integrates the functionality from:
- `experiments/run_experiment.py` - Full encoder and coherence pipeline
- Basic CoT generation pipeline with UNO-Bench audio support
- `src/coherence/cross_modal_coherence.py` - Multi-modality support

Key improvements:
- Support for both images AND audio (not just images)
- Flexible command-line interface without requiring config files
- Direct modality detection from samples
- Comprehensive save options for all intermediate results
