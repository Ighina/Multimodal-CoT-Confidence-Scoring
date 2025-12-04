# Audio Modality Integration Guide

## Overview

This document describes the audio modality integration into the Multimodal CoT Confidence Scoring framework. The system now supports **images, audio, and any other modality** through a unified interface.

## Summary of Changes

### 1. New AudioEncoder Class (`src/embeddings/multimodal_encoder.py`)

A new `AudioEncoder` class has been added that supports:
- **Wav2Vec2**: General audio feature extraction from HuggingFace
- **CLAP (Contrastive Language-Audio Pretraining)**: Audio-text alignment

**Key Features:**
- Load audio from files or numpy arrays
- Automatic resampling to target sample rate
- Support for text-audio alignment (with CLAP)
- Normalized embeddings for cosine similarity

**Usage Example:**
```python
from src.embeddings import AudioEncoder

# Initialize with Wav2Vec2
audio_encoder = AudioEncoder(
    model_name="facebook/wav2vec2-base-960h",
    device="cuda",
    use_clap=False
)

# Encode audio from file
audio_embeddings = audio_encoder.encode_audio_from_file("audio.wav")

# Or from numpy array
import numpy as np
audio_data = np.random.randn(16000 * 3)  # 3 seconds at 16kHz
audio_embeddings = audio_encoder.encode_audio(audio_data, sample_rate=16000)
```

### 2. Generalized Cross-Modal Coherence (`src/coherence/cross_modal_coherence.py`)

**Before:** Image-specific parameters and methods
- `image_embeddings`
- `negative_image_embeddings`
- `compute_step_image_alignment()`

**After:** Modal-agnostic parameters and methods
- `modal_embeddings` (can be image, audio, or any modality)
- `negative_modal_embeddings`
- `compute_step_modal_alignment()`

**Backward Compatibility:** Old image-specific parameters still work via aliases

**Usage Example:**
```python
from src.coherence import CrossModalCoherenceMetric

metric = CrossModalCoherenceMetric(similarity_metric="cosine", use_attention=True)

# Works with any modality
scores = metric(
    step_embeddings=reasoning_steps_embeddings,
    modal_embeddings=audio_embeddings  # or image_embeddings, or video_embeddings
)
```

### 3. Updated Chain Confidence Scorer (`src/coherence/chain_confidence.py`)

**Changes:**
- `forward()` now accepts `modal_embeddings` (preferred) or `image_embeddings` (backward compatible)
- `negative_modal_embeddings` replaces `negative_image_embeddings` (backward compatible)
- `AdaptiveConfidenceScorer` updated similarly

**Usage Example:**
```python
from src.coherence import ChainConfidenceScorer

scorer = ChainConfidenceScorer(
    internal_weight=0.5,
    cross_modal_weight=0.5
)

result = scorer(
    step_embeddings=reasoning_steps,
    modal_embeddings=audio_embeddings  # Can be audio, image, etc.
)

confidence = result['confidence']
```

### 4. Extended UNOBenchSample (`src/data/uno_bench_loader.py`)

**New Fields:**
- `audio_paths: Optional[List[str]]` - Paths to audio files
- `audio_data: Optional[List[np.ndarray]]` - Loaded audio waveforms (lazy loading)

**Updated Loader:**
- `UNOBenchLoader._load_samples()` now loads audio paths if present in dataset
- Audio data is loaded lazily for memory efficiency

**Dataset Format:**
```json
{
  "id": "sample_001",
  "question": "What instrument is playing?",
  "answer": "violin",
  "image_paths": ["images/001.jpg"],
  "audio_paths": ["audio/001.wav"],
  "modality": "audio",
  "reasoning_type": "auditory"
}
```

### 5. Updated Data Processor (`src/data/data_processor.py`)

**Changes:**
- `CoTDataset.__getitem__()` now includes `audio_paths` and `audio_data`
- `collate_fn()` batches audio data alongside images

### 6. New Example Notebook (`notebooks/audio_modality_example.ipynb`)

A comprehensive notebook demonstrating:
- Audio encoder setup (Wav2Vec2 and CLAP)
- Encoding audio from files or arrays
- Computing cross-modal coherence with audio
- Overall confidence scoring with audio modality
- Best practices and integration tips

## Integration Guide

### Step 1: Install Dependencies

```bash
# Required for audio processing
pip install librosa

# Optional: For CLAP audio-text alignment
pip install laion-clap

# Core dependencies (already in requirements.txt)
pip install transformers torch numpy
```

### Step 2: Add Audio to Your Dataset

Update your dataset JSON to include audio paths:
```json
{
  "id": "sample_001",
  "question": "What sound is this?",
  "answer": "dog barking",
  "image_paths": [],
  "audio_paths": ["audio/sample_001.wav"],
  "modality": "audio",
  "reasoning_type": "auditory"
}
```

### Step 3: Load Audio Data

```python
from src.data import UNOBenchLoader
from src.embeddings import AudioEncoder

# Load dataset with audio
loader = UNOBenchLoader(data_path="path/to/dataset", split="test")
sample = loader[0]

# Initialize audio encoder
audio_encoder = AudioEncoder(
    model_name="facebook/wav2vec2-base-960h",
    device="cuda"
)

# Encode audio
if sample.audio_paths:
    audio_embeddings = audio_encoder.encode_audio_from_file(sample.audio_paths[0])
```

### Step 4: Compute Confidence with Audio

```python
from src.embeddings import TextEncoder
from src.coherence import ChainConfidenceScorer

# Encode reasoning steps
text_encoder = TextEncoder()
step_embeddings = text_encoder.encode_cot_steps(reasoning_steps)

# Compute confidence
scorer = ChainConfidenceScorer()
result = scorer(
    step_embeddings=step_embeddings,
    modal_embeddings=audio_embeddings
)

confidence = result['confidence'].item()
```

### Step 5: Combine Multiple Modalities

For tasks with both audio and images:

```python
from src.embeddings import AudioEncoder, MultimodalEncoder
import torch

# Encode both modalities
audio_encoder = AudioEncoder(...)
image_encoder = MultimodalEncoder(...)

audio_emb = audio_encoder.encode_audio_from_file("audio.wav")
image_emb = image_encoder.encode_images(image)

# Option 1: Concatenate (requires projection if dims differ)
combined_emb = torch.cat([audio_emb, image_emb], dim=-1)

# Option 2: Average (requires same dimensions)
combined_emb = (audio_emb + image_emb) / 2

# Option 3: Use separately and ensemble confidences
audio_conf = scorer(step_embeddings, modal_embeddings=audio_emb)
image_conf = scorer(step_embeddings, modal_embeddings=image_emb)
final_conf = (audio_conf['confidence'] + image_conf['confidence']) / 2
```

## Backward Compatibility

All existing code using `image_embeddings` will continue to work:

```python
# OLD CODE - Still works!
result = scorer(
    step_embeddings=steps,
    image_embeddings=img_emb
)

# NEW CODE - Recommended
result = scorer(
    step_embeddings=steps,
    modal_embeddings=audio_emb  # or img_emb, or video_emb
)
```

## Architecture Recommendations

### 1. Modality-Specific Encoders

For different modalities, use appropriate encoders:

| Modality | Encoder | Model |
|----------|---------|-------|
| Images | `MultimodalEncoder` | CLIP, OpenCLIP |
| Audio | `AudioEncoder` (Wav2Vec2) | facebook/wav2vec2-base-960h |
| Audio-Text | `AudioEncoder` (CLAP) | 630k-audioset-best.pt |
| Text | `TextEncoder` | sentence-transformers/all-mpnet-base-v2 |
| Video | `MultimodalEncoder` + temporal | CLIP + temporal aggregation |

### 2. Embedding Dimension Alignment

When combining modalities with different dimensions:

```python
import torch.nn as nn

# Create projection layers
audio_dim = 768  # Wav2Vec2
image_dim = 512  # CLIP
target_dim = 512

audio_proj = nn.Linear(audio_dim, target_dim)
audio_emb_proj = audio_proj(audio_emb)

# Now can combine
combined = torch.cat([audio_emb_proj, image_emb], dim=0)
```

### 3. Multi-Modal Fusion Strategies

**Early Fusion:** Combine embeddings before coherence computation
```python
combined_emb = torch.cat([audio_emb, image_emb], dim=0)
scores = metric(step_embeddings, modal_embeddings=combined_emb)
```

**Late Fusion:** Compute confidences separately then combine
```python
audio_conf = scorer(steps, modal_embeddings=audio_emb)
image_conf = scorer(steps, modal_embeddings=image_emb)
final_conf = 0.5 * audio_conf['confidence'] + 0.5 * image_conf['confidence']
```

**Attention Fusion:** Learn to weight modalities
```python
# Use weighted_alignment from attention mechanism
scores = metric(
    step_embeddings,
    modal_embeddings=torch.cat([audio_emb, image_emb], dim=0),
    use_attention=True
)
attention_weights = scores['attention_weights']  # Shows which modality each step attends to
```

## Additional Files That May Need Updates

While we've covered the core components, consider updating these files based on your needs:

1. **`experiments/run_experiment.py`**: Update to handle audio modality in experiments
2. **`tests/test_coherence.py`**: Add tests for audio embeddings
3. **`README.md`**: Update main README with audio modality documentation
4. **`examples/quick_start.py`**: Add audio example

## Performance Considerations

1. **Memory**: Audio files can be large. Use lazy loading (store paths, load on demand)
2. **Preprocessing**: Resample audio offline to target sample rate
3. **Caching**: Cache audio embeddings to avoid recomputation
4. **Batching**: Process multiple audio files in batches for efficiency

Example caching:
```python
from src.embeddings import EmbeddingCache

cache = EmbeddingCache(cache_dir="./audio_embeddings_cache")

# Check cache first
audio_id = "sample_001"
if audio_id in cache:
    audio_emb = cache.get(audio_id)
else:
    audio_emb = audio_encoder.encode_audio_from_file(audio_path)
    cache.set(audio_id, audio_emb)
```

## Testing

Test the audio integration:

```python
# Test 1: Audio encoder loads correctly
audio_encoder = AudioEncoder(device="cpu")
assert audio_encoder.get_embedding_dim() > 0

# Test 2: Audio encoding works
audio = np.random.randn(16000 * 2)
emb = audio_encoder.encode_audio(audio, sample_rate=16000)
assert emb.shape[0] == audio_encoder.get_embedding_dim()

# Test 3: Cross-modal coherence accepts audio
metric = CrossModalCoherenceMetric()
step_emb = torch.randn(5, 768)
scores = metric(step_emb, modal_embeddings=emb.unsqueeze(0))
assert 'alignment' in scores

# Test 4: Backward compatibility
scores_old = metric(step_emb, image_embeddings=emb.unsqueeze(0))
assert 'alignment' in scores_old
```

## Future Enhancements

1. **Video Modality**: Extend to video by adding temporal aggregation
2. **Multi-Modal Transformers**: Use models like ImageBind for unified embeddings
3. **Region-Level Audio**: Adapt `RegionAlignmentMetric` for audio segments
4. **Learned Fusion**: Train fusion weights instead of manual weighting

## Questions?

For issues or questions about audio integration:
1. Check the example notebook: `notebooks/audio_modality_example.ipynb`
2. Review the AudioEncoder docstrings in `src/embeddings/multimodal_encoder.py`
3. See the generalized CrossModalCoherenceMetric in `src/coherence/cross_modal_coherence.py`

## Summary

The framework now supports:
✅ Audio modality through `AudioEncoder`
✅ Generalized cross-modal coherence for any modality
✅ Backward compatibility with existing image-based code
✅ Multi-modal fusion capabilities
✅ Comprehensive example notebook

The API is designed to be modality-agnostic, making it easy to add video, depth maps, or any other modality in the future!
