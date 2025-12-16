# Multimodal Chain-of-Thought Confidence Scoring

A comprehensive framework for evaluating confidence in multimodal reasoning chains by measuring internal textual coherence and cross-modal alignment between reasoning steps and visual information.

## Overview

This framework implements a novel approach to confidence estimation for Large Vision-Language Models (LVLMs) performing Chain-of-Thought (CoT) reasoning on multimodal tasks. The key insight is that confident, correct reasoning should exhibit:

1. **Internal Textual Coherence**: Reasoning steps should flow logically and progress toward the goal
2. **Cross-Modal Coherence**: Reasoning steps should align with visual information from images
3. **Typicality**: Reasoning patterns should be similar to correct training examples

## Features

- **🎯 Unified Multimodal CoT Generator** ⭐ NEW
  - Support for 80+ vision and audio language models
  - Seamless integration with UNO-Bench dataset
  - Auto-detection of model types
  - Batch processing with VLLM optimization
  - See [**MULTIMODAL_COT_GENERATOR.md**](MULTIMODAL_COT_GENERATOR.md) for detailed usage

- **Multi-level Coherence Analysis**
  - Local smoothness: semantic similarity between consecutive steps
  - Goal-directedness: progression toward the answer
  - Cross-modal alignment: text-image coherence
  - Contrastive scoring: alignment to true vs. distractor images

- **Multiple Embedding Backends**
  - Sentence Transformers for text
  - CLIP/OpenCLIP for multimodal alignment
  - Support for LVLM internal representations
  - Audio embeddings (Wav2Vec2, CLAP)

- **Density-based Anomaly Detection**
  - KDE (Kernel Density Estimation)
  - GMM (Gaussian Mixture Models)
  - Neural density models

- **Comprehensive Evaluation**
  - AUC-ROC and AUC-PR
  - Calibration metrics (ECE, MCE)
  - Risk-coverage curves for selective prediction
  - Abstention analysis

- **Strong Baselines**
  - CoT length
  - Log probability
  - Majority vote / Self-consistency
  - LLM-as-judge
  - Semantic entropy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal-cot-confidence
cd multimodal-cot-confidence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 0. Generating Chain-of-Thought from Multimodal Models (NEW!)

Generate CoT reasoning from 80+ vision/audio language models with a unified interface:

```python
from src.dataset.uno_bench_loader import UNOBenchLoader
from src.dataset.cot_generator import CoTGenerator

# Load UNO-Bench dataset
loader = UNOBenchLoader(
    data_path="path/to/uno_bench",
    split="validation"
)

# Initialize CoT generator (model type auto-inferred)
generator = CoTGenerator(
    model_name="llava-hf/llava-1.5-7b-hf",
    batch_size=8
)

# Generate CoT for a single sample
sample = loader[0]
chains = generator.generate_cot_from_sample(sample, num_chains=5)

# Or batch process multiple samples
chains = generator.generate_cot_from_samples_batch(loader.samples[:10])

for chain in chains:
    print(f"Final Answer: {chain.final_answer}")
    print(f"Reasoning Steps: {chain.steps}")
```

**📖 For comprehensive usage guide, model list, and advanced examples, see [MULTIMODAL_COT_GENERATOR.md](MULTIMODAL_COT_GENERATOR.md)**

### 1. Basic Usage

```python
from src.embeddings import TextEncoder, MultimodalEncoder
from src.coherence import (
    InternalCoherenceMetric,
    CrossModalCoherenceMetric,
    ChainConfidenceScorer
)

# Setup encoders
text_encoder = TextEncoder(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
multimodal_encoder = MultimodalEncoder(
    model_name="openai/clip-vit-large-patch14"
)

# Setup coherence metrics
internal_metric = InternalCoherenceMetric(
    similarity_metric="cosine",
    aggregation="mean"
)
cross_modal_metric = CrossModalCoherenceMetric(
    similarity_metric="cosine",
    use_attention=True
)

# Create confidence scorer
scorer = ChainConfidenceScorer(
    internal_metric=internal_metric,
    cross_modal_metric=cross_modal_metric
)

# Compute confidence for a reasoning chain
step_embeddings = text_encoder.encode_cot_steps(reasoning_steps)
image_embeddings = multimodal_encoder.encode_images(images)

result = scorer(
    step_embeddings=step_embeddings,
    image_embeddings=image_embeddings
)

confidence = result['confidence']
print(f"Confidence score: {confidence:.3f}")
```

### 2. Running Experiments

```bash
# Run full experiment on UNO-Bench
python experiments/run_experiment.py \
    --config config/config.yaml \
    --experiment_name my_experiment \
    --device cuda

# Run ablation study
python experiments/ablation_study.py \
    --config config/config.yaml \
    --test_data path/to/test_data.pkl
```

### 3. Configuration

Edit `config/config.yaml` to customize:

- Data sources and preprocessing
- Model architectures and hyperparameters
- Coherence metric weights
- Training and evaluation settings
- Logging and output options

## Architecture

### Project Structure

```
multimodal-cot-confidence/
├── config/
│   └── config.yaml              # Main configuration
├── src/
│   ├── dataset/
│   │   ├── uno_bench_loader.py  # UNO-Bench dataset loader
│   │   ├── cot_generator.py     # CoT generation from LVLMs (80+ models)
│   │   ├── multimodal_models.py # Model-specific prompt formatters
│   │   └── data_processor.py    # Data preprocessing
│   ├── embeddings/
│   │   ├── text_encoder.py      # Text embedding extraction
│   │   ├── multimodal_encoder.py # Multimodal embeddings (CLIP)
│   │   └── embedding_utils.py   # Similarity and caching utilities
│   ├── coherence/
│   │   ├── internal_coherence.py    # Internal textual coherence
│   │   ├── cross_modal_coherence.py # Cross-modal alignment
│   │   └── chain_confidence.py      # Combined confidence scoring
│   ├── models/
│   │   ├── confidence_head.py   # MLP for confidence prediction
│   │   └── density_model.py     # Density estimation models
│   ├── baselines/
│   │   └── baseline_methods.py  # Baseline confidence methods
│   ├── evaluation/
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── evaluator.py         # Comprehensive evaluator
│   └── utils/
│       ├── visualization.py     # Plotting utilities
│       └── logging_utils.py     # Logging and experiment tracking
├── experiments/
│   ├── run_experiment.py        # Main experiment script
│   └── ablation_study.py        # Ablation study script
├── requirements.txt
├── setup.py
└── README.md
```

### Key Components

#### 1. Internal Coherence Metric

Measures how well reasoning steps connect:

- **Local Smoothness**: Cosine similarity between consecutive steps
- **Goal-Directedness**: Progression toward answer embedding
- **Semantic Density**: Overall compactness in embedding space

```python
internal_scores = internal_metric(
    step_embeddings=step_embeddings,
    answer_embedding=answer_embedding
)
# Returns: {'overall', 'smoothness', 'goal_directedness', 'semantic_density'}
```

#### 2. Cross-Modal Coherence Metric

Measures text-image alignment:

- **Step-Image Alignment**: Similarity of each step to images
- **Contrastive Coherence**: Alignment to true vs. negative images
- **Attention-Weighted**: Soft attention over image regions

```python
cross_modal_scores = cross_modal_metric(
    step_embeddings=step_embeddings,
    image_embeddings=image_embeddings,
    negative_image_embeddings=negative_samples
)
# Returns: {'overall', 'alignment', 'contrastive_score', 'per_step_coherence'}
```

#### 3. Chain Confidence Scorer

Combines all components:

```python
result = scorer(
    step_embeddings=step_embeddings,
    image_embeddings=image_embeddings
)
# Returns: {
#   'confidence': overall confidence score,
#   'internal': internal coherence scores,
#   'cross_modal': cross-modal scores,
#   'density': density score (if enabled),
#   'weights': component weights
# }
```

## Experiments

### Supported Benchmarks

- **UNO-Bench**: Unified uni-modal and omni-modal reasoning tasks
- **MMMU**: Massive Multimodal Understanding benchmark
- **ScienceQA**: Science question answering with diagrams
- **M3-Bench**: Multimodal reasoning tasks

### Evaluation Metrics

1. **Discrimination**
   - AUC-ROC: Area under ROC curve
   - AUC-PR: Area under precision-recall curve

2. **Calibration**
   - ECE: Expected Calibration Error
   - MCE: Maximum Calibration Error
   - Calibration curves

3. **Selective Prediction**
   - Risk-coverage curves
   - Abstention accuracy at different thresholds

4. **Correlation**
   - Pearson and Spearman correlation with correctness

### Ablation Studies

The framework supports systematic ablation studies:

```python
ablations = [
    'internal_only',      # Only internal coherence
    'cross_modal_only',   # Only cross-modal coherence
    'no_density',         # Without density component
    'different_encoders', # CLIP vs sentence-transformers
]
```

## Baseline Methods

Implemented baselines for comparison:

1. **CoT Length**: Longer chains = higher confidence
2. **Log Probability**: Token-level probability from LVLM
3. **Majority Vote**: Consistency across multiple samples
4. **LLM-as-Judge**: Prompt-based confidence rating
5. **Semantic Entropy**: Entropy across semantically clustered answers

## Advanced Usage

### Custom Coherence Metrics

```python
from src.coherence import InternalCoherenceMetric

class CustomCoherenceMetric(InternalCoherenceMetric):
    def compute_custom_score(self, step_embeddings):
        # Your custom logic
        return score

custom_metric = CustomCoherenceMetric()
```

### Training Confidence Head

```python
from src.models import ConfidenceHead
import torch.nn as nn

# Create learnable confidence head
head = ConfidenceHead(
    input_dim=10,
    hidden_dims=[512, 256, 128],
    dropout=0.3
)

# Train on labeled data
optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
criterion = nn.BCELoss()

for features, labels in train_loader:
    predictions = head(features)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
```

### Integration with Your LVLM

```python
from src.data import CoTGenerator

# Custom CoT generator for your model
class MyLVLMGenerator(CoTGenerator):
    def _generate_single_chain(self, question, images, **kwargs):
        # Call your LVLM
        output = self.model.generate(
            question=question,
            images=images,
            **kwargs
        )
        return self._parse_output(output)

generator = MyLVLMGenerator(model_name="my-lvlm")
```

## Results and Visualization

The framework automatically generates:

- Calibration curves
- Risk-coverage plots
- Confidence distributions
- Method comparison charts
- Ablation study visualizations

Results are saved to the specified output directory and logged to TensorBoard and/or Weights & Biases.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{multimodal-cot-confidence-2024,
  title={Confidence Scoring for Multimodal Chain-of-Thought Reasoning via Internal and Cross-Modal Coherence},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Documentation

- **[MULTIMODAL_COT_GENERATOR.md](MULTIMODAL_COT_GENERATOR.md)** - Complete guide for using the multimodal CoT generator with 80+ models
- **[AUDIO_INTEGRATION.md](AUDIO_INTEGRATION.md)** - Audio modality integration and usage guide
- **[README.md](README.md)** - This file: project overview and main framework usage

## Acknowledgments

This framework builds upon:

- **UNO-Bench** for unified multimodal evaluation
- **VLLM** for efficient LLM inference
- **CLIP** for cross-modal embeddings
- **Sentence Transformers** for text embeddings
- **CMRF** and **MM-PEAR-CoT** for multimodal reasoning insights
- **Semantic Entropy** work by Kuhn et al.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: your.email@example.com

## Roadmap

Future enhancements:

- [x] Support for audio modality
- [x] Integration with 80+ vision/audio language models via VLLM
- [ ] Support for video modality
- [ ] Online learning for confidence calibration
- [ ] Integration with more LVLMs (Gemini, GPT-4V, etc.)
- [ ] Real-time confidence estimation
- [ ] Active learning for selective annotation
- [ ] Multi-lingual support
