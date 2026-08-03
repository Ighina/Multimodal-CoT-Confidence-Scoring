# Estimating Uncertainty of Omnimodal Large Language Models via External Omnimodal Embeddings

Code for the anonymous ACL submission *"Estimating Uncertainty of Omnimodal Large Language Models via External Omnimodal Embeddings"*.

We propose a model-agnostic, sampling-free uncertainty quantification (UQ) framework for omni-modal LLMs. A generated Chain-of-Thought (CoT) and its final answer are projected into a shared semantic space with an external omnimodal embedding model (e.g., E5-Omni), and confidence is estimated from:

- **Grounding** (`G_avg`, `G_max`): alignment between reasoning steps and the multimodal inputs (audio, images, text), in uni-modal and omni-modal variants.
- **Coherence** (`S_smooth`, `S_dens`, `S_goal`): local smoothness, semantic density, and answer convergence of the reasoning chain.
- **`C_chain`**: a logistic-regression aggregation of the above signals.

No logits, hidden states, or repeated sampling from the evaluated model are required.

## Repository Structure

```
├── src/
│   ├── coherence/                  # Core scoring methods
│   │   ├── internal_coherence.py           # S_smooth, S_dens, S_goal (chain geometry)
│   │   ├── internal_coherence_sampling.py  # Candidate-pool variants of internal scores
│   │   ├── cross_modal_coherence.py        # Grounding scores (G_avg, G_max) vs. inputs
│   │   ├── cross_modal_coherence_sampling.py # Candidate-pool variants of grounding
│   │   ├── chain_confidence.py             # Combined chain-level confidence scorer
│   │   ├── answer_aggregation.py           # Geometry-weighted majority voting (Sec. 3.5, App. J)
│   │   ├── expectation_maximization.py     # EM-based embedding fusion utilities
│   │   ├── nli_coherence.py / prm_coherence.py # Auxiliary text-based coherence signals
│   │   └── answer_agreement_mixin.py
│   ├── embeddings/                 # External encoders
│   │   ├── omnimodal_encoder.py            # Omnimodal embedding models (e.g., E5-Omni)
│   │   ├── multimodal_encoder.py           # CLIP-style image and audio encoders
│   │   ├── text_encoder.py                 # Sentence-transformer text encoders
│   │   └── embedding_utils.py              # Similarity computation, caching
│   ├── coherence_models/           # Density models (KDE/GMM) and confidence heads
│   ├── dataset/                    # Data loading and answer generation
│   │   ├── uno_bench_loader.py             # UNO-Bench dataset loader
│   │   ├── download_unobench.py            # Dataset download helper
│   │   ├── cot_generator.py                # CoT generation (vLLM + OpenAI-compatible APIs)
│   │   ├── multimodal_models.py            # vLLM configs for multimodal generators
│   │   ├── create_gemini_cot_batch.py      # Build Gemini batch requests
│   │   └── generate_gemini_batch_fixed.py  # Submit/poll Gemini batch jobs
│   └── evaluation/                 # Metrics (AUROC, ECE, risk-coverage) and evaluator
│       └── umpire_metrics.py               # Metric utilities for the UMPIRE baseline
├── experiments/
│   ├── run_experiments_temp.py     # Main pipeline: CoT generation → embeddings → scores
│   ├── README_run_experiments_temp.md      # Detailed usage guide for the pipeline
│   ├── new_evaluate_cot.py         # Evaluation: AUROC/ECE/AURAC, bootstrap tests,
│   │                               #   logistic-regression aggregation (C_chain)
│   ├── evaluate_weighted_frequency.py      # Geometry-weighted voting evaluation (Sec. 3.5, App. J)
│   └── run_self_verbalization_experiment.py # Self-verbalisation baseline generation
├── evaluation/                     # Full paper evaluation stack (see evaluation/README.md):
│   │                               #   100-run protocol, bootstrap significance tests,
│   │                               #   all appendix analyses, LaTeX table generation
│   └── data/                       # Committed small artifacts (splits, cluster caches)
├── examples/                       # Small usage examples
├── tests/                          # Unit tests for coherence and aggregation modules
└── docs/ANSWER_AGGREGATION.md      # Documentation of the weighted-voting extension
```

## Installation

```bash
pip install -r requirements.txt
```

Optional dependencies, depending on which components you run:

```bash
pip install vllm           # local CoT generation with open models (e.g., MiniCPM-o)
pip install google-genai   # answer generation with Gemini
pip install openai         # OpenAI-compatible API generation
pip install qwen-omni-utils # required by the omnimodal encoder wrapper
```

## Reproducing the Paper Pipeline

### 1. Download UNO-Bench

```bash
python src/dataset/download_unobench.py
```

### 2. Generate answers / CoT chains

Either locally through vLLM (e.g., MiniCPM-o), via an OpenAI-compatible API, or with the Gemini batch scripts (`src/dataset/create_gemini_cot_batch.py` and `src/dataset/generate_gemini_batch_fixed.py`). Generation can also be run directly inside the main pipeline (step 3).

### 3. Extract embeddings and compute scores

`experiments/run_experiments_temp.py` runs the full pipeline (CoT generation → embedding extraction → scoring) and supports staged execution with `--save_cots/--load_cots` and `--save_embeddings/--load_embeddings`. For the paper's omni-modal setting:

```bash
python experiments/run_experiments_temp.py \
    --data_path uno-bench \
    --skip_cot_generation --load_cots results/cots.json \
    --omnimodal_encoder <e5-omni-model-path> \
    --save_embeddings results/embeddings.json \
    --save_scores results/scores.json
```

See `experiments/README_run_experiments_temp.md` for all options.

### 4. Evaluate

```bash
python experiments/new_evaluate_cot.py \
    --cots_path results/cots.json \
    --scores_path results/scores.json \
    --output_file results/metrics.json \
    --multiple_experiments --shuffle
```

This computes AUROC, ECE, and AURAC per UNO-Bench split, fits the logistic-regression aggregation (`C_chain`), and runs the randomized repetitions and statistical tests reported in the paper.

### 5. Geometry-weighted majority voting (Section 3.5, Appendix J)

```bash
python experiments/evaluate_weighted_frequency.py --help
```

Re-weights sampled answers with the embedding-based confidence scores (`src/coherence/answer_aggregation.py`).

### 6. Reproduce the paper's tables

The complete evaluation stack used for the submission — the 100-run
randomised protocol behind every table, the item-level bootstrap and paired
significance tests, all appendix analyses (feature ablations, weight
transferability, attention-style grounding, sampling-budget sweep,
dispersion baselines, single-generation regime) and the LaTeX table
generation — lives in [`evaluation/`](evaluation/README.md), with a
script-by-script map to the paper's tables and appendices.

## Tests

```bash
pytest tests/
```
