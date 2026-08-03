# Paper Evaluation Stack

This directory contains the complete evaluation stack behind every table and
figure of *"Estimating Uncertainty of Omnimodal Large Language Models via
External Omnimodal Embeddings"*: the 100-run evaluation protocol, the
statistical significance testing, all appendix analyses, and the LaTeX table
generation. The scoring pipeline that *produces* the chain-level features
(grounding and coherence scores) lives in `src/` and `experiments/`; the
scripts here consume its outputs.

## Conventions

- **Working root.** The scripts read the released data artifacts and write
  their run outputs relative to a single working directory. Scripts with a
  CLI take explicit paths; the replay/analysis scripts resolve everything
  against `OMNIEVAL_ROOT` (environment variable, default: the current
  working directory). Typical usage is therefore:

  ```bash
  cd /path/to/working-root        # contains the released artifacts
  export OMNIEVAL_ROOT=$PWD       # optional, this is the default
  python /path/to/repo/evaluation/<script>.py ...
  ```

- **Determinism.** The published numbers use split seeds 42..141 (100 runs),
  `val_fraction 0.2`, and the canonical partition (seed 42) for item-level
  analyses. Every replay script contains a verification gate that reproduces
  the stored numbers before emitting anything new.

- **Naming.** `no_majority` is the paper's **main protocol** (every method
  grades the same count-majority answer); `majority_weighted` is the
  **geometry-weighted voting** protocol of Section 3.5 / Appendices J-K.
  Feature keys map to the paper as: `internal_smoothness` = S_smooth,
  `internal_semantic_density` = S_dens, `internal_goal_directedness` =
  S_goal, `cross_modal_coherence` = G_avg, `cross_modal_grounding_max` =
  G_max (Gemini+E5 key names; the MiniCPM/LCO runs use `cross_modal_overall`
  / `cross_modal_max_step_coherence` for the two grounding scores).

## Data artifacts

Small artifacts are committed under `evaluation/data/` — copy them into your
working root before running the pipeline:

| File | Content |
|---|---|
| `unobench_processed.jsonl` | UNO-Bench questions with split labels (Audio / Visual / MC / MO) |
| `gemini-majority-vote-clusters{,_top6}.json` | Semantic-cluster cache for Gemini-generated answers (Appendix A clustering, Qwen3.5-9B) |
| `minicpm-majority-vote-clusters{,_top6}.json` | Same for MiniCPM-generated answers |

The large artifacts (generated answers + CoT traces with correctness labels
and per-candidate scores, answer embeddings, input embeddings) are part of
the separate data release and must be placed in the working root:

- `gemini_cots_reformatted.json` — Gemini answers + all chain/baseline scores
- `lco_gemini_cots_reformatted.json` — same, scores from the LCO encoder (Appendix H)
- `minicpm_cots_with_correctness_and_my_scores.json` — MiniCPM answers + scores
- `gemini_cots_with_additional_baselines.json` — Gemini answers + SE/NumSets
- `gemini-answer-embeddings.json`, `minicpmo-answer-embeddings.json` — per-candidate answer embeddings (RDS / Semantic Volume)
- `embeddings/e5-all-modalities.json` — per-modality input embeddings

## Pipeline

### 0. Generate answers and chain-level scores

Answer generation, CoT step segmentation, embedding extraction and the
grounding/coherence scores are produced by the pipeline in the repository
root (see the top-level README): `experiments/run_experiments_temp.py` with
`src/embeddings` and `src/coherence`.

### 1. Assemble the evaluation records

```bash
python add_my_scores.py          # merge chain scores into the CoT records
python substitute_scores.py      # update/overwrite scores in merged records
python convert_gemini_clusters.py            # build the cluster cache (Gemini)
python convert_gemini_to_minicpm_format.py   # unify Gemini → MiniCPM record format
python add_semantic_baselines.py             # add SE / NumSets baselines
python extract_answer_embeddings.py          # answer embeddings for RDS / SV (GPU)
```

### 2. Main protocol — 100 randomised runs

`final_evaluate.py` is one evaluation run (both settings); the paper numbers
are means over 100 seeds:

```bash
python multi_seed_evaluate.py -n 100 --base-seed 42 -o gemini-majority-multirun-logistic-reg \
    -- --input-json gemini_cots_reformatted.json \
       --cluster-cache-base gemini-majority-vote-clusters.json \
       --jsonl-path unobench_processed.jsonl \
       --max-candidates 6 --weight-search-method logistic_regression
```

Repeat with the MiniCPM and LCO record files for Tables 2/9 and Appendix H.
Dispersion-baseline rows (RDS, Semantic Volume) are added by
`add_dispersion_baselines_gemini.py` / `add_dispersion_baselines_minicpm.py`.

### 3. Statistical support

```bash
python bootstrap_significance.py --input-json gemini_cots_reformatted.json \
    --cluster-cache gemini-majority-vote-clusters_top6.json \
    --run-dir gemini-majority-multirun-logistic-reg/run_0 \
    --output-dir bootstrap-results/gemini
python paired_test_ours_vs_baselines.py --results-root gemini-majority-multirun-logistic-reg
```

### 4. Appendix analyses and 5. Tables

`generate_paper_tables.py` renders the LaTeX table bodies from the stored
run statistics and bootstrap outputs.

## Script → paper map

| Script | Paper artifact |
|---|---|
| `final_evaluate.py` | Main protocol, both settings (Sections 4-5) |
| `multi_seed_evaluate.py` | 100-run means of Tables 1, 2, 9, 15, 16, 18, 19 |
| `paired_test_ours_vs_baselines.py` | Run-level paired t-tests: Tables 3, 10, 17, 20 |
| `bootstrap_significance.py` | Section 5.1 item-level bootstrap; Appendix C (Tables 4-6, Platt calibration); shared replay library |
| `generate_paper_tables.py` | LaTeX bodies of the main, Appendix H and Appendix J-K tables |
| `ablation_feature_subsets.py` | Appendix F, Table 11 |
| `gamma_sweep_eval.py`, `ablation_gamma_split.py` | Appendix F, S_goal hyperparameter sensitivity (deferred to code release) |
| `cchain_transfer_eval.py` | Appendix G (weight transferability) |
| `grounding_attention_eval.py` | Appendix I, Table 14 (attention-style grounding) |
| `sweep_candidates_evaluate.py` | Appendix L, Figure 6 (sampling-budget sweep) |
| `add_semantic_baselines.py` | Appendix M: SE / Num. Semantic Sets baselines |
| `extract_answer_embeddings.py` | Appendix M: answer embeddings for RDS / SV |
| `add_dispersion_baselines_gemini.py`, `add_dispersion_baselines_minicpm.py` | RDS / Semantic Volume rows of Tables 1-2 and 21 |
| `framework_dispersion_eval.py`, `hybrid_dispersion_eval.py` | Appendix M: dispersion analysis and hybrid aggregate (Table 21) |
| `single_chain_eval.py` | Appendix N, Table 22 (strictly single-generation regime) |
| `score_selected_eval.py` | Appendix N: best-of-n (score-selected) experiment |
| `best_vs_sc_analysis.py`, `cchain_selfverb_eval.py` | Section 5.2 / Limitations discussion |
| `add_my_scores.py`, `substitute_scores.py`, `convert_gemini_clusters.py`, `convert_gemini_to_minicpm_format.py` | Data preparation |

## Requirements

The analysis/replay scripts need only `numpy`, `scipy`, `scikit-learn`.
`final_evaluate.py` additionally imports `vllm` and `transformers` (used to
compute semantic clusters when no cluster cache is supplied — with the
committed caches no GPU is needed); `extract_answer_embeddings.py` and the
`--embed` phase of `gamma_sweep_eval.py` need a GPU with
`sentence-transformers`.
