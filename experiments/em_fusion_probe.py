"""
Lightweight EM-fusion probe.

Inputs:
  Either the same `--scores_path` / `--cots_path` JSON files used by
  `new_evaluate_cot.py` (nested score dicts + separate cots file with labels),
  OR a single `--combined_path` file in the cots-style layout
  (`List[List[dict]]`) where each candidate dict already contains the scores
  as flat top-level keys alongside `metadata.correct`. The two modes are
  mutually exclusive.

For a list of score names (dotted paths into the nested dict, e.g.
`cross_modal.alignment`; or flat top-level keys like `internal_overall` for
the combined file), the script computes AUROC for:

  1. Each individual score on its own.
  2. Every non-trivial subset (size >= 2) fused via the EM algorithm in
     `src.coherence.expectation_maximization.compute_em_fused_embeddings_only`.

Results are sorted by AUROC and printed as a single table. Optionally also
saved to JSON via `--output_file`.

Examples:
  # Separate scores + cots files
  python experiments/em_fusion_probe.py \\
    --scores_path runs/foo_scores.json \\
    --cots_path   runs/foo_cots.json   \\
    --scores cross_modal.alignment internal.overall nli.overall \\
    --n_chains 20

  # Combined file (scores already merged into the cots dict)
  python experiments/em_fusion_probe.py \\
    --combined_path ../Downloads/gemini-with-all-scores-new.json \\
    --scores grounding internal_overall semantic_density \\
    --n_chains 7
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coherence.expectation_maximization import compute_em_fused_embeddings_only


def load_json(file_path: str) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)


def parse_path(path: str) -> List[str]:
    """Parse a dotted score path into a list of keys.

    Accepts both `a.b.c` and `["a"]["b"]["c"]` syntaxes — the latter is
    flattened to the former by stripping brackets/quotes.
    """
    cleaned = (
        path.replace("[", ".").replace("]", "").replace('"', "").replace("'", "")
    )
    return [seg for seg in cleaned.split(".") if seg]


def get_nested(d: Any, keys: List[str], default: float = 0.0) -> float:
    """Walk a nested dict by keys; return default if any step is missing."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    if isinstance(cur, (int, float)):
        return float(cur)
    return default


def extract_labels(cots_data: List[List[Dict]], n_chains: int) -> np.ndarray:
    """Extract binary correctness labels of shape (n_examples, n_chains)."""
    out = []
    for chains in cots_data:
        row = []
        for idx in range(min(n_chains, len(chains))):
            row.append(
                1 if chains[idx].get("metadata", {}).get("correct", False) else 0
            )
        while len(row) < n_chains:
            row.append(0)
        out.append(row)
    return np.array(out, dtype=int)


def extract_score_array(
    scores_data: List[List[Dict]], keys: List[str], n_chains: int
) -> np.ndarray:
    """Extract one (n_examples, n_chains) array for the given nested path."""
    out = []
    for example_scores in scores_data:
        row = []
        for idx in range(min(n_chains, len(example_scores))):
            row.append(get_nested(example_scores[idx], keys))
        while len(row) < n_chains:
            row.append(0.0)
        out.append(row)
    return np.array(out, dtype=float)


def safe_auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Flatten and compute AUROC; returns None if undefined (single-class)."""
    s = scores.flatten()
    y = labels.flatten()
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores_path",
        type=str,
        default=None,
        help="Path to scores JSON (nested dicts). Pair with --cots_path.",
    )
    parser.add_argument(
        "--cots_path",
        type=str,
        default=None,
        help="Path to cots JSON (with metadata.correct for labels).",
    )
    parser.add_argument(
        "--combined_path",
        type=str,
        default=None,
        help=(
            "Path to a single cots-style JSON file that already contains the "
            "scores as flat top-level keys per candidate (alongside "
            "metadata.correct). Mutually exclusive with --scores_path/--cots_path."
        ),
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help=(
            "Dotted nested paths into the score dict, e.g. "
            "`cross_modal.alignment internal.overall nli.overall`. "
            "Bracket syntax (`[\"cross_modal\"][\"alignment\"]`) also works."
        ),
    )
    parser.add_argument(
        "--n_chains",
        type=int,
        default=None,
        help="Cap on chains per example. Defaults to the max in the file.",
    )
    parser.add_argument(
        "--min_subset_size",
        type=int,
        default=2,
        help="Smallest subset size to fuse with EM (default 2).",
    )
    parser.add_argument(
        "--max_subset_size",
        type=int,
        default=None,
        help=(
            "Largest subset size to fuse with EM (default = all). Lower this "
            "for tractability when many scores are provided — the powerset "
            "grows combinatorially and EM has to fit a Gaussian in "
            "subset-size dimensions per call."
        ),
    )
    parser.add_argument(
        "--covariance_type",
        type=str,
        default="full",
        choices=["full", "tied", "diag", "spherical"],
        help=(
            "Covariance shape for the EM GaussianMixture. Use 'diag' or "
            "'spherical' to keep EM tractable when fusing many features "
            "(fewer free parameters than 'full')."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional JSON path to dump the full results table.",
    )
    args = parser.parse_args()

    using_combined = args.combined_path is not None
    using_separate = args.scores_path is not None or args.cots_path is not None
    if using_combined and using_separate:
        parser.error(
            "--combined_path is mutually exclusive with --scores_path/--cots_path."
        )
    if not using_combined and not (args.scores_path and args.cots_path):
        parser.error(
            "Provide either --combined_path OR both --scores_path and --cots_path."
        )

    if using_combined:
        print(f"Loading combined file from: {args.combined_path}")
        combined = load_json(args.combined_path)
        scores_data = combined
        cots_data = combined
    else:
        print(f"Loading scores from: {args.scores_path}")
        scores_data = load_json(args.scores_path)
        print(f"Loading CoTs from:   {args.cots_path}")
        cots_data = load_json(args.cots_path)

    n_examples = min(len(scores_data), len(cots_data))
    if len(scores_data) != len(cots_data):
        print(
            f"WARNING: scores ({len(scores_data)}) and cots ({len(cots_data)}) "
            f"have different lengths — truncating to {n_examples}."
        )
        scores_data = scores_data[:n_examples]
        cots_data = cots_data[:n_examples]

    if args.n_chains is None:
        n_chains = min(
            max(len(ex) for ex in scores_data),
            max(len(ex) for ex in cots_data),
        )
    else:
        n_chains = args.n_chains
    print(f"Using n_examples={n_examples}, n_chains={n_chains}")

    labels = extract_labels(cots_data, n_chains)

    # Extract each requested score into a (n_examples, n_chains) array.
    parsed_paths: List[Tuple[str, List[str]]] = [(s, parse_path(s)) for s in args.scores]
    arrays: Dict[str, np.ndarray] = {}
    for orig, keys in parsed_paths:
        if not keys:
            print(f"WARNING: could not parse score path `{orig}` — skipping.")
            continue
        arrays[orig] = extract_score_array(scores_data, keys, n_chains)

    if len(arrays) < 1:
        print("ERROR: no usable score paths.")
        sys.exit(1)

    # 1) AUROC for each individual score.
    results: List[Dict[str, Any]] = []
    for name, arr in arrays.items():
        auroc = safe_auroc(arr, labels)
        results.append(
            {
                "kind": "raw",
                "size": 1,
                "scores": [name],
                "auroc": auroc,
            }
        )

    # 2) AUROC for every non-trivial subset fused with EM.
    score_names = list(arrays.keys())
    max_size = args.max_subset_size or len(score_names)
    min_size = max(2, args.min_subset_size)

    for k in range(min_size, max_size + 1):
        for combo in combinations(score_names, k):
            try:
                fused = compute_em_fused_embeddings_only(
                    [arrays[name] for name in combo],
                    covariance_type=args.covariance_type,
                )
                auroc = safe_auroc(fused, labels)
            except Exception as e:
                print(f"  EM fusion failed for {combo}: {e}")
                auroc = None
            results.append(
                {
                    "kind": "em",
                    "size": k,
                    "scores": list(combo),
                    "auroc": auroc,
                }
            )

    # Sort: defined AUROC first (desc), then None at the bottom.
    results.sort(key=lambda r: (r["auroc"] is None, -(r["auroc"] or 0.0)))

    # Print
    name_width = max(
        len(", ".join(r["scores"])) for r in results
    )
    name_width = max(name_width, 10)
    print()
    print(f"{'AUROC':>8}  {'kind':>4}  {'k':>2}  scores")
    print("-" * (8 + 2 + 4 + 2 + 2 + 2 + name_width))
    for r in results:
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "  N/A "
        print(
            f"{auroc_str:>8}  {r['kind']:>4}  {r['size']:>2}  "
            f"{', '.join(r['scores'])}"
        )

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "scores_path": args.scores_path,
                    "cots_path": args.cots_path,
                    "combined_path": args.combined_path,
                    "n_examples": n_examples,
                    "n_chains": n_chains,
                    "min_subset_size": min_size,
                    "max_subset_size": max_size,
                    "covariance_type": args.covariance_type,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
