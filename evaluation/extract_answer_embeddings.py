#!/usr/bin/env python3
"""
Extract per-candidate answer embeddings with E5-Omni for the embedding-dispersion
baselines (KLE, Semantic Volume, RDS, centroid weighting).

Reads a generation file in the gemini_cots_reformatted.json format (one record
per question with `question_id` and `generations_text`, a list of N sampled
generations) and writes, for every question, one embedding per candidate.

Paper: produces the answer embeddings behind the RDS and Semantic Volume
dispersion baselines (Tables 1-2, Appendix M).

The output is a JSON list aligned with the input order:
    [{"question_id": 0,
      "embeddings": [[...d floats...], ...N entries...]},
     ...]

Usage (on the GPU server):
    python extract_answer_embeddings.py \
        --input-json gemini_cots_reformatted.json \
        --output-json embeddings/gemini-answer-embeddings.json \
        [--model Haon-Chen/e5-omni-3B] \
        [--mode generation | final_answer] \
        [--batch-size 32] [--dtype bfloat16] [--max-length 1024]

Notes:
  * --mode generation (default) embeds the full generated text (CoT + answer);
    --mode final_answer embeds only the extracted final answer, mirroring the
    text units used for semantic clustering in final_evaluate.py.
  * Output is written incrementally every --save-every questions so an
    interrupted job can be resumed (already-embedded questions are skipped).
"""

import argparse
import json
import os
import re

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def extract_final_answer(text: str) -> str:
    """Same final-answer extraction used for clustering in final_evaluate.py."""
    if not text:
        return ""
    t = re.sub(r"<\|[^|>]*\|>", "", text)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    hits = re.findall(r"ANSWER:\s*(.*)", t, flags=re.IGNORECASE)
    if hits:
        return hits[-1].strip()
    lines = [ln for ln in t.strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-json", default="gemini_cots_reformatted.json")
    p.add_argument("--output-json", required=True)
    p.add_argument("--model", default="Haon-Chen/e5-omni-3B",
                   help="HF id of the omni-modal encoder (default: %(default)s)")
    p.add_argument("--mode", choices=["generation", "final_answer"], default="generation",
                   help="Embed the full generation or only the extracted final answer.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=1024,
                   help="Token truncation length for long CoT generations.")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device", default=None,
                   help="Override device (default: cuda if available, else cpu).")
    p.add_argument("--save-every", type=int, default=200,
                   help="Write partial output every this many questions (resume-safe).")
    p.add_argument("--round", type=int, default=None,
                   help="Optionally round embedding values to this many decimals "
                   "to shrink the output file (e.g., 5).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    # Resume support: load whatever was already written.
    results = []
    done_ids = set()
    if os.path.exists(args.output_json):
        with open(args.output_json, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["question_id"] for r in results}
        print(f"Resuming: {len(done_ids)} questions already embedded.")

    print(f"Loading {args.model} on {device} ({args.dtype}) ...")
    model = SentenceTransformer(
        args.model,
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": dtype},
    )
    model.max_seq_length = args.max_length

    def save():
        tmp = args.output_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f)
        os.replace(tmp, args.output_json)

    since_save = 0
    for rec in tqdm(data, desc="Embedding questions"):
        qid = rec.get("question_id")
        if qid in done_ids:
            continue
        texts = rec.get("generations_text") or []
        if args.mode == "final_answer":
            texts = [extract_final_answer(t) for t in texts]
        texts = [t if t else " " for t in texts]  # guard against empty strings

        embs = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        if args.round is not None:
            embs = [[round(v, args.round) for v in e] for e in embs]

        results.append({"question_id": qid, "embeddings": embs})
        since_save += 1
        if since_save >= args.save_every:
            save()
            since_save = 0

    save()
    dims = len(results[0]["embeddings"][0]) if results and results[0]["embeddings"] else 0
    print(f"Done: {len(results)} questions, embedding dim {dims} -> {args.output_json}")


if __name__ == "__main__":
    main()
