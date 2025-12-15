from huggingface_hub import snapshot_download
import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"

def download_dataset(hf_token, split="audio", max_workers=2, max_retries=5):
  for retry in max_retries:
    try:
      snapshot_download(
          repo_id="meituan-longcat/UNO-Bench",
          repo_type="dataset",
          allow_patterns=f"{split}/**",
          local_dir=f"uno-bench/{split}",
          local_dir_use_symlinks=False,
          token=hf_token,
          max_workers=max_workers
      )
      snapshot_download(
          repo_id="meituan-longcat/UNO-Bench",
          repo_type="dataset",
          allow_patterns="validation.parquet",
          local_dir=f"uno-bench/validation.parquet",
          local_dir_use_symlinks=False,
          token=hf_token,
          max_workers=max_workers
      )
      return
    except Exception as e:
      print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
  download_dataset(
  sys.argv[1],
  sys.argv[2],
  sys.argv[3],
  sys.argv[4]
  )
