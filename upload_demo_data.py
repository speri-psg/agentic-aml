"""
upload_demo_data.py — Upload ARIA demo data to HF Hub dataset.

Run once (or after data regeneration) before building Docker images:

    python upload_demo_data.py --token <HF_TOKEN>
    python upload_demo_data.py               # reads HF_TOKEN env var

Uploads to speri420/aria-demo-data (public dataset):
  aria_synth_xl/   the full bank-scale dataset (100K customers, 21 rules
                   incl. modern typology, network features pre-computed)
                   — used by docker/Dockerfile.app at build time so the
                   resulting image is self-contained and runnable on a
                   user machine with no extra downloads.

Excludes:
  .cache/          pickled DataFrames built at first run (regenerable)
  __pycache__/     Python bytecode

Cleanup (default ON): deletes the legacy aria_synth/ folder + the two
legacy docs/*.csv files from the dataset root before uploading. This
keeps the dataset size down so docker builds don't re-pull the OLD 5K
data alongside the new XL data. Pass --no-cleanup to skip.
"""

import argparse
import os
from pathlib import Path

HERE = Path(__file__).parent
REPO = "speri420/aria-demo-data"

# Upload target: aria_synth_xl/ at repo root.
LOCAL_DIR  = HERE / "aria_synth_xl"
REMOTE_DIR = "aria_synth_xl"

# Legacy paths to remove from the dataset root (one-time migration).
LEGACY_PATHS = [
    "aria_synth",                            # old 5K data folder
    "docs/ds_segmentation_synth.csv",        # moved into aria_synth_xl/
    "docs/customer_cluster_labels.csv",      # moved into aria_synth_xl/
]


def main():
    parser = argparse.ArgumentParser(description="Upload ARIA demo data to HF Hub")
    parser.add_argument("--token",      default=os.environ.get("HF_TOKEN"), help="HF token")
    parser.add_argument("--repo",       default=REPO,  help="HF dataset repo ID")
    parser.add_argument("--private",    action="store_true", help="Create repo as private")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip removing legacy paths")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("No HF token provided. Pass --token or set HF_TOKEN env var.")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    print(f"Repo: {args.repo}")
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True, private=args.private)

    # ── Cleanup legacy paths ──────────────────────────────────────────────
    if not args.no_cleanup:
        for legacy in LEGACY_PATHS:
            try:
                api.delete_folder(path_in_repo=legacy, repo_id=args.repo, repo_type="dataset")
                print(f"  removed legacy folder: {legacy}")
            except Exception:
                # delete_folder may not exist (older versions) or path may already be gone;
                # try delete_file as a fallback for single-file paths.
                try:
                    api.delete_file(path_in_repo=legacy, repo_id=args.repo, repo_type="dataset")
                    print(f"  removed legacy file:   {legacy}")
                except Exception:
                    pass  # already gone or never existed — fine

    # ── Upload XL data ────────────────────────────────────────────────────
    if not LOCAL_DIR.exists():
        raise SystemExit(f"Local dir {LOCAL_DIR} not found — generate it via aria_synth_xl/build.py first.")

    print(f"  uploading {LOCAL_DIR} -> {REMOTE_DIR} (this may take a while; ~6.6 GB) ...")
    api.upload_folder(
        folder_path=str(LOCAL_DIR),
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=REMOTE_DIR,
        ignore_patterns=[
            ".cache/**",
            "__pycache__/**",
            "*.pyc",
        ],
    )
    print("    done.")

    print(f"\nDataset: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
