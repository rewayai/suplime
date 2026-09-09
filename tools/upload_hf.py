#!/usr/bin/env python
"""Upload the hf/ staging directory to the Hugging Face Hub.

    python tools/upload_hf.py --repo rewayai/suplime [--private] [--dir hf]

Needs a write token: `hf auth login` (or HF_TOKEN in the environment).
Uploads config.yaml, README.md, LICENSE, segmentation/, embedding/ and
reproducible_research/ as one commit; re-running only pushes changed files.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. rewayai/suplime")
    ap.add_argument("--dir", type=Path, default=Path("hf"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--message", default="Upload suplime pipeline")
    a = ap.parse_args()

    for required in ("config.yaml", "README.md", "segmentation/pytorch_model.bin", "embedding/pytorch_model.bin"):
        if not (a.dir / required).exists():
            raise SystemExit(f"missing {a.dir / required}")

    api = HfApi()
    # create_repo's `private` applies only when it actually creates the repo: with
    # exist_ok=True it swallows the 409 and the flag is silently ignored, so a re-upload
    # meant to be private would push new weights straight onto a public repo.
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    if a.private:
        api.update_repo_settings(repo_id=a.repo, repo_type="model", private=True)
        print(f"[upload] {a.repo} confirmed private")
    else:
        info = api.repo_info(a.repo, repo_type="model")
        if not info.private:
            print(f"[upload] NOTE: {a.repo} is PUBLIC — this upload is immediately visible")
    info = api.upload_folder(
        folder_path=str(a.dir),
        repo_id=a.repo,
        repo_type="model",
        commit_message=a.message,
        ignore_patterns=["**/__pycache__/**", "*.pyc", ".DS_Store"],
    )
    print(f"uploaded -> https://huggingface.co/{a.repo}  ({info})")


if __name__ == "__main__":
    main()
