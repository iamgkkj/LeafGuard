#!/usr/bin/env python3
"""
Download New Plant Diseases Dataset (Kaggle, 87k images, 38 classes)
- Primary: kagglehub (kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset"))
- Fallback: instruct manual download if credentials missing
"""
import argparse
import json
import os
import sys
from pathlib import Path


DATASET_SLUG = "vipoooool/new-plant-diseases-dataset"
LOCAL_CLASSES = Path(__file__).parent / "classes.json"


def try_kagglehub_download(output_dir: Path):
    try:
        import kagglehub  # type: ignore
        print(f"[download] Attempting kagglehub download: {DATASET_SLUG}")
        # This will download to kagglehub cache; we then symlink/copy
        cache_path = kagglehub.dataset_download(DATASET_SLUG)
        print(f"[download] Cached at: {cache_path}")
        # The dataset contains 'New Plant Diseases Dataset(Augmented)' folder
        # Move/copy to output_dir if different
        src = Path(cache_path)
        # Find inner dataset folder
        candidates = list(src.rglob("*.jpg"))
        print(f"[download] Found {len(candidates)} images in cache")
        if src != output_dir:
            print(f"[download] Dataset available at {src}")
            print(f"[download] Suggested: set --data-dir to {src} or copy contents")
        return src
    except Exception as e:
        print(f"[download] kagglehub failed: {e}")
        return None


def manual_instructions(output_dir: Path):
    print("\n=== Manual Download Instructions ===")
    print("1. Go to https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print("2. Click 'Download' (requires Kaggle account) or use Kaggle API:")
    print("   pip install kaggle")
    print("   kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p ./data --unzip")
    print(f"3. Unzip so that {output_dir}/New Plant Diseases Dataset(Augmented)/train/ exists")
    print("   Expected structure:")
    print("   data/New Plant Diseases Dataset(Augmented)/")
    print("     train/  (38 class folders)")
    print("     valid/  (38 class folders)")
    print("   Alternative PlantVillage structure:")
    print("   data/plantvillage/ with 38 folders also works via --data-dir")
    print("\nClasses expected (38):")
    try:
        with open(LOCAL_CLASSES) as f:
            data = json.load(f)
            for c in data["classes"]:
                print(f"  - {c}")
    except Exception:
        pass
    print("====================================\n")


def main():
    parser = argparse.ArgumentParser(description="Download New Plant Diseases Dataset")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent), help="output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = try_kagglehub_download(output_dir)
    if result is None:
        manual_instructions(output_dir)
        print("[download] No dataset downloaded. Follow manual instructions above.")
        sys.exit(1)
    else:
        print("[download] Done. Use this path as --data-dir for train.py / preprocess.py")
        print(f"[download] Path: {result}")


if __name__ == "__main__":
    main()
