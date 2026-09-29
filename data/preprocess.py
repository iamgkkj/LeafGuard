#!/usr/bin/env python3
"""
Preprocess + inspect PlantVillage dataset
- Discovers ImageFolder structure, splits, class stats, mean/std estimation
- Generates data/classes.json if needed
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def scan_dataset(data_dir: Path):
    data_dir = Path(data_dir)
    # Try common subpaths
    candidates = [
        data_dir,
        data_dir / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "train",
        data_dir / "New Plant Diseases Dataset(Augmented)" / "train",
        data_dir / "train",
        data_dir / "valid",
    ]
    for cand in candidates:
        if cand.exists():
            # count class folders
            subdirs = [d for d in cand.iterdir() if d.is_dir()]
            if len(subdirs) >= 10:
                return cand
    return data_dir


def count_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
    counter = Counter()
    total = 0
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        n = sum(1 for f in cls_dir.iterdir() if f.suffix in exts)
        counter[cls_dir.name] = n
        total += n
    return counter, total


def estimate_mean_std(root: Path, num_samples: int = 500):
    """Estimate mean/std on random subset (fast)."""
    exts = {".jpg", ".jpeg", ".png"}
    all_files = []
    for cls_dir in root.iterdir():
        if cls_dir.is_dir():
            all_files.extend([p for p in cls_dir.iterdir() if p.suffix.lower() in exts])
    if not all_files:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    random.seed(0)
    sample = random.sample(all_files, min(num_samples, len(all_files)))
    means = []
    stds = []
    for p in sample:
        try:
            img = Image.open(p).convert("RGB").resize((224, 224))
            arr = np.array(img).astype(np.float32) / 255.0  # HWC 0-1
            means.append(arr.mean(axis=(0, 1)))
            stds.append(arr.std(axis=(0, 1)))
        except Exception:
            continue
    if not means:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean = np.mean(means, axis=0).tolist()
    std = np.mean(stds, axis=0).tolist()
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Preprocess / inspect PlantVillage dataset")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent), help="dataset root")
    parser.add_argument("--estimate-stats", action="store_true", help="estimate mean/std")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[preprocess] Scanning {data_dir}")
    root = scan_dataset(data_dir)
    print(f"[preprocess] Using root: {root}")

    if not root.exists():
        print(f"[preprocess] Path does not exist: {root}")
        return

    counter, total = count_images(root)
    print(f"[preprocess] Total images: {total}")
    print(f"[preprocess] Classes found: {len(counter)}")
    for cls, n in sorted(counter.items()):
        print(f"  {cls}: {n}")

    # check missing vs expected
    expected = set(DEFAULT_CLASSES)
    found = set(counter.keys())
    missing = expected - found
    extra = found - expected
    if missing:
        print(f"[preprocess] Missing expected classes ({len(missing)}): {missing}")
    if extra:
        print(f"[preprocess] Extra classes ({len(extra)}): {extra}")

    if args.estimate_stats:
        mean, std = estimate_mean_std(root)
        print(f"[preprocess] Estimated mean: {mean}")
        print(f"[preprocess] Estimated std: {std}")

    # save class metadata
    out = Path(__file__).parent / "classes.json"
    if not out.exists():
        with open(out, "w") as f:
            json.dump({"classes": DEFAULT_CLASSES, "num_classes": len(DEFAULT_CLASSES)}, f, indent=2)
        print(f"[preprocess] Wrote {out}")


if __name__ == "__main__":
    main()
