import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_descriptors_and_labels(npz_path: Path, metadata_csv: Path | None = None):
    """
    Load descriptors and floor labels.

    Priority:
      1) 'labels' in NPZ
      2) 'floor_labels' in NPZ
      3) 'floor_label' column from metadata CSV (if given)
    """
    data = np.load(npz_path)
    print(f"Loaded NPZ from: {npz_path}")
    print("Keys in NPZ:", list(data.keys()))

    descriptors = data["descriptors"]
    labels = None

    if "labels" in data:
        labels = data["labels"]
        print("Using 'labels' from NPZ.")
    elif "floor_labels" in data:
        labels = data["floor_labels"]
        print("Using 'floor_labels' from NPZ.")
    else:
        if metadata_csv is None:
            raise ValueError("No labels in NPZ and no metadata CSV provided.")
        df = pd.read_csv(metadata_csv)
        if "floor_label" not in df.columns:
            raise ValueError(
                f"'floor_label' column not found in metadata CSV: {metadata_csv}"
            )
        labels = df["floor_label"].values
        print(f"Loaded labels from metadata CSV: {metadata_csv}")

    labels = labels.astype(int)
    print(f"Descriptors shape: {descriptors.shape}")
    print(f"Labels shape: {labels.shape}")
    return descriptors, labels


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_hard_triplets(descriptors: np.ndarray, labels: np.ndarray):
    """
    For each anchor i:
      - pick the *most similar* same-floor image as positive
      - pick the *most similar* other-floor image as hard negative

    Returns three arrays:
      anchors_idx, positives_idx, negatives_idx  (each shape: (M,))
    """
    N, D = descriptors.shape
    print(f"Building triplets from {N} descriptors of dim {D}...")

    # Cosine similarity because descriptors are normalized.
    # Precompute full similarity matrix for simplicity (N x N, ~1.7M elements).
    print("Computing full similarity matrix (N x N)...")
    sims = descriptors @ descriptors.T  # (N, N)

    anchors = []
    positives = []
    negatives = []

    for i in range(N):
        floor_i = labels[i]

        # Same-floor candidates (excluding self)
        same_mask = (labels == floor_i)
        same_mask[i] = False
        same_indices = np.where(same_mask)[0]

        # Other-floor candidates (potential negatives)
        other_indices = np.where(labels != floor_i)[0]

        if len(same_indices) == 0 or len(other_indices) == 0:
            # Shouldn't happen here, but be safe
            continue

        # Most similar same-floor image (hard positive)
        same_sims = sims[i, same_indices]
        pos_idx = same_indices[np.argmax(same_sims)]

        # Most similar other-floor image (hard negative)
        other_sims = sims[i, other_indices]
        neg_idx = other_indices[np.argmax(other_sims)]

        anchors.append(i)
        positives.append(pos_idx)
        negatives.append(neg_idx)

    anchors = np.array(anchors, dtype=np.int64)
    positives = np.array(positives, dtype=np.int64)
    negatives = np.array(negatives, dtype=np.int64)

    print(f"Built {len(anchors)} triplets.")
    return anchors, positives, negatives


def main():
    parser = argparse.ArgumentParser(
        description="Build hard triplets (anchor, positive, negative) for metric learning."
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="../isec_resnet18_descriptors.npz",
        help="Path to NPZ with 'descriptors' and labels.",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="../isec_2f_5f_metadata.csv",
        help="Path to metadata CSV if labels are not in NPZ.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../isec_triplets_resnet18.npz",
        help="Output NPZ for triplets.",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    metadata_path = Path(args.metadata_csv)
    if not metadata_path.exists():
        print(f"Metadata CSV not found at {metadata_path}, will use NPZ labels only.")

    # 1) Load descriptors and labels
    descriptors, labels = load_descriptors_and_labels(
        npz_path, metadata_csv=metadata_path if metadata_path.exists() else None
    )

    # 2) Normalize descriptors
    descriptors = l2_normalize_rows(descriptors)

    # 3) Build hard triplets
    anchors, positives, negatives = build_hard_triplets(descriptors, labels)

    # 4) Save
    out_path = Path(args.out)
    np.savez(
        out_path,
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        labels=labels,
    )
    print(f"Saved triplets to: {out_path}")


if __name__ == "__main__":
    main()
