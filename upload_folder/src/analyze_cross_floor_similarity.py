import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, METADATA_CSV


def load_descriptors(npz_path: Path):
    data = np.load(npz_path)
    if "descriptors" not in data or "floor_labels" not in data:
        raise ValueError(f"NPZ {npz_path} must contain 'descriptors' and 'floor_labels'")
    desc = data["descriptors"].astype(np.float32)
    labels = data["floor_labels"].astype(int)
    # L2-normalize
    norms = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8
    desc = desc / norms
    return desc, labels


def main():
    parser = argparse.ArgumentParser(
        description="Show top cross-floor confusing pairs: 2nd-floor queries vs 5th-floor DB."
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=str(PROJECT_ROOT / "isec_resnet18_descriptors.npz"),
        help="Path to descriptor NPZ file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="How many most similar cross-floor pairs to print.",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    print(f"Using NPZ: {npz_path}")
    desc, labels = load_descriptors(npz_path)
    df = pd.read_csv(METADATA_CSV)

    if len(df) != desc.shape[0]:
        print(f"WARNING: metadata rows ({len(df)}) != descriptors ({desc.shape[0]}). "
              "Make sure you used the same ordering when extracting features.")

    # Indices for each floor
    idx2 = np.where(labels == 2)[0]
    idx5 = np.where(labels == 5)[0]
    print(f"Floor 2 images: {len(idx2)}")
    print(f"Floor 5 images: {len(idx5)}\n")

    # Slice descriptors
    D2 = desc[idx2]  # (N2, D)
    D5 = desc[idx5]  # (N5, D)

    # Cross-floor cosine similarity matrix: (N2 x N5)
    print("Computing cross-floor cosine similarity matrix (2nd as queries, 5th as database)...\n")
    S_2to5 = D2 @ D5.T

    # Build list of (sim, global_idx_query, global_idx_db)
    pairs = []
    N2, N5 = S_2to5.shape
    for i in range(N2):
        for j in range(N5):
            sim = float(S_2to5[i, j])
            pairs.append((sim, idx2[i], idx5[j]))

    # Sort descending by similarity
    pairs.sort(key=lambda x: x[0], reverse=True)
    top_k = min(args.top_k, len(pairs))

    print("=== Top cross-floor confusing pairs (2nd floor query -> 5th floor match) ===\n")
    for rank in range(top_k):
        sim, qi, di = pairs[rank]
        q_row = df.iloc[qi]
        d_row = df.iloc[di]

        print(f"Rank {rank + 1}")
        print(f"  Similarity: {sim:.4f}")
        print(
            f"  Query (2nd floor):   {q_row['filename']}   | path: {q_row['image_path']}"
        )
        print(
            f"  Match (5th floor):   {d_row['filename']}   | path: {d_row['image_path']}"
        )
        print()


if __name__ == "__main__":
    main()
