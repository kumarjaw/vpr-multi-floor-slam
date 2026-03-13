import argparse
import numpy as np


def compute_cosine_sim_matrix(X, Y):
    """
    X: [N, D], Y: [M, D]
    Returns: [N, M] cosine similarity matrix.
    """
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    # Numerical safety
    Xn[np.isnan(Xn)] = 0.0
    Yn[np.isnan(Yn)] = 0.0

    return Xn @ Yn.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Path to descriptors npz")
    args = parser.parse_args()

    data = np.load(args.npz)
    print(f"Loaded NPZ from: {args.npz}")
    print(f"Keys in NPZ: {list(data.keys())}")

    desc = data["descriptors"]      # [N, D]
    labels = None

    # Use 'floor_labels' if available; fall back to 'labels'
    if "floor_labels" in data:
        labels = data["floor_labels"]
        print("Using 'floor_labels' from NPZ.")
    elif "labels" in data:
        labels = data["labels"]
        print("Using 'labels' from NPZ.")
    else:
        raise KeyError("NPZ file must contain 'floor_labels' or 'labels'.")

    print(f"Descriptors shape: {desc.shape}")
    print(f"Labels shape: {labels.shape}")

    # Split by floor
    floor2_idx = np.where(labels == 2)[0]
    floor5_idx = np.where(labels == 5)[0]

    desc2 = desc[floor2_idx]
    desc5 = desc[floor5_idx]

    print(f"Floor 2 images: {len(desc2)}")
    print(f"Floor 5 images: {len(desc5)}")

    # --------------------------------------------------------
    # Same-floor similarity stats (for context)
    # --------------------------------------------------------
    print("\nComputing same-floor cosine similarity stats for context...")

    sims_2_2 = compute_cosine_sim_matrix(desc2, desc2)
    sims_5_5 = compute_cosine_sim_matrix(desc5, desc5)

    # Remove diagonal (self-similarity = 1)
    mask_2 = ~np.eye(sims_2_2.shape[0], dtype=bool)
    mask_5 = ~np.eye(sims_5_5.shape[0], dtype=bool)

    sims_2_2_vals = sims_2_2[mask_2]
    sims_5_5_vals = sims_5_5[mask_5]

    mu_2 = sims_2_2_vals.mean()
    mu_5 = sims_5_5_vals.mean()

    # --------------------------------------------------------
    # Cross-floor similarity stats
    # --------------------------------------------------------
    print("\nComputing cross-floor cosine similarity matrices...")
    sims_2_5 = compute_cosine_sim_matrix(desc2, desc5)  # [N2, N5]
    sims_5_2 = sims_2_5.T                              # [N5, N2]

    sims_2_5_vals = sims_2_5.flatten()
    sims_5_2_vals = sims_5_2.flatten()

    mu_cross = 0.5 * (sims_2_5_vals.mean() + sims_5_2_vals.mean())

    print("\n=== Similarity Stats (cosine) ===")
    print(f"Mean same-floor (2nd): {mu_2:.4f}")
    print(f"Mean same-floor (5th): {mu_5:.4f}")
    print(f"Mean cross-floor (2↔5): {mu_cross:.4f}")

    # --------------------------------------------------------
    # Adaptive thresholds between cross-floor and same-floor means
    # --------------------------------------------------------
    low = min(mu_2, mu_5, mu_cross)
    high = max(mu_2, mu_5, mu_cross)

    # If everything is almost the same, add a small margin
    if abs(high - low) < 1e-3:
        low -= 0.05
        high += 0.05

    thresholds = np.linspace(low, high, 8)

    print("\n===== Cross-floor False Positive Rate (FPR) =====")
    print("Threshold  |  FPR 2→5 (2nd floor queries)  |  FPR 5→2 (5th floor queries)")
    print("--------------------------------------------------------------------")

    for thr in thresholds:
        # For each 2nd floor query, does any 5th floor image exceed thr?
        # That is a cross-floor false positive in SLAM context.
        fp_2_5 = (sims_2_5 > thr).any(axis=1).mean() * 100.0
        fp_5_2 = (sims_5_2 > thr).any(axis=1).mean() * 100.0

        print(f"  {thr:6.3f}  |         {fp_2_5:5.2f}%              |          {fp_5_2:5.2f}%")

    # Optionally save matrices for later plotting
    # (overwrite the same file name each time)
    np.savez_compressed(
        "../cross_floor_similarity_stats.npz",
        sims_2_5=sims_2_5,
        sims_5_2=sims_5_2,
        sims_2_2=sims_2_2,
        sims_5_5=sims_5_5,
        thresholds=thresholds,
    )
    print("\nSaved similarity matrices to: ../cross_floor_similarity_stats.npz")


if __name__ == "__main__":
    main()
