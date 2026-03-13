# plot_similarity_histograms.py
import argparse
import numpy as np
import matplotlib.pyplot as plt

def compute_cosine_sim(A, B):
    # A: (N, D), B: (M, D)
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T  # (N, M)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to descriptors npz (with 'descriptors' and 'floor_labels').")
    parser.add_argument("--out", type=str, default="similarity_histograms.png",
                        help="Output PNG filename.")
    args = parser.parse_args()

    data = np.load(args.npz)
    X = data["descriptors"]         # (N, D)
    y = data["floor_labels"]        # (N,)

    idx2 = np.where(y == 2)[0]
    idx5 = np.where(y == 5)[0]
    X2 = X[idx2]
    X5 = X[idx5]

    # Same-floor similarities (upper triangle only to avoid duplicates)
    sim2 = compute_cosine_sim(X2, X2)
    sim5 = compute_cosine_sim(X5, X5)
    same2 = sim2[np.triu_indices_from(sim2, k=1)]
    same5 = sim5[np.triu_indices_from(sim5, k=1)]

    # Cross-floor similarities (all pairs)
    sim25 = compute_cosine_sim(X2, X5).ravel()

    plt.figure(figsize=(7,5))
    bins = 40

    plt.hist(same2, bins=bins, alpha=0.5, label="Same floor (2nd)")
    plt.hist(same5, bins=bins, alpha=0.5, label="Same floor (5th)")
    plt.hist(sim25, bins=bins, alpha=0.5, label="Cross-floor (2nd↔5th)")

    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("Similarity distributions (before/after metric learning)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved histogram to: {args.out}")

if __name__ == "__main__":
    main()
