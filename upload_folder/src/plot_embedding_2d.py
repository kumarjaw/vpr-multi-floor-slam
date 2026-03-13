# plot_embedding_2d.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to descriptors npz (with 'descriptors' and 'floor_labels').")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"],
                        help="Dimensionality reduction method.")
    parser.add_argument("--out", type=str, default="embedding_2d.png",
                        help="Output PNG filename.")
    args = parser.parse_args()

    data = np.load(args.npz)
    X = data["descriptors"]
    y = data["floor_labels"]

    if args.method == "pca":
        reducer = PCA(n_components=2)
        Z = reducer.fit_transform(X)
        title = "PCA embedding of descriptors"
    else:
        reducer = TSNE(n_components=2, perplexity=30, init="random", learning_rate="auto")
        Z = reducer.fit_transform(X)
        title = "t-SNE embedding of descriptors"

    # Split by floor for plotting
    Z2 = Z[y == 2]
    Z5 = Z[y == 5]

    plt.figure(figsize=(6,6))
    plt.scatter(Z2[:,0], Z2[:,1], s=10, alpha=0.6, label="2nd floor")
    plt.scatter(Z5[:,0], Z5[:,1], s=10, alpha=0.6, label="5th floor")
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved embedding plot to: {args.out}")

if __name__ == "__main__":
    main()
