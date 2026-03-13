import numpy as np
from config import DESCRIPTORS_NPZ


def l2_normalize(x, axis=1, eps=1e-8):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def main():
    data = np.load(DESCRIPTORS_NPZ)
    descriptors = data["descriptors"]  # (N, 512)
    labels = data["floor_labels"]      # (N,)

    print("Loaded descriptors:", descriptors.shape)
    print("Loaded labels:", labels.shape)

    # 1. L2-normalize for cosine similarity
    desc_norm = l2_normalize(descriptors, axis=1)

    # 2. Cosine similarity between all pairs: sim[i, j] = <d_i, d_j>
    print("Computing similarity matrix (this might take a moment)...")
    sim = np.dot(desc_norm, desc_norm.T)  # (N, N)

    # Don't let an image retrieve itself
    np.fill_diagonal(sim, -np.inf)

    # 3. For each image, find top-1 nearest neighbor
    nn_indices = np.argmax(sim, axis=1)          # (N,)
    nn_labels = labels[nn_indices]               # predicted floor by NN
    true_labels = labels

    # 4. Overall top-1 same-floor accuracy
    top1_same_floor = np.mean(nn_labels == true_labels)

    # 5. Inter-floor confusion
    floor2_mask = true_labels == 2
    floor5_mask = true_labels == 5

    # For queries from 2nd floor, how often does NN come from 5th?
    f2_confusion = np.mean(nn_labels[floor2_mask] == 5)

    # For queries from 5th floor, how often does NN come from 2nd?
    f5_confusion = np.mean(nn_labels[floor5_mask] == 2)

    # 6. Print stats
    print("\n===== Baseline VPR Evaluation (ResNet-18 descriptors) =====")
    print(f"Total images: {len(true_labels)}")
    print(f"Top-1 same-floor accuracy: {top1_same_floor * 100:.2f}%")
    print(f"2nd floor -> mis-matched as 5th: {f2_confusion * 100:.2f}% of 2nd floor queries")
    print(f"5th floor -> mis-matched as 2nd: {f5_confusion * 100:.2f}% of 5th floor queries")

    print("\nInterpretation:")
    print("- Higher top-1 accuracy = better place recognition.")
    print("- High 2nd->5th or 5th->2nd confusion = strong perceptual aliasing between floors.")


if __name__ == "__main__":
    main()
