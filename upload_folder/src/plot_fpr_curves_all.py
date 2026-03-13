import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------------
# config: paths to the three descriptor npz files
# -------------------------------------------------------------------
DESCRIPTOR_FILES = {
    "ResNet18": "../isec_resnet18_descriptors.npz",
    "ResNet18+MetricHead": "../isec_resnet18_metric_head_descriptors.npz",
    "NetVLAD": "../isec_netvlad_descriptors.npz",
}


def load_desc_and_labels(path):
    data = np.load(path)
    desc = data["descriptors"].astype(np.float32)
    # labels can be 'labels' or 'floor_labels' depending on file
    if "floor_labels" in data:
        labels = data["floor_labels"].astype(int)
    else:
        labels = data["labels"].astype(int)
    return desc, labels


def normalize_feats(feats):
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    return feats / norms


def compute_similarity_matrix(feats):
    # feats are assumed L2-normalized
    return feats @ feats.T


def compute_fpr_and_recall(sim_mat, labels, thresholds):
    """
    Robotics-style definition:
      - For each query i, if ANY same-floor image j (j!=i) has sim >= t, that query counts as a "true positive" at t.
      - If ANY cross-floor image j has sim >= t, that query counts as a "false positive" at t.
    So we compute:
      recall(t) = (#queries with at least one same-floor >=t) / N
      FPR(t)    = (#queries with at least one cross-floor >=t) / N
    """
    labels = labels.astype(int)
    N = len(labels)

    # same-floor mask (excluding self), shape [N, N]
    same_mask = labels[:, None] == labels[None, :]
    np.fill_diagonal(same_mask, False)
    cross_mask = ~same_mask

    fpr_list = []
    recall_list = []

    for t in thresholds:
        above_t = sim_mat >= t

        same_above = above_t & same_mask
        cross_above = above_t & cross_mask

        # queries that have at least one same-floor match >= t
        same_hits = same_above.any(axis=1)
        # queries that have at least one cross-floor match >= t
        cross_hits = cross_above.any(axis=1)

        recall_t = same_hits.mean()
        fpr_t = cross_hits.mean()

        recall_list.append(recall_t)
        fpr_list.append(fpr_t)

    return np.array(fpr_list), np.array(recall_list)


def main():
    # 1) Load everything, compute similarity matrices, track global min/max similarity
    sim_mats = {}
    label_dict = {}
    min_sim = 1e9
    max_sim = -1e9

    print("Loading descriptor sets and computing similarity matrices...")
    for name, path in DESCRIPTOR_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing file for {name}: {path}")
            continue

        desc, labels = load_desc_and_labels(path)
        desc = normalize_feats(desc)
        S = compute_similarity_matrix(desc)

        sim_mats[name] = S
        label_dict[name] = labels

        min_sim = min(min_sim, float(S.min()))
        max_sim = max(max_sim, float(S.max()))

        print(f"  {name}: desc shape = {desc.shape}, sim range = [{S.min():.3f}, {S.max():.3f}]")

    # 2) Build a common range of thresholds
    # Use a slightly padded range so we cover all regimes
    t_min = min_sim - 0.01
    t_max = max_sim + 0.01
    thresholds = np.linspace(t_min, t_max, 100)

    # 3) Compute FPR + recall for each descriptor set
    results = {}
    for name in sim_mats.keys():
        print(f"Computing FPR/recall curve for {name} ...")
        S = sim_mats[name]
        labels = label_dict[name]
        fpr, rec = compute_fpr_and_recall(S, labels, thresholds)
        results[name] = (fpr, rec)

    # 4) Plot FPR vs threshold
    plt.figure(figsize=(8, 5))
    for name, (fpr, rec) in results.items():
        plt.plot(thresholds, fpr, label=name)
    plt.xlabel("Similarity threshold")
    plt.ylabel("Cross-floor false positive rate")
    plt.title("Cross-floor FPR vs threshold (2nd↔5th floor)")
    plt.grid(True)
    plt.legend()
    out_path = "../fpr_curves_all.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved FPR curves plot to: {out_path}")

    # 5) (Optional) plot FPR vs recall (DET-style)
    plt.figure(figsize=(6, 5))
    for name, (fpr, rec) in results.items():
        plt.plot(rec, fpr, marker="o", linestyle="-", label=name)
    plt.xlabel("Recall (same-floor loop closures)")
    plt.ylabel("Cross-floor FPR")
    plt.title("FPR vs recall (loop closure safety trade-off)")
    plt.grid(True)
    plt.legend()
    out_path2 = "../fpr_vs_recall_all.png"
    plt.tight_layout()
    plt.savefig(out_path2, dpi=200)
    print(f"Saved FPR vs recall plot to: {out_path2}")


if __name__ == "__main__":
    main()
