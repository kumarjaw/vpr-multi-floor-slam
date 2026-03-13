import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import models

from dataset_isec import load_isec_metadata, ISECImageDataset, DEFAULT_TRANSFORM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_NPZ = PROJECT_ROOT / "isec_netvlad_descriptors.npz"
CENTERS_NPY = PROJECT_ROOT / "netvlad_centers.npy"


def build_backbone(device):
    """
    ResNet18 backbone truncated before avg pool.
    Output: [B, 512, H, W]
    """
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-2]  # keep convs only
    backbone = nn.Sequential(*modules)
    backbone.eval()
    backbone.to(device)
    return backbone


def netvlad_descriptor(feat_map, centers, alpha=100.0):
    """
    Compute NetVLAD descriptor for a single feature map.

    feat_map: [C, H, W]  (torch tensor on device)
    centers:  [K, C]     (torch tensor on device)

    Returns: [K*C] 1D tensor (L2-normalized)
    """
    C, H, W = feat_map.shape
    K = centers.size(0)

    # Flatten spatial locations: [N, C]
    x = feat_map.view(C, -1).permute(1, 0)  # [N, C], N = H*W

    # Compute soft-assignments
    x_exp = x.unsqueeze(1)        # [N, 1, C]
    c_exp = centers.unsqueeze(0)  # [1, K, C]
    dist2 = (x_exp - c_exp).pow(2).sum(dim=2)  # [N, K]

    a = torch.softmax(-alpha * dist2, dim=1)  # [N, K]

    # Residuals: (x - c_k)
    residual = x_exp - c_exp                  # [N, K, C]
    a_exp = a.unsqueeze(2)                    # [N, K, 1]

    # Aggregate
    vlad = (a_exp * residual).sum(dim=0)      # [K, C]

    # Intra-normalization
    vlad = F.normalize(vlad, p=2, dim=1)      # [K, C]

    # Flatten and L2-normalize whole vector
    vlad = vlad.view(-1)                      # [K*C]
    vlad = F.normalize(vlad, p=2, dim=0)
    return vlad


def collect_kmeans_features(backbone, loader, device, max_samples, num_clusters):
    """
    First pass: gather conv features from a subset of images,
    run KMeans to get NetVLAD cluster centers.
    """
    print(f"[KMeans] Collecting up to {max_samples} feature vectors...")
    all_feats = []
    total = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)  # [B, C, H, W]
            B, C, H, W = feats.shape

            # Flatten: [B*H*W, C]
            feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
            feats_cpu = feats.cpu().numpy().astype(np.float32)

            all_feats.append(feats_cpu)
            total += feats_cpu.shape[0]

            if total >= max_samples:
                break

    X = np.concatenate(all_feats, axis=0)
    if X.shape[0] > max_samples:
        X = X[:max_samples]

    print(f"[KMeans] Final feature matrix shape: {X.shape}")
    print(f"[KMeans] Running KMeans with K={num_clusters}...")
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.astype(np.float32)
    print(f"[KMeans] Centers shape: {centers.shape}")
    return centers


def extract_netvlad_descriptors(backbone, loader, centers, device, alpha):
    """
    Second pass: compute NetVLAD descriptor for every image.
    """
    centers_t = torch.from_numpy(centers).to(device)  # [K, C]
    K, C = centers.shape

    # Figure out total N
    dataset = loader.dataset
    N = len(dataset)
    D = K * C

    descriptors = np.zeros((N, D), dtype=np.float32)
    floor_labels = np.zeros(N, dtype=np.int64)

    idx = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            feats = backbone(imgs)  # [B, C, H, W]
            B = feats.size(0)

            for b in range(B):
                vlad = netvlad_descriptor(feats[b], centers_t, alpha=alpha)
                descriptors[idx] = vlad.cpu().numpy()
                floor_labels[idx] = int(labels[b].item())
                idx += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"[NetVLAD] Processed {idx}/{N} images...")

    print(f"[NetVLAD] Finished all {N} images.")
    return descriptors, floor_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num_clusters", type=int, default=32)
    parser.add_argument("--max_kmeans_samples", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=100.0)
    args = parser.parse_args()

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load metadata + dataset
    df = load_isec_metadata()
    dataset = ISECImageDataset(df, transform=DEFAULT_TRANSFORM)

    loader_for_kmeans = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    loader_for_descriptors = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    # Backbone
    backbone = build_backbone(device)

    # Step 1: cluster centers (or load if we already have them)
    if CENTERS_NPY.exists():
        print(f"[KMeans] Loading existing centers from {CENTERS_NPY}")
        centers = np.load(CENTERS_NPY)
    else:
        centers = collect_kmeans_features(
            backbone,
            loader_for_kmeans,
            device,
            max_samples=args.max_kmeans_samples,
            num_clusters=args.num_clusters,
        )
        np.save(CENTERS_NPY, centers)
        print(f"[KMeans] Saved centers to {CENTERS_NPY}")

    # Step 2: NetVLAD descriptors
    descriptors, floor_labels = extract_netvlad_descriptors(
        backbone,
        loader_for_descriptors,
        centers,
        device,
        alpha=args.alpha,
    )

    print(f"[NetVLAD] Descriptor matrix shape: {descriptors.shape}")
    np.savez_compressed(
        OUT_NPZ,
        descriptors=descriptors,
        floor_labels=floor_labels,
    )
    print(f"[NetVLAD] Saved descriptors to: {OUT_NPZ}")


if __name__ == "__main__":
    main()
