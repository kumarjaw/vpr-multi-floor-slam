import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TripletIndexDataset(Dataset):
    """
    Wraps (anchors, positives, negatives) + descriptor matrix.
    """

    def __init__(self, descriptors: np.ndarray, anchors, positives, negatives):
        self.X = torch.from_numpy(descriptors).float()  # (N, D)
        self.anchors = torch.from_numpy(anchors).long()
        self.positives = torch.from_numpy(positives).long()
        self.negatives = torch.from_numpy(negatives).long()

    def __len__(self):
        return self.anchors.shape[0]

    def __getitem__(self, idx):
        a_idx = self.anchors[idx].item()
        p_idx = self.positives[idx].item()
        n_idx = self.negatives[idx].item()
        return a_idx, p_idx, n_idx


class MetricHead(nn.Module):
    """
    Simple linear projection + L2 normalization.

    Input dim: 512 (ResNet-18 descriptors)
    Output dim: 128 (can be changed with --out_dim)
    """

    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        z = self.fc(x)  # (B, out_dim)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-12)
        return z


def main():
    parser = argparse.ArgumentParser(
        description="Train a metric head with triplet loss on ISEC descriptors."
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="../isec_resnet18_descriptors.npz",
        help="NPZ with 'descriptors' and labels.",
    )
    parser.add_argument(
        "--triplets",
        type=str,
        default="../isec_triplets_resnet18.npz",
        help="NPZ with 'anchors', 'positives', 'negatives'.",
    )
    parser.add_argument(
        "--out_npz",
        type=str,
        default="../isec_resnet18_metric_head_descriptors.npz",
        help="NPZ to save transformed descriptors.",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="metric_head_linear.pth",
        help="Path to save metric head weights.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' or 'cuda' (if you have GPU drivers set up).",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1) Load descriptors + labels
    npz_data = np.load(args.npz)
    descriptors = npz_data["descriptors"]          # (N, 512)
    if "floor_labels" in npz_data:
        labels = npz_data["floor_labels"]
    elif "labels" in npz_data:
        labels = npz_data["labels"]
    else:
        raise ValueError("No labels found in descriptors NPZ.")

    N, D = descriptors.shape
    print(f"Loaded descriptors: {descriptors.shape}")
    print(f"Loaded labels: {labels.shape}")

    # 2) Normalize input descriptors (good practice before learning)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-12
    descriptors_norm = descriptors / norms

    # 3) Load triplets
    trip_data = np.load(args.triplets)
    anchors = trip_data["anchors"]
    positives = trip_data["positives"]
    negatives = trip_data["negatives"]

    print(f"Loaded triplets: {anchors.shape[0]}")

    dataset = TripletIndexDataset(descriptors_norm, anchors, positives, negatives)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4) Build model + loss + optimizer
    model = MetricHead(in_dim=D, out_dim=args.out_dim).to(device)
    criterion = nn.TripletMarginLoss(margin=0.3, p=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5) Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for a_idx, p_idx, n_idx in loader:
            a_idx = a_idx.to(device)
            p_idx = p_idx.to(device)
            n_idx = n_idx.to(device)

            # Gather descriptors
            X_all = dataset.X.to(device)
            anc = X_all[a_idx]  # (B, D)
            pos = X_all[p_idx]
            neg = X_all[n_idx]

            # Forward
            z_anc = model(anc)
            z_pos = model(pos)
            z_neg = model(neg)

            loss = criterion(z_anc, z_pos, z_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch:02d}/{args.epochs} - Avg Triplet Loss: {avg_loss:.4f}")

    # 6) Save model
    torch.save(model.state_dict(), args.model_out)
    print(f"Saved metric head weights to: {args.model_out}")

    # 7) Transform ALL descriptors with the learned head
    model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(descriptors_norm).float().to(device)
        Z_all = model(X_all)  # (N, out_dim)
        Z_all = Z_all.cpu().numpy()

    print(f"New descriptor matrix shape: {Z_all.shape}")

    # 8) Save new descriptors NPZ (same labels)
    out_npz_path = Path(args.out_npz)
    np.savez(
        out_npz_path,
        descriptors=Z_all,
        floor_labels=labels,
    )
    print(f"Saved transformed descriptors to: {out_npz_path}")


if __name__ == "__main__":
    main()
