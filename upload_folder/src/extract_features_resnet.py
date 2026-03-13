import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

from dataset_isec import load_isec_metadata, ISECFloorDataset
from config import DESCRIPTORS_NPZ


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU 💻🔥")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def build_feature_extractor(device):
    """
    Use ResNet-18 pretrained on ImageNet, drop the final FC layer.
    Output: 512-D descriptor per image.
    """
    model = models.resnet18(pretrained=True)
    # Remove the final FC layer -> global pooled feature of size 512
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def main():
    device = get_device()

    # 1. Load metadata
    df = load_isec_metadata(save_merged=True)

    # 2. Define image transforms
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # 3. Build dataset and dataloader
    dataset = ISECFloorDataset(df, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # set >0 if you want speedup
    )

    # 4. Build feature extractor
    feature_extractor = build_feature_extractor(device)

    all_feats = []
    all_labels = []

    # 5. Loop through all images
    with torch.no_grad():
        for batch_idx, (images, floor_labels) in enumerate(loader):
            images = images.to(device)

            feats = feature_extractor(images)  # (B, 512, 1, 1)
            feats = feats.view(feats.size(0), -1)  # (B, 512)

            all_feats.append(feats.cpu().numpy())
            all_labels.append(floor_labels.numpy())

            if (batch_idx + 1) % 20 == 0:
                print(f"Processed { (batch_idx + 1) * loader.batch_size } images...")

    # 6. Stack everything
    descriptors = np.vstack(all_feats).astype(np.float32)
    floor_labels = np.concatenate(all_labels).astype(np.int32)

    print("Descriptor matrix shape:", descriptors.shape)  # (N, 512)

    # 7. Save to disk
    np.savez(
        DESCRIPTORS_NPZ,
        descriptors=descriptors,
        floor_labels=floor_labels,
    )

    print(f"Saved descriptors to: {DESCRIPTORS_NPZ}")
    print("Feature extraction DONE ✅")


if __name__ == "__main__":
    main()
