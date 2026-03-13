import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FLOOR2_DIR = PROJECT_ROOT / "2nd floor" / "floor2_extracted"
FLOOR5_DIR = PROJECT_ROOT / "5th floor" / "floor5_extracted"

FLOOR2_CSV = FLOOR2_DIR / "floor2_bag42.csv"
FLOOR5_CSV = FLOOR5_DIR / "floor5_bag9.csv"

FLOOR2_IMG_DIR = FLOOR2_DIR / "floor2_bag42"
FLOOR5_IMG_DIR = FLOOR5_DIR / "floor5_bag9"

METADATA_CSV = PROJECT_ROOT / "isec_2f_5f_metadata.csv"

# Standard ImageNet transform for ResNet18
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ---------------------------------------------------------------------
# Metadata building / loading
# ---------------------------------------------------------------------
def _build_and_save_metadata() -> pd.DataFrame:
    """
    Build a single dataframe with both floors and save it to CSV.

    Columns:
      filename, floor, bag_id, frame_idx, timestamp_sec,
      floor_label (2 or 5), image_path
    """
    if not FLOOR2_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {FLOOR2_CSV}")
    if not FLOOR5_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {FLOOR5_CSV}")

    # Read CSVs with their own header row
    cols = ["filename", "floor", "bag_id", "frame_idx", "timestamp_sec"]

    df2 = pd.read_csv(FLOOR2_CSV)
    df5 = pd.read_csv(FLOOR5_CSV)

    # Make sure column names are consistent
    df2.columns = cols
    df5.columns = cols

    # Add floor labels
    df2["floor_label"] = 2
    df5["floor_label"] = 5

    # Build absolute image paths
    df2["image_path"] = df2["filename"].apply(
        lambda name: str(FLOOR2_IMG_DIR / name)
    )
    df5["image_path"] = df5["filename"].apply(
        lambda name: str(FLOOR5_IMG_DIR / name)
    )

    merged = pd.concat([df2, df5], ignore_index=True)

    # Drop any rows whose image_path does NOT actually exist on disk
    exists_mask = merged["image_path"].apply(os.path.exists)
    missing = merged[~exists_mask]

    if not missing.empty:
        print("WARNING: Some image paths do not exist. Dropping them.")
        print("First few missing rows:")
        print(missing.head())

    merged = merged[exists_mask].reset_index(drop=True)

    print("=== ISEC metadata built from CSVs ===")
    print(f"Total images: {len(merged)}")
    print(f"Floor 2 images: {(merged['floor_label'] == 2).sum()}")
    print(f"Floor 5 images: {(merged['floor_label'] == 5).sum()}")

    merged.to_csv(METADATA_CSV, index=False)
    print(f"Merged metadata saved to: {METADATA_CSV}")

    return merged


def load_isec_metadata(save_merged: bool = True) -> pd.DataFrame:
    """
    Public function used by the other scripts.

    `save_merged` is kept for backward compatibility with older code,
    but we always rebuild from the original CSVs and overwrite the
    merged metadata CSV.
    """
    return _build_and_save_metadata()


# ---------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------
class ISECImageDataset(Dataset):
    """
    Simple dataset: returns (image_tensor, floor_label)
    """

    def __init__(self, metadata_df: pd.DataFrame, transform=None):
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # Open image
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        floor_label = int(row["floor_label"])  # 2 or 5
        return img, floor_label


# ---------------------------------------------------------------------
# Backwards compatibility alias
# ---------------------------------------------------------------------
ISECFloorDataset = ISECImageDataset
