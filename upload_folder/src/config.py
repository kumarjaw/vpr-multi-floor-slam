from pathlib import Path

# Root of the project:  .../Mobile robotics/Project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------- Baseline descriptors (ResNet18) ----------
# This is what extract_features_resnet.py already created
DESCRIPTORS_NPZ = PROJECT_ROOT / "isec_resnet18_descriptors.npz"

# ---------- Metadata CSV ----------
# This is what load_isec_metadata() / dataset_isec.py saved
METADATA_CSV = PROJECT_ROOT / "isec_2f_5f_metadata.csv"

# ---------- Metric-learning (fine-tuned) model paths ----------
# Checkpoint of the fine-tuned ResNet18 metric model
METRIC_CHECKPOINT = PROJECT_ROOT / "isec_floor_metric_resnet18.pth"

# Descriptors extracted from the fine-tuned model
METRIC_DESCRIPTORS_NPZ = PROJECT_ROOT / "isec_resnet18_metric_descriptors.npz"
