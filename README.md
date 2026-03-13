# Visual Place Recognition for Multi-Floor SLAM

A metric learning approach to solving perceptual aliasing in multi-floor indoor environments, enabling robust loop closure detection for visual SLAM systems.

## Problem

Multi-floor buildings present a critical challenge for visual SLAM: architecturally similar floors produce high visual similarity scores (82–98% false positive rate with baseline CNN descriptors), causing catastrophic loop closures where the system incorrectly matches locations across different floors.

## Approach

This project addresses cross-floor perceptual aliasing through three techniques:

1. **Triplet-Loss Metric Learning** — Remaps the embedding space so same-floor locations cluster together while cross-floor pairs are pushed apart
2. **NetVLAD Aggregation** — Uses a K-means visual vocabulary to generate compact, discriminative place descriptors
3. **Cross-Floor Similarity Reduction** — Achieves measurable improvement in floor discrimination for SLAM loop closure

## Results

| Metric | Baseline | After Metric Learning |
|--------|----------|-----------------------|
| Mean cross-floor similarity | 0.854 | 0.796 |

The approach successfully reduces false cross-floor matches, preventing catastrophic loop closures in multi-floor SLAM pipelines.

## Dataset

- **1,331 images** captured across multiple floors of Northeastern University's ISEC building
- Images capture the architectural repetition (similar corridors, lighting, materials) that causes perceptual aliasing
- Dataset organized by floor for training triplet-loss networks

## Repository Structure

```
vpr-multi-floor-slam/
├── README.md
├── requirements.txt
├── data/
│   └── README.md              # Dataset description and download instructions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_similarity.ipynb
│   ├── 03_triplet_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── dataset.py             # Data loading and preprocessing
│   ├── model.py               # Network architecture and triplet loss
│   ├── netvlad.py             # NetVLAD aggregation layer
│   ├── evaluate.py            # Similarity computation and metrics
│   └── utils.py               # Visualization and helper functions
├── results/
│   ├── similarity_matrices/   # Before/after similarity heatmaps
│   ├── embeddings/            # t-SNE or PCA visualizations
│   └── figures/               # Publication-ready figures
└── docs/
    └── report.pdf             # Full project report
```

## Getting Started

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.12
torchvision
numpy
matplotlib
scikit-learn
opencv-python
```

### Installation

```bash
git clone https://github.com/kumarjaw/vpr-multi-floor-slam.git
cd vpr-multi-floor-slam
pip install -r requirements.txt
```

### Running

```bash
# Explore the dataset and baseline similarity
jupyter notebook notebooks/01_data_exploration.ipynb

# Train the triplet-loss metric learning model
jupyter notebook notebooks/03_triplet_training.ipynb

# Evaluate results
jupyter notebook notebooks/04_evaluation.ipynb
```

## Technical Details

- **Feature Extraction**: CNN backbone for generating image descriptors
- **Metric Learning**: Triplet loss with online hard negative mining to learn floor-discriminative embeddings
- **Aggregation**: NetVLAD-style pooling with K-means visual vocabulary for compact place representations
- **Evaluation**: Cross-floor similarity matrices, precision-recall analysis for loop closure detection

## Built With

- Python, PyTorch
- OpenCV for image processing
- scikit-learn for K-means clustering and evaluation metrics
- MATLAB (supplementary analysis)

## Author

**Jawahar Nishit Kumar**
MS Robotics, Mechatronics and Automation — Northeastern University
[LinkedIn](https://linkedin.com/in/jawahar-nishit)

## Acknowledgments

Developed as part of the Mobile Robotics course at Northeastern University, Fall 2025.

## License

This project is for academic and portfolio purposes. Please contact the author for any usage inquiries.
