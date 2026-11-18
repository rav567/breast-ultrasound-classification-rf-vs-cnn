# Breast Cancer Ultrasound Classification: Random Forest vs CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

> Can a classical, handcrafted-feature model hold its own against a deep CNN on small medical imaging data?
> This project benchmarks a **HOG + Random Forest** pipeline against a **PyTorch CNN** for benign vs malignant breast ultrasound classification on **BreastMNIST**, with controlled experiments on model capacity, data augmentation, regularisation and reproducibility.

📄 **Report:** email: ravpreet.s.gill@gmail.com

---

## Results

Final test-set performance of the best configuration of each model (selected on validation F1, then retrained on train + validation):

| Model | Best configuration | Accuracy | Precision | Recall | F1 |
|---|---|:---:|:---:|:---:|:---:|
| **A** · Random Forest | `FEATURES + CAPACITY` | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| **B** · CNN (PyTorch) | `DROPOUT / AUG SETTING` | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

<p align="center">
  <img src="assets/confusion_matrices.png" width="45%" alt="Confusion matrices for both models">
  &nbsp;
  <img src="assets/cnn_training_curves.png" width="45%" alt="CNN training and validation curves">
</p>

## Key findings

- **FINDING 1** — e.g. which model won and by how much on F1.
- **FINDING 2** — e.g. whether HOG features beat raw pixels for the Random Forest.
- **FINDING 3** — e.g. how augmentation and dropout changed CNN overfitting.

---

## Approach

### Model A · Classical pipeline
- **Features:** raw flattened pixels (784-d) vs **Histogram of Oriented Gradients (HOG)** edge/texture descriptors
- **Classifier:** scikit-learn `RandomForestClassifier`
- **Capacity study:** low (50 trees, depth 5) vs high (200 trees, unlimited depth)
- **Augmentation study:** training set tripled with random flips, rotations, brightness and contrast jitter
- **Data-efficiency study:** trained on 10%, 30%, 60% and 100% of the training data

### Model B · Convolutional neural network
- **Architecture:** 3 conv blocks (32 → 64 → 128 filters, BatchNorm, ReLU, MaxPool) + 2-layer classifier head
- **Training:** Adam (lr 1e-3, weight decay 1e-4), StepLR schedule, 50 epochs, cross-entropy loss
- **Ablation:** 2 × 2 grid of dropout (on/off) × data augmentation (on/off)

### Evaluation protocol
1. Every configuration is trained on the training split and compared on **validation F1**.
2. The winning configuration is retrained on **train + validation**.
3. It is evaluated **once** on the held-out test split, reporting accuracy, precision, recall and F1.

F1 is used for model selection rather than accuracy because the classes in BreastMNIST are imbalanced.

---

## Dataset

[**BreastMNIST**](https://medmnist.com/) (part of MedMNIST v2) contains 780 breast ultrasound images, resized to 28 × 28 grayscale, with official train / validation / test splits (546 / 78 / 156).

Download `breastmnist.npz` from the [MedMNIST Zenodo record](https://zenodo.org/records/10519652) and place it here:

```
Datasets/
└── breastmnist.npz
```

Or fetch it with the `medmnist` package:

```bash
pip install medmnist
python -c "from medmnist import BreastMNIST; BreastMNIST(split='train', download=True, root='Datasets')"
```

## Quick start

```bash
git clone https://github.com/rav567/breast-ultrasound-classification-rf-vs-cnn.git
cd breast-ultrasound-classification-rf-vs-cnn
pip install -r requirements.txt
python main.py
```

`main.py` loads the data, runs every Model A and Model B experiment, and prints the final test metrics for both models. Plotting utilities in `Code/A/plots_a.py` and `Code/B/plots_b.py` regenerate the report figures; uncomment the plotting lines in `main.py` to enable them.

## Project structure

```
├── Code/
│   ├── A/
│   │   ├── model_a.py         # Random Forest pipeline and experiments
│   │   └── plots_a.py         # Capacity, feature and overfitting plots
│   ├── B/
│   │   ├── model_b.py         # CNN architecture, training loop and ablation
│   │   └── plots_b.py         # Training curves and comparison plots
│   └── utils/
│       ├── data_loader.py     # Loads and normalises BreastMNIST splits
│       ├── preprocessing.py   # Flattening and HOG feature extraction
│       └── augmentation.py    # Flip, rotation, brightness and contrast jitter
├── Datasets/                  # Place breastmnist.npz here
├── main.py                    # Runs all experiments end to end
└── requirements.txt
```

## Reproducibility

Random seeds are fixed for NumPy and PyTorch (`seed = 42`), and cuDNN is set to deterministic mode, so repeated runs give consistent results.

## Tech stack

Python · PyTorch · scikit-learn · scikit-image · NumPy · SciPy · Matplotlib · Seaborn

## Author

**Ravpreet Gill** · [LinkedIn](YOUR_LINKEDIN_URL) · [Website](YOUR_WEBSITE_URL)

## Acknowledgements

Dataset: Yang, J. et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 2023. Original images from Al-Dhabyani, W. et al. *Dataset of breast ultrasound images.* Data in Brief, 2020.
