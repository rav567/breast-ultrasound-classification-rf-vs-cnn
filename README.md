# Breast Cancer Ultrasound Classification: Random Forest vs CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-BreastMNIST-blueviolet)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

> **Does deep learning actually beat classical machine learning on small medical imaging datasets, and is it worth the extra compute?**
>
> This project benchmarks a **Random Forest** classifier against a **Convolutional Neural Network** for benign vs malignant breast ultrasound classification on **BreastMNIST**, under identical data splits, preprocessing and augmentation, so the comparison is fair.

**TL;DR:** the CNN wins on every metric (test **F1 0.908** vs **0.873**), but the Random Forest reaches **97% of the CNN's performance** while training in seconds on a CPU.

Built for the Applied Machine Learning Systems module (ELEC0134) at UCL, 2025/26.

---

## Results

### Test-set performance

Best configuration of each model, retrained on train + validation and evaluated once on the untouched test set (156 images):

| Metric | Model A · Random Forest | Model B · CNN | Difference |
|---|:---:|:---:|:---:|
| Accuracy | 0.801 | **0.859** | +5.8% |
| Precision | 0.817 | **0.865** | +4.8% |
| Recall | 0.939 | **0.956** | +1.7% |
| **F1 score** | 0.873 | **0.908** | **+3.5%** |

### Selected configurations

| Model | Best configuration | Validation F1 | Test F1 | Val → test drop |
|---|---|:---:|:---:|:---:|
| A · Random Forest | Raw pixels + high capacity + augmentation | 0.923 | 0.873 | 0.050 |
| B · CNN | Dropout + augmentation | 0.941 | 0.908 | **0.033** |

---

## Key findings

1. **The CNN generalised better, despite having far more parameters.** It lost only 0.033 F1 between validation and test, compared with 0.050 for the Random Forest. Convolutions keep the spatial structure that flattening throws away.
2. **The Random Forest is the efficient option.** It got 97% of the CNN's F1 while training in seconds with no GPU, no epochs and no learning-rate schedule.
3. **HOG features lost to raw pixels.** At 28 × 28 resolution, the hand-crafted HOG descriptor consistently underperformed raw pixels at both model capacities. The fine texture cues it relies on don't survive the downscaling.
4. **Capacity alone leads to memorisation.** The high-capacity Random Forest hit a perfect training F1 of 1.00 with a 0.092 train–validation gap. The unregularised CNN drove training loss towards zero while validation loss plateaued around 0.3.
5. **Regularisation works best in combination.** For the CNN, dropout + augmentation gave the most stable convergence and the best validation F1 (0.941). On their own, dropout (0.940) beat augmentation (0.930). For the Random Forest, augmentation added a 1.7% validation gain.
6. **Class imbalance shapes the metrics.** The classes are imbalanced at roughly 2.7:1 in every split, which pushed both models towards high recall (above 0.93) with lower precision.

---

## Approach

### Model A · Random Forest (classical ML)
- **Features:** raw flattened pixels (784-d) vs **Histogram of Oriented Gradients** (9 orientations, 4 × 4 pixels per cell, 2 × 2 cells per block)
- **Classifier:** scikit-learn `RandomForestClassifier` with Gini impurity and a fixed seed
- **Capacity study:** low (50 trees, max depth 5) vs high (200 trees, unlimited depth)
- **Augmentation study:** training set tripled with augmented copies before feature extraction
- **Data-efficiency study:** trained on 10%, 30%, 60% and 100% of the training data

### Model B · Convolutional Neural Network (deep learning)
- **Architecture:** 3 conv blocks (32 → 64 → 128 filters, each Conv2d + BatchNorm + ReLU + MaxPool), then fully connected layers 1152 → 64 → 2, with optional dropout (0.5)
- **Training:** Adam (lr 0.001, weight decay 1e-4), learning rate halved every 15 epochs, 50 epochs, batch size 32, cross-entropy loss
- **Ablation:** 2 × 2 grid of dropout (on/off) × data augmentation (on/off)

### Shared augmentation (training data only)
| Transform | Setting | Probability |
|---|---|:---:|
| Horizontal flip | – | 0.5 |
| Rotation | ±15° | 0.5 |
| Brightness | ±0.1 | 0.5 |
| Contrast | ±0.1 | 0.5 |

### Evaluation protocol
1. Every configuration is trained on the training split and compared on **validation F1**, chosen over accuracy because the classes are imbalanced.
2. The winning configuration is retrained on **train + validation** to make the most of the small dataset.
3. It is evaluated **once** on the held-out test split, reporting accuracy, precision, recall and F1.

---

## Dataset

[**BreastMNIST**](https://medmnist.com/) is part of MedMNIST v2. It contains **780 breast ultrasound images**, resized to 28 × 28 grayscale, with the official splits:

| Split | Images |
|---|:---:|
| Train | 546 |
| Validation | 78 |
| Test | 156 |

Pixel values are normalised from [0, 255] to [0, 1]. In BreastMNIST, label 0 is malignant and label 1 is normal or benign, so the precision and recall above are reported for label 1.

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

---

## Quick start

```bash
git clone https://github.com/rav567/AMLS_assignment25_26.git
cd AMLS_assignment25_26
pip install -r requirements.txt
python main.py
```

`main.py` loads the data, runs every Model A and Model B experiment, and prints the final test metrics for both models. The plotting utilities in `Code/A/plots_a.py` and `Code/B/plots_b.py` regenerate the report figures; uncomment the plotting lines in `main.py` to enable them.

## Project structure

```
├── Code/
│   ├── A/
│   │   ├── model_a.py         # Random Forest pipeline and experiments
│   │   └── plots_a.py         # Capacity, feature and overfitting plots
│   ├── B/
│   │   ├── model_b.py         # CNN architecture, training loop and ablation
│   │   └── plots_b.py         # Learning curves and comparison plots
│   └── utils/
│       ├── data_loader.py     # Loads and normalises the BreastMNIST splits
│       ├── preprocessing.py   # Pixel flattening and HOG feature extraction
│       └── augmentation.py    # Flip, rotation, brightness and contrast jitter
├── Datasets/                  # Place breastmnist.npz here
├── main.py                    # Runs all experiments end to end
└── requirements.txt
```

## Reproducibility

Random seeds are fixed for NumPy and PyTorch (`seed = 42`), and cuDNN runs in deterministic mode, so repeated runs give consistent results.

## Limitations and future work

- **Small, low-resolution data:** 780 images at 28 × 28 pixels lose much of the fine texture in clinical ultrasound, which likely understates what both models could achieve.
- **Class imbalance:** the 2.7:1 ratio skews both models towards recall over precision.
- **Next steps:** evaluate both approaches on higher-resolution, larger and more balanced datasets, and test imbalance-aware techniques such as weighted loss functions.

## Tech stack

Python · PyTorch · scikit-learn · scikit-image · NumPy · SciPy · Matplotlib · Seaborn

## Author

**Ravpreet Gill** · [@rav567](https://github.com/rav567)

## References

- Yang, J. et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 10, 2023.
- Al-Dhabyani, W. et al. *Dataset of breast ultrasound images.* Data in Brief, 2020.
