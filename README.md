# 🧠 Multiclass Hemorrhage Segmentation

> A deep learning pipeline for multiclass hemorrhage segmentation in CT scans, using multiple UNet-based architectures. 🏥

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Project Overview

Our goal is to segment 5 types of hemorrhages (plus background) from CT images using state-of-the-art deep learning models:

- **Goal:** Segment 5 types of hemorrhages (plus background) from CT images.

- **Data:** BHSD - 3D CT scans, processed into 2D slices, with strong augmentation support. [Link](https://huggingface.co/datasets/mbhseg/mbhseg24) to dataset.

- **UNetLite** 🚀 - A lightweight UNet architecture
- **ResNetUNet** 🧩 - UNet with a ResNet-101 backbone
- **ResNetUNet-Att** 🔍 - UNet with a ResNet-101 backbone and attention gates

## 📁 Project Structure

### 🧮 Core Models
```
models/
├── lite_unet_model.py      # Lightweight UNet implementation
├── resunet_model.py        # ResNet-UNet implementation
└── resunet_att_model.py    # ResNet-UNet with attention gates
```

### 📊 Dataset & Processing
```
dataset/
├── dataset.py              # Base dataset class
├── dataset_augmented.py    # Dataset with augmentation
├── augmentation.py         # Augmentation techniques
├── filter_with_masks.py    # Mask filtering utilities
├── save_augmented_data.py  # Run augmented data pipeline and save to directory
└── slice_images.py         # Convert 3D to 2D slices
```

### 📓 Kaggle Notebooks
```
kaggle_notebooks/
├── Lite U-Net train.ipynb           # UNetLite training
├── ResNet U-Net train.ipynb         # ResNetUNet training
└── ResNet Attention U-Net train.ipynb # ResNetUNet-Att training
```

### 📂 Data Organization
```
├── data/                  # Processed 2D data
│   ├── images/           # CT scan slices
│   └── masks/            # Segmentation masks
├── MBH_train_label/      # Original 3D data
│   ├── imagesTr/         # Training volumes
│   └── labelsTr/         # Training labels
├── augmented_data/       # Pre-generated augmented data
└── checkpoints/          # Model weights
    ├── lite_unet.pth
    ├── resnet_unet.pth
    └── resnet_att_unet.pth
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hemorrhage-segmentation.git
cd hemorrhage-segmentation

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1️⃣ Data Preparation
```bash
# Convert 3D scans to 2D slices and filter empty masks
python dataset/filter_with_masks.py

# Generate and save augmented data
python dataset/save_augmented_data.py
```

### 2️⃣ Training
which model? each has a training script in `/train`. Script includes training and testing. While training logs and a csv file for tracking epoch statistics are generated for the process. When testing a csv file is generated with Dice and IoU scores of each class and graphs of predictions vs ground truth is generated and saved.
```bash
# Train with augmented data
python train/train_<model-type>.py
```

### 3️⃣ Prediction with a sampel image
Use the `predict.ipynb` notebook. Make sure to have the models in the `checkpoints/` folder with the appropriate names. 

## 🧠 Model Details

### UNetLite 🚀
- ⚡ Lightweight architecture
- 🔢 Reduced filters (32-256)
- 🏃‍♂️ Fast inference

### ResNetUNet 🧩
- 🎯 ResNet-101 backbone
- 🔄 Standard UNet decoder
- ⚖️ Balanced performance

### ResNetUNet-Att 🔍
- 🎯 ResNet-101 backbone
- 🔍 Attention gates
- 🏆 Best performance

## 📊 Data Pipeline

1. **3D to 2D Conversion** 🔄
   - Convert volumes to slices
   - Filter empty masks

2. **Preprocessing** 🛠️
   - HU value normalization
   - Image resizing
   - Data augmentation

3. **Training/Inference** 🧠
   - Multi-class segmentation
   - 6 output channels
   - Probability maps

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.