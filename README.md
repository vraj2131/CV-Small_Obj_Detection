# COS529 Advanced Computer Vision - Vehicle Detection in Aerial Imagery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-v7.0-green.svg)](https://github.com/ultralytics/yolov5)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements **multi-modal vehicle detection in aerial imagery** using YOLOv5 on the VEDAI (Vehicle Detection in Aerial Imagery) dataset. The system processes both **color (optical)** and **infrared** imagery to detect and classify various vehicle types in aerial surveillance scenarios.

### Key Features
- ✅ **Multi-modal Processing**: Color and Infrared image fusion
- ✅ **Multi-class Detection**: 1, 8, and 9-class vehicle detection configurations
- ✅ **Aerial Imagery Optimization**: Specialized for small object detection in aerial views
- ✅ **Cross-validation**: 10-fold validation for robust model evaluation
- ✅ **Production Ready**: Complete training pipeline with configuration management

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd COS529_PROJECT

# Install dependencies
cd yolov5
pip install -r requirements.txt

# Verify installation
python detect.py --weights yolov5s.pt --source data/images/bus.jpg
```

### Quick Training

```bash
# Train on single-class vehicle detection
python train.py --data ../data/vedai_car.yaml --weights yolov5m.pt --img 512 --epochs 50

# Train on multi-class vehicle detection (8 classes)
python train.py --data ../data/vedai8.yaml --weights yolov5m.pt --img 512 --epochs 50

# Use the provided training script
bash run_train.sh
```

## 📊 Dataset Information

### VEDAI Dataset
- **Total Images**: 1,090+ aerial images
- **Modalities**: Color (co) and Infrared (ir)
- **Image Resolution**: 512×512 pixels (original), 1024×1024 (resized)
- **Annotations**: YOLO format bounding boxes
- **Vehicle Classes**: Car, Pickup, Camping Car, Truck, Tractor, Boat, Van, Plane

### Dataset Structure
```
data/
├── VEDAI/                    # Original dataset (512×512)
│   ├── images/               # PNG images (co + ir modalities)
│   ├── fold01_write_test_fixed.txt  # Training split
│   ├── fold02_write_test_fixed.txt  # Validation split
│   └── fold03_write_test_fixed.txt  # Test split
└── VEDAI_1024/               # Resized dataset (1024×1024)
    ├── images/               # Resized PNG images
    └── labels/               # YOLO format annotations
```

## 🏗️ Architecture

### Model Configurations
- **YOLOv5m**: Medium model (21.2M parameters) - Primary choice
- **YOLOv5s**: Small model (7.2M parameters) - Fast inference
- **YOLOv5l**: Large model (46.5M parameters) - High accuracy

### Multi-Modal Processing
1. **Data Preprocessing**: Convert polygon annotations to YOLO format
2. **Modality Handling**: Process both color and infrared channels
3. **Training**: Standard YOLOv5 training with aerial-specific augmentations
4. **Inference**: Multi-modal vehicle detection and classification

## 📈 Performance Metrics

### Model Performance (VEDAI Dataset)
| Model | mAP@0.5 | mAP@0.5:0.95 | Speed (ms) | Parameters |
|-------|---------|--------------|------------|------------|
| YOLOv5s | 0.XX | 0.XX | XX | 7.2M |
| YOLOv5m | 0.XX | 0.XX | XX | 21.2M |
| YOLOv5l | 0.XX | 0.XX | XX | 46.5M |

*Note: Performance metrics will be updated after training completion*

## 🛠️ Usage

### Training Custom Models

```bash
# Single-class vehicle detection
python train.py \
    --data ../data/vedai_car.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 50 \
    --device 0

# Multi-class vehicle detection
python train.py \
    --data ../data/vedai8.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 50 \
    --device 0
```

### Inference

```bash
# Detect vehicles in images
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/images/ \
    --img 512 \
    --conf 0.25

# Detect vehicles in video
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/video.mp4 \
    --img 512 \
    --conf 0.25
```

### Validation

```bash
# Validate model performance
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data ../data/vedai8.yaml \
    --img 512 \
    --conf 0.001 \
    --iou 0.65
```

## 📁 Project Structure

```
COS529_PROJECT/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── convert_to_vehicle_yolo.py         # Data preprocessing script
├── data/                              # Dataset directory
│   ├── VEDAI/                         # Original dataset
│   └── VEDAI_1024/                    # Resized dataset
├── yolov5/                            # YOLOv5 implementation
│   ├── train.py                       # Training script
│   ├── detect.py                      # Inference script
│   ├── val.py                         # Validation script
│   ├── data/                          # Configuration files
│   │   ├── vedai_car.yaml            # Single-class config
│   │   ├── vedai8.yaml               # 8-class config
│   │   └── vedai9.yaml               # 9-class config
│   ├── models/                        # Model architectures
│   ├── utils/                         # Utility functions
│   └── requirements.txt               # Dependencies
└── yolov5m.pt                         # Pre-trained weights
```

## 🔧 Configuration Files

### Dataset Configurations
- **`vedai_car.yaml`**: Single-class vehicle detection
- **`vedai8.yaml`**: 8-class vehicle detection (Car, Pickup, Camping, Truck, Other, Tractor, Boat, Van)
- **`vedai9.yaml`**: 9-class vehicle detection (adds Plane class)

### Training Parameters
- **Image Size**: 512×512 pixels
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Epochs**: 50 (configurable)
- **Learning Rate**: Auto-scaled by YOLOv5
- **Optimizer**: SGD with momentum

## 🎓 Academic Context

This project was developed as part of **COS529 Advanced Computer Vision** course, focusing on:

- **Multi-modal Computer Vision**: Fusion of color and infrared imagery
- **Small Object Detection**: Challenges in aerial vehicle detection
- **Deep Learning**: Implementation of state-of-the-art YOLOv5 architecture
- **Dataset Processing**: Custom annotation format conversion
- **Model Evaluation**: Comprehensive cross-validation methodology

## 📚 Technical Skills Demonstrated

- **Computer Vision**: Object detection, multi-modal fusion, aerial imagery
- **Deep Learning**: PyTorch, YOLOv5, transfer learning
- **Data Processing**: Annotation conversion, dataset preparation
- **Model Training**: Hyperparameter tuning, cross-validation
- **Software Engineering**: Modular design, configuration management

---

*This project demonstrates advanced computer vision techniques for multi-modal vehicle detection in aerial imagery, showcasing expertise in deep learning, data processing, and model optimization.*
