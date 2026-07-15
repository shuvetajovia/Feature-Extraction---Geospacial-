# Geospatial Semantic Segmentation

This repository contains the training pipeline and experimental setup for semantic segmentation of satellite imagery. The goal is to classify each pixel into one of 6 distinct land-cover classes using state-of-the-art Deep Learning models.

## 🚀 Overview

The project uses the `segmentation-models-pytorch` library to build and train a **U-Net** architecture with a robust **ResNet-50** backbone. 
The pipeline includes data loading, augmentation (using `albumentations`), and reading satellite imagery via `rasterio` and `OpenCV`.

## 🧠 Model Architecture

The segmentation network employs a **U-Net** structure, heavily relying on a pre-trained **ResNet-50** encoder to extract dense spatial features from satellite images.

<img width="1024" height="1024" alt="WhatsApp Image 2026-07-15 at 11 17 28 AM" src="https://github.com/user-attachments/assets/06c0d107-3d98-4fa9-aa21-a64973de2dbd" />


```mermaid
graph TD
    subgraph Encoder: ResNet-50
        A[Input Image: 3 Channels] --> B(Conv1: 7x7, 64)
        B --> C(MaxPool)
        C --> D(Layer 1: Bottleneck Blocks)
        D --> E(Layer 2: Bottleneck Blocks)
        E --> F(Layer 3: Bottleneck Blocks)
        F --> G(Layer 4: Bottleneck Blocks)
    end
    
    subgraph Decoder: U-Net
        G --> H(Decoder Block 1)
        F -. Skip Connection .-> H
        H --> I(Decoder Block 2)
        E -. Skip Connection .-> I
        I --> J(Decoder Block 3)
        D -. Skip Connection .-> J
        J --> K(Decoder Block 4)
        B -. Skip Connection .-> K
        K --> L(Decoder Block 5)
    end
    
    L --> M[Segmentation Head: Conv2D]
    M --> N[Output Mask: 6 Classes]
```

### Key Components:
- **Encoder (ResNet-50)**: Uses weights pre-trained on ImageNet to extract robust hierarchical features (edges, textures, object parts).
- **Decoder**: A standard U-Net decoder that progressively upsamples the feature maps.
- **Skip Connections**: Connect the encoder stages directly to their corresponding decoder stages to recover fine-grained spatial details lost during max-pooling.
- **Output Head**: A 1x1 Convolution layer mapping the final decoder features to the 6 target classes.

## ⚙️ Training Details

The notebook `train_model.ipynb` contains the full training pipeline with the following hyperparameters:
- **Loss Function**: Combined `DiceLoss` (multiclass) + `CrossEntropyLoss` to handle class imbalances and pixel-wise accuracy.
- **Optimizer**: `AdamW` (Adam with Weight Decay)
- **Batch Size**: 2
- **Epochs**: 80
- **Device**: PyTorch CUDA/CPU auto-detection

## 📂 Repository Contents
- `train_model.ipynb`: Jupyter notebook containing the full data loading, model definition, training loop, and evaluation logic.
- `feature_extraction_unet_resnet50.pth.zip`: The serialized weights of the fully trained model.

## 🛠️ Setup & Installation

To run the notebook, install the necessary dependencies:

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install rasterio
```
