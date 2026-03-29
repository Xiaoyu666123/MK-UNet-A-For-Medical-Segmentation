# MK-UNet-A: Multi-Kernel UNet for Brain Tumor Segmentation

A deep learning project for brain tumor segmentation using an enhanced UNet architecture with multi-kernel inverted residual blocks and boundary-aware attention mechanisms.

## Features

- **Multi-Kernel Inverted Residual Blocks**: Captures multi-scale features through parallel depthwise convolutions with different kernel sizes (1x1, 3x3, 5x5)
- **CBAM Attention**: Convolutional Block Attention Module for channel and spatial attention
- **Grouped Attention Gates**: Efficient attention mechanism for encoder-decoder skip connections
- **Boundary Detection Head**: Dedicated head for boundary-aware segmentation
- **Improved Loss Function**: Combined Dice Loss, Focal Loss, and Boundary Loss with size-adaptive weighting

## Model Architecture
<img width="2816" height="1536" alt="MK-UNet-A" src="https://github.com/user-attachments/assets/2cd72af9-ba4f-421e-bcfc-210f43704e9d" />


### MK-Unet-A (Main Model)
The enhanced model includes:
- Multi-kernel depthwise convolutions for multi-scale feature extraction
- CBAM attention with configurable strategies (`deep`, `shallow`, `all`, `none`)
- Boundary prediction head for improved edge detection
- Bilinear upsampling with learnable convolutions


## Project Structure

```
paper/
├── MK_UNet_A.py              # Enhanced MKUnet model definition
├── MK_UNet_Baseline.py       # Baseline MK-UNet model
├── MK_UNet_A_train.py        # Training script for MKUNet-A
├── MK_UNet_Base_train.py     # Training script for baseline
├── Improved_loss.py          # Advanced loss functions
├── Loss.py                   # Basic loss functions and metrics
├── Data_Process.py           # Dataset and data augmentation
├── LoadData.py               # Data loading utilities
├── refined_prediction.py     # prediction refinement
├── requirements.txt          # Dependencies
├── test_model.py          # Test model
└── dataset/
    ├── train/
    ├── valid/
    └── test/
```

## Installation

```bash
conda create -n mkuneta python=3.9
conda activate mkuneta
pip install -r requirements.txt
```

## Dataset preparation

The project expects COCO-format annotations. Download the dataset from [Kaggle: Brain Tumor Image Dataset (Semantic Segmentation)](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation).

## Training

```bash
python MK_UNet_A_train.py
```

### Training Configuration
- Batch size: 16
- Learning rate: 1e-4
- Epochs: 100
- Optimizer: Adam
- Image size: 256x256



###  Visualize the training results

If you want to visualize the segmentation results of the training process, you can disable the code annotations related to SwanLab in the experiment.

~~~bash
pip install swanlab

swanlab login #sign up you person account
~~~

The steps are as follows:

```python
swanlab.init(
    project="your project name",
    experiment_name="your experiment_name",
    config={
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    },
)


```

Visualize your experimental data:

~~~python
swanlab.log({
            'train/loss': train_loss,
            'train/dice': train_dice,
            'train/epoch': train_epoch,
            'train/lr': lr,
            'val/loss': val_loss,
            'val/dice': val_dice,
        }, step=epoch + 1)
~~~

##  TestModel

~~~bash
python test_model.py
~~~



## Citation

If you use this code, please cite:

```bibtex
@misc{mk-unet,
  author = {Your Name},
  title = {MK-UNet: Multi-Kernel UNet for Brain Tumor Segmentation},
  year = {2026}
}
```
