import torch
import torch.nn as nn
import torch.optim as optim
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import swanlab
import torch.nn.functional as F

# 数据路径设置
train_dir = 'dataset/train'
val_dir = 'dataset/valid'
test_dir = 'dataset/test'

# COCO的JSON文件路径
train_annoation_file = 'dataset/train/_annotations.coco.json'
test_annoation_file = 'dataset/test/_annotations.coco.json'
val_annoation_file = 'dataset/valid/_annotations.coco.json'

# 加载COCO数据集
train_coco = COCO(train_annoation_file)
val_coco = COCO(val_annoation_file)
test_coco = COCO(test_annoation_file)

def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    """焦点损失"""
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def dice_loss_with_logits(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=[1, 2, 3])
    union = probs.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def combined_loss(logits_or_list, target, weight_dice=0.5, weight_bce=0.3, weight_focal=0.2):
    if isinstance(logits_or_list, (list, tuple)):
        logits = logits_or_list[0]
    else:
        logits = logits_or_list

    target = target.float()

    dice_loss_val = dice_loss_with_logits(logits, target)
    bce_loss_val = F.binary_cross_entropy_with_logits(logits, target)
    focal_loss_val = focal_loss_with_logits(logits, target)

    return weight_dice * dice_loss_val + weight_bce * bce_loss_val + weight_focal * focal_loss_val