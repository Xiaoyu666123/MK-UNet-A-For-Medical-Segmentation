import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import morphology
import numpy as np
from scipy.spatial.distance import cdist

def compute_hd95(pred, target, spacing=1.0):
    pred = pred.astype(bool)
    target = target.astype(bool)

    # 如果某张图完全没有预测出肿瘤（空的），或者全是肿瘤，HD95无定义
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0  # 或者返回 np.nan，视情况而定

    # 提取边缘
    pred_border = pred ^ morphology.binary_erosion(pred, structure=np.ones((3, 3)))
    target_border = target ^ morphology.binary_erosion(target, structure=np.ones((3, 3)))

    # 获取边缘点的坐标
    pred_coords = np.argwhere(pred_border)
    target_coords = np.argwhere(target_border)

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0

    # 计算点对点距离矩阵 (注意：如果图很大，这一步可能会慢，256x256通常没问题)
    dists = cdist(pred_coords, target_coords)

    # Pred 到 Target 的最近距离
    min_dists_p2t = dists.min(axis=1)
    # Target 到 Pred 的最近距离
    min_dists_t2p = dists.min(axis=0)

    # 合并两边的距离
    all_dists = np.concatenate([min_dists_p2t, min_dists_t2p])

    # 取第 95% 分位数的距离
    hd95 = np.percentile(all_dists, 95)

    return hd95 * spacing