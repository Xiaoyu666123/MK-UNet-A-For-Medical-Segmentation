import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftHausdorffLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)

        # 检查边界是否为空
        if target.sum() < 10:
            return torch.tensor(0.0, device=pred.device)

        # 提取边界
        target_boundary = target - (-F.max_pool2d(-target, kernel_size=3, stride=1, padding=1))
        target_boundary = torch.clamp(target_boundary, 0, 1)

        pred_boundary = pred_prob - (-F.max_pool2d(-pred_prob, kernel_size=3, stride=1, padding=1))
        pred_boundary = torch.clamp(pred_boundary, 0, 1)

        # 计算边界损失
        boundary_loss = F.binary_cross_entropy(pred_boundary, target_boundary, reduction='mean')

        # 使用 L2 距离近似
        alignment_loss = F.mse_loss(pred_boundary, target_boundary)

        return boundary_loss + 0.5 * alignment_loss


class ImprovedBoundaryLoss(nn.Module):

    def __init__(self, alpha=0.5, focal_gamma=2.0, size_adaptive=True, boundary_width=3,
                 use_hausdorff=False, hd_weight=0.1, boundary_weight=0.3):
        super().__init__()
        self.alpha = alpha
        self.focal_gamma = focal_gamma
        self.size_adaptive = size_adaptive
        self.boundary_width = boundary_width
        self.use_hausdorff = use_hausdorff
        self.hd_weight = hd_weight
        self.boundary_weight = boundary_weight  # 边界预测损失权重

        # 注册膨胀和腐蚀核
        self.register_buffer('dilate_kernel', torch.ones(1, 1, 3, 3))
        self.register_buffer('erode_kernel', torch.ones(1, 1, 3, 3))

        # Soft Hausdorff Loss
        if self.use_hausdorff:
            self.hausdorff_loss = SoftHausdorffLoss()

    def dice_loss(self, pred, target, smooth=1.0):
        """Dice Loss"""
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    def focal_loss(self, pred, target):
        """Focal Loss - 关注难样本"""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** self.focal_gamma) * bce
        return focal_loss.mean()

    def _extract_boundary(self, mask):
        # 腐蚀操作 (最小池化)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)

        # 边界 = 原图 - 腐蚀
        boundary = mask - eroded

        # 膨胀边界区域，使边界更宽
        if self.boundary_width > 1:
            boundary = F.max_pool2d(boundary, kernel_size=self.boundary_width * 2 + 1,
                                    stride=1, padding=self.boundary_width)

        return boundary

    def boundary_loss(self, pred, target):
        # 使用拉普拉斯核提取边界
        kernel = torch.tensor([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=torch.float32, device=target.device)
        kernel = kernel.view(1, 1, 3, 3)

        # 计算边界
        boundary = F.conv2d(target.float(), kernel, padding=1)
        boundary = torch.clamp(boundary, 0, 1)

        # 边界区域权重
        boundary_weight = 1.0 + 5.0 * boundary

        # BCE with boundary weighting
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (bce * boundary_weight).mean()

        return weighted_bce

    def boundary_pred_loss(self, boundary_pred, target):
        kernel = torch.tensor([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=torch.float32, device=target.device)
        kernel = kernel.view(1, 1, 3, 3)

        boundary_gt = F.conv2d(target.float(), kernel, padding=1)
        boundary_gt = torch.clamp(boundary_gt, 0, 1)

        boundary_gt = (boundary_gt > 0.1).float()

        loss = F.binary_cross_entropy_with_logits(boundary_pred, boundary_gt)

        return loss

    def size_adaptive_weight(self, target):
        """根据肿瘤大小自适应调整权重"""
        if not self.size_adaptive:
            return 1.0

        # 计算前景比例
        fg_ratio = target.sum() / target.numel()

        # 小肿瘤 (< 5%) 增加权重
        if fg_ratio < 0.05:
            return 2.0
        # 中等肿瘤 (5-20%)
        elif fg_ratio < 0.2:
            return 1.5
        # 大肿瘤 (> 20%)
        else:
            return 1.0

    def forward(self, pred, target, boundary_pred=None):

        # 1. Dice Loss
        dice_l = self.dice_loss(pred, target)

        # 2. Focal Loss
        focal_l = self.focal_loss(pred, target)

        # 3. Boundary Loss
        boundary_l = self.boundary_loss(pred, target)

        # 4. Soft Hausdorff Loss
        hd_l = 0.0
        if self.use_hausdorff:
            hd_l = self.hausdorff_loss(pred, target)

        # 5. 边界预测损失
        boundary_pred_l = 0.0
        if boundary_pred is not None:
            boundary_pred_l = self.boundary_pred_loss(boundary_pred, target)

        # 6. 自适应权重
        size_weight = self.size_adaptive_weight(target)

        # 7. 组合损失
        base_loss = (
            (1 - self.alpha) * (dice_l + 0.5 * focal_l) +  # 区域损失
            self.alpha * boundary_l                         # 边界损失
        )

        # 加入 Hausdorff Loss 和边界预测损失
        total_loss = (base_loss + self.hd_weight * hd_l + self.boundary_weight * boundary_pred_l) * size_weight

        return total_loss, dice_l.item(), boundary_l.item()