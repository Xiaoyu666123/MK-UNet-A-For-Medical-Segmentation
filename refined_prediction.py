import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import morphology, measure

class RefinedPredictor:
    def __init__(self,
                 threshold=0.5,
                 use_morphology=False,
                 use_crf=False,
                 min_size=100,
                 smooth_kernel=0):
        self.threshold = threshold
        self.use_morphology = use_morphology
        self.use_crf = use_crf
        self.min_size = min_size
        self.smooth_kernel = smooth_kernel

    def __call__(self, pred_prob, image=None):
        # 转为 numpy
        if isinstance(pred_prob, torch.Tensor):
            pred_prob = pred_prob.cpu().numpy()

        # 1. 自适应阈值
        binary_mask = self._adaptive_threshold(pred_prob)

        # 2. 形态学操作
        if self.use_morphology:
            binary_mask = self._morphology_refine(binary_mask)

        # 3. 去除小连通域
        binary_mask = self._remove_small_objects(binary_mask)

        # 4. 边界平滑
        if self.smooth_kernel > 0:
            binary_mask = self._smooth_boundary(binary_mask)

        # 5. CRF 优化
        if self.use_crf and image is not None:
            binary_mask = self._crf_refine(pred_prob, image, binary_mask)

        return binary_mask
    
    def _adaptive_threshold(self, pred_prob):
        # 如果预测很确定 (大部分接近 0 或 1)，用固定阈值
        if pred_prob.max() > 0.9 and pred_prob.min() < 0.1:
            threshold = self.threshold
        else:
            # 否则用 Otsu 自动阈值
            threshold = self._otsu_threshold(pred_prob)
        
        return (pred_prob >= threshold).astype(np.uint8)
    
    def _otsu_threshold(self, pred_prob):
        pred_uint8 = (pred_prob * 255).astype(np.uint8)
        threshold, _ = cv2.threshold(pred_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold / 255.0
    
    def _morphology_refine(self, binary_mask):
        # 1. 闭运算 - 填充小孔
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 2. 开运算 - 去除小噪点
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        
        return binary_mask
    
    def _remove_small_objects(self, binary_mask):
        # 使用 skimage 的 remove_small_objects
        binary_mask_bool = binary_mask.astype(bool)
        cleaned = morphology.remove_small_objects(binary_mask_bool, min_size=self.min_size)
        return cleaned.astype(np.uint8)
    
    def _smooth_boundary(self, binary_mask):
        # 如果 smooth_kernel <= 0，跳过平滑
        if self.smooth_kernel <= 0:
            return binary_mask

        # 核大小必须是奇数
        k = self.smooth_kernel if self.smooth_kernel % 2 == 1 else self.smooth_kernel + 1

        # 高斯模糊
        blurred = cv2.GaussianBlur(binary_mask.astype(np.float32), (k, k), 0)

        # 重新二值化
        smoothed = (blurred > 0.5).astype(np.uint8)

        return smoothed
    
    def _crf_refine(self, pred_prob, image, binary_mask):
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except ImportError:
            print("Warning: pydensecrf not installed, skipping CRF refinement")
            return binary_mask
        
        # 转换为 CRF 输入格式
        H, W = pred_prob.shape
        
        # 构建 unary potential
        prob_fg = pred_prob.flatten()
        prob_bg = 1 - prob_fg
        unary = np.vstack([prob_bg, prob_fg])
        unary = -np.log(unary + 1e-8)
        
        # 创建 CRF
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(unary.astype(np.float32))
        
        # 添加 pairwise potential
        # 1. 外观一致性 (appearance kernel)
        if image is not None:
            image_uint8 = (image * 255).astype(np.uint8)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_uint8, compat=10)
        else:
            d.addPairwiseGaussian(sxy=3, compat=3)
        
        # 推理
        Q = d.inference(5)
        map_result = np.argmax(Q, axis=0).reshape((H, W))
        
        return map_result.astype(np.uint8)


class TTAPredictor:
    def __init__(self, model, transforms=None, device='cuda'):
        self.model = model
        self.device = device
        self.transforms = transforms or ['hflip', 'vflip']

    def _transform(self, image, transform):
        """应用变换"""
        if transform == 'hflip':
            return torch.flip(image, dims=[3])
        elif transform == 'vflip':
            return torch.flip(image, dims=[2])
        elif transform == 'rotate90':
            return torch.rot90(image, k=1, dims=[2, 3])
        elif transform == 'rotate180':
            return torch.rot90(image, k=2, dims=[2, 3])
        elif transform == 'rotate270':
            return torch.rot90(image, k=3, dims=[2, 3])
        return image

    def _inverse_transform(self, pred, transform):
        """逆变换"""
        if transform == 'hflip':
            return torch.flip(pred, dims=[3])
        elif transform == 'vflip':
            return torch.flip(pred, dims=[2])
        elif transform == 'rotate90':
            return torch.rot90(pred, k=3, dims=[2, 3])
        elif transform == 'rotate180':
            return torch.rot90(pred, k=2, dims=[2, 3])
        elif transform == 'rotate270':
            return torch.rot90(pred, k=1, dims=[2, 3])
        return pred

    def predict(self, image):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            # 原始预测
            outputs = self.model(image)
            if isinstance(outputs, tuple):
                pred = outputs[0]
            else:
                pred = outputs
            pred = torch.sigmoid(pred)
            predictions.append(pred)

            # 变换预测
            for transform in self.transforms:
                # 变换
                transformed = self._transform(image, transform)
                # 预测
                outputs = self.model(transformed)
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                else:
                    pred = outputs
                pred = torch.sigmoid(pred)
                # 逆变换
                pred = self._inverse_transform(pred, transform)
                predictions.append(pred)

        # 平均融合
        avg_pred = torch.mean(torch.stack(predictions), dim=0)

        return avg_pred
