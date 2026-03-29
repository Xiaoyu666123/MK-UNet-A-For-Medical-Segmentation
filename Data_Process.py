import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import scipy.ndimage as ndi

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
    _HAS_COCO = True
except Exception:
    _HAS_COCO = False

class BrainTumorAugmentation:

    @staticmethod
    def elastic_transform(image, mask, alpha=30, sigma=4):

        if random.random() > 0.5:  # 50% 概率应用
            return image, mask

        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        h, w = image.shape[:2]

        # 生成随机位移场
        dx = ndi.gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = ndi.gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha

        # 生成网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # 应用变形
        image_warped = np.zeros_like(image)
        for c in range(image.shape[2]):
            image_warped[:, :, c] = ndi.map_coordinates(image[:, :, c], indices, order=1).reshape(h, w)

        mask_warped = ndi.map_coordinates(mask.astype(np.float32), indices, order=0).reshape(h, w)

        if image_warped.shape[2] == 1:
            image_warped = image_warped[:, :, 0]

        return image_warped, mask_warped

    @staticmethod
    def random_rotation(image, mask, angles=[0, 90, 180, 270]):
        angle = random.choice(angles)
        if angle == 0:
            return image, mask

        image = TF.rotate(Image.fromarray(image), angle)
        mask = TF.rotate(Image.fromarray(mask), angle, interpolation=Image.NEAREST)

        return np.array(image), np.array(mask)

    @staticmethod
    def random_scale(image, mask, scale_range=(0.8, 1.2)):
        scale = random.uniform(*scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        image_scaled = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
        mask_scaled = np.array(Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST))

        # 如果缩放后尺寸变化，需要填充或裁剪
        if scale < 1.0:
            # 填充
            pad_h = h - new_h
            pad_w = w - new_w
            image_scaled = np.pad(image_scaled, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0, 0)), mode='constant')
            mask_scaled = np.pad(mask_scaled, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')
        elif scale > 1.0:
            # 中心裁剪
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            if image_scaled.ndim == 3:
                image_scaled = image_scaled[start_h:start_h+h, start_w:start_w+w]
            else:
                image_scaled = image_scaled[start_h:start_h+h, start_w:start_w+w]
            mask_scaled = mask_scaled[start_h:start_h+h, start_w:start_w+w]

        return image_scaled, mask_scaled

    @staticmethod
    def brightness_contrast(image, brightness_range=0.1, contrast_range=0.1):
        if random.random() > 0.5:
            brightness = 1 + random.uniform(-brightness_range, brightness_range)
            contrast = 1 + random.uniform(-contrast_range, contrast_range)

            image = TF.adjust_brightness(Image.fromarray(image), brightness)
            image = TF.adjust_contrast(image, contrast)

            return np.array(image)
        return image


class COCOSegmentationDataset(Dataset):
    def __init__(self, coco, images_dir, mask_dir=None, resize=(256, 256), transform=None,
                 use_augmentation=True):
        self.coco = coco if _HAS_COCO and hasattr(coco, 'getAnnIds') else None
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.resize = resize
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.augmentor = BrainTumorAugmentation()

        if self.coco:
            imgs = list(self.coco.imgs.values())
            self.ids = [img['id'] for img in imgs]
            self.image_paths = [os.path.join(self.images_dir, img['file_name']) for img in imgs]
        else:
            patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
            files = []
            for p in patterns:
                files.extend(glob.glob(os.path.join(self.images_dir, p)))
            files = sorted(files)
            self.image_paths = files
            if mask_dir is None:
                self.mask_paths = [None] * len(self.image_paths)
            else:
                self.mask_paths = [os.path.join(self.mask_dir, os.path.basename(p)) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def _build_mask_from_coco(self, img_id, h, w):
        if not _HAS_COCO:
            return np.zeros((h, w), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann:
                try:
                    m = self.coco.annToMask(ann)
                except Exception:
                    seg = ann.get('segmentation', None)
                    if isinstance(seg, dict) and 'counts' in seg:
                        m = maskUtils.decode(seg)
                    else:
                        m = None
                if m is not None:
                    if m.shape != mask.shape:
                        m = np.array(Image.fromarray(m).resize((w, h), resample=Image.NEAREST))
                    mask = np.logical_or(mask, m)
        return mask.astype(np.uint8)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        mask = None
        if not self.coco and self.mask_dir is not None:
            mask_path = self.mask_paths[idx]
            if mask_path is not None and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
        elif self.coco:
            img_id = self.ids[idx]
            w, h = image.size
            m = self._build_mask_from_coco(img_id, h, w)
            mask = Image.fromarray((m * 255).astype(np.uint8))

        if mask is None:
            mask = Image.new('L', image.size, 0)

        if self.resize is not None:
            image = image.resize(self.resize, resample=Image.BILINEAR)
            mask = mask.resize(self.resize, resample=Image.NEAREST)

        # 转为 numpy 进行增强
        image_np = np.array(image)
        mask_np = np.array(mask)

        # 数据增强 (仅训练时使用)
        if self.use_augmentation:
            # 1. 弹性变形 (对边界学习很重要)
            image_np, mask_np = self.augmentor.elastic_transform(image_np, mask_np)

            # 2. 随机旋转
            image_np, mask_np = self.augmentor.random_rotation(image_np, mask_np)

            # 3. 随机缩放
            if random.random() < 0.3:  # 30% 概率
                image_np, mask_np = self.augmentor.random_scale(image_np, mask_np)

            # 4. 亮度对比度
            image_np = self.augmentor.brightness_contrast(image_np)

        # 水平翻转
        if random.random() < 0.5:
            image_np = np.fliplr(image_np).copy()
            mask_np = np.fliplr(mask_np).copy()

        image = TF.to_tensor(image_np)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = TF.to_tensor(mask_np)
        mask = (mask > 0.5).float()

        return image, mask