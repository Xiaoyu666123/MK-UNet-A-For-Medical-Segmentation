import argparse
import os
from typing import Dict, Tuple
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from Loss import compute_hd95
from MK_UNet_A import AB_MKUnet


class COCOTestDataset(Dataset):
    """Deterministic COCO test dataset without random augmentations."""

    def __init__(self, annotation_file: str, images_dir: str, resize: Tuple[int, int] = (256, 256)):
        self.coco = COCO(annotation_file)
        self.images_dir = images_dir
        self.resize = resize
        self.ids = [img["id"] for img in self.coco.imgs.values()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        image_path = os.path.join(self.images_dir, img_info["file_name"])

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        mask = np.zeros((h, w), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            m = self.coco.annToMask(ann)
            mask = np.logical_or(mask, m)

        mask = (mask.astype(np.uint8) * 255)
        mask = Image.fromarray(mask)

        if self.resize is not None:
            image = image.resize(self.resize, resample=Image.BILINEAR)
            mask = mask.resize(self.resize, resample=Image.NEAREST)

        image = TF.to_tensor(np.array(image))
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = TF.to_tensor(np.array(mask))
        mask = (mask > 0.5).float()

        return image, mask, img_info["file_name"]


def extract_pred(outputs: torch.Tensor) -> torch.Tensor:
    if isinstance(outputs, (tuple, list)):
        pred = outputs[0]
    else:
        pred = outputs
    pred = pred.float()
    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = torch.sigmoid(pred)
    return pred


def dice_coeff_from_probs(probs: torch.Tensor, target: torch.Tensor, threshold: float, eps: float = 1e-6) -> float:
    probs_bin = (probs > threshold).float()
    inter = (probs_bin * target).sum(dim=[1, 2, 3])
    union = probs_bin.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
    dice = ((2 * inter + eps) / (union + eps)).mean().item()
    return dice


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # If dict values look like tensors, treat as plain state_dict.
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("[Warning] Loaded checkpoint with strict=False. Please verify model config matches training.")


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset = COCOTestDataset(
        annotation_file=args.annotation,
        images_dir=args.test_dir,
        resize=(args.image_size, args.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = AB_MKUnet().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True) if args.save_dir else None

    total_dice = 0.0
    total_hd95 = 0.0
    valid_hd95 = 0

    with torch.no_grad():
        for images, masks, file_names in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            probs = extract_pred(outputs)
            preds = (probs > args.threshold).float()

            total_dice += dice_coeff_from_probs(probs, masks, args.threshold)

            for i in range(images.size(0)):
                pred_np = preds[i].squeeze().cpu().numpy().astype(np.uint8)
                gt_np = masks[i].squeeze().cpu().numpy().astype(np.uint8)

                if gt_np.sum() > 0:
                    total_hd95 += compute_hd95(pred_np, gt_np)
                    valid_hd95 += 1

                if args.save_dir:
                    pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
                    out_name = os.path.splitext(file_names[i])[0] + "_pred.png"
                    pred_img.save(os.path.join(args.save_dir, out_name))

    avg_dice = total_dice / len(loader) if len(loader) > 0 else 0.0
    avg_hd95 = total_hd95 / valid_hd95 if valid_hd95 > 0 else 0.0

    return {
        "dice": avg_dice,
        "hd95": avg_hd95,
        "num_images": float(len(dataset)),
        "num_valid_hd95": float(valid_hd95),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MKUNet-A with a .pth checkpoint on COCO test set.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Directory containing test images")
    parser.add_argument(
        "--annotation",
        type=str,
        default="dataset/test/_annotations.coco.json",
        help="COCO annotation JSON for test set",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Resize H/W for testing")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--threshold", type=float, default=0.6, help="Binary threshold for mask prediction")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save-dir", type=str, default="", help="Optional directory to save predicted masks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)

    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test images: {int(metrics['num_images'])}")
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"HD95: {metrics['hd95']:.4f} (valid samples: {int(metrics['num_valid_hd95'])})")
    if args.save_dir:
        print(f"Saved prediction masks to: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()