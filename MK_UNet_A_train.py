from LoadData import *
from Data_Process import COCOSegmentationDataset
from MK_UNet_A import AB_MKUnet
from Loss import compute_hd95
from refined_prediction import *
from Improved_loss import ImprovedBoundaryLoss
import random
import os
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_pred(outputs, sigmoid_therehold=True):
    if isinstance(outputs, (list, tuple)):
        pred = outputs[0]
    else:
        pred = outputs
    pred = pred.float()
    # 如果值不在 [0,1]，认为是 logits，应用 sigmoid
    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = torch.sigmoid(pred)
    return pred


def dice_coeff_from_probs(probs, target, eps=1e-6):
    probs_bin = (probs > 0.5).float()
    inter = (probs_bin * target).sum(dim=[1, 2, 3])  # per-batch
    union = probs_bin.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
    dice = ((2 * inter + eps) / (union + eps)).mean().item()
    return dice


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 处理模型输出 (分割预测, 边界预测)
            if isinstance(outputs, tuple):
                seg_pred, boundary_pred = outputs
            else:
                seg_pred = outputs
                boundary_pred = None

            loss, d_loss, b_loss = criterion(seg_pred, masks, boundary_pred)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()

            pred = extract_pred(seg_pred)
            train_loss += loss.item()
            train_dice += dice_coeff_from_probs(pred, masks)

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                # 处理模型输出
                if isinstance(outputs, tuple):
                    seg_pred, boundary_pred = outputs
                else:
                    seg_pred = outputs
                    boundary_pred = None

                loss, d_loss, b_loss = criterion(seg_pred, masks, boundary_pred)

                pred = extract_pred(seg_pred)
                val_loss += loss.item()
                val_dice += dice_coeff_from_probs(pred, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
                'train_dice': train_dice,
                'val_dice': val_dice,
            }, 'best_model.pth')

        swanlab.log({
            'train/loss': train_loss,
            'train/dice': train_dice,
            'train/epoch': epoch + 1,
            'train/lr': optimizer.param_groups[0]['lr'],
            'val/loss': val_loss,
            'val/dice': val_dice,
        }, step=epoch + 1)

        print(f'Epoch {epoch + 1} / {num_epochs}:')
        print(
            f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_dice': best_val_dice,
        'final_train_dice': train_dice,
        'final_val_dice': val_dice,
    }, 'final_model.pth')

    torch.save(model.state_dict(), 'final_model_weights_only.pth')

    # 加载最佳模型
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    torch.save(model.state_dict(), 'best_model_weights_only.pth')


def main():
    swanlab.init(
        project="paper",
        experiment_name="AB-MK-Unet-T-epochs100",
        config={
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "device": "cuda:3" if torch.cuda.is_available() else "cpu",
        },
    )

    batch_size = swanlab.config['batch_size']
    device = torch.device(swanlab.config['device'])

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集 (关闭数据增强，使用TTA替代)
    train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform, use_augmentation=False)
    val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform, use_augmentation=False)
    test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform, use_augmentation=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=batch_size)

    # 初始化模型
    model = AB_MKUnet().to(device)

    # 设置优化器和学习率
    optimizer = optim.Adam(model.parameters(), lr=swanlab.config['learning_rate'])

    # 使用改进的损失函数 (包含边界预测损失)
    criterion = ImprovedBoundaryLoss(
        alpha=0.5,
        focal_gamma=2.0,
        size_adaptive=True,
        use_hausdorff=False,
        hd_weight=0.1,
        boundary_weight=0.5  # 边界预测损失权重
    ).to(device)

    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=swanlab.config['num_epochs'],
        device=device
    )

    # 在测试集上评估 - 使用 TTA (测试时增强)
    model.eval()

    # 使用固定最优阈值 0.6
    best_threshold = 0.6

    # 使用最优阈值测试
    test_loss = 0.0
    test_dice = 0.0
    test_hd95 = 0.0
    valid_samples = 0

    # 创建 TTA 预测器
    tta_predictor = TTAPredictor(model, transforms=['hflip', 'vflip'], device=device)

    # 创建精细化预测器
    refiner = RefinedPredictor(
        threshold=best_threshold,
        use_morphology=False,
        use_crf=False,
        min_size=100,
        smooth_kernel=0
    )

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            # 使用 TTA 预测
            pred_prob = tta_predictor.predict(images)

            # 计算损失 (用原始模型输出)
            outputs = model(images)
            if isinstance(outputs, tuple):
                seg_pred, boundary_pred = outputs
            else:
                seg_pred = outputs
                boundary_pred = None

            loss, _, _ = criterion(seg_pred, masks, boundary_pred)

            test_loss += loss.item()
            # 使用最优阈值预测
            pred_binary_basic = (pred_prob > best_threshold).float()
            test_dice += dice_coeff_from_probs(pred_prob, masks)
            batch_size = images.size(0)

            for i in range(batch_size):
                # 取出单张图，转为 numpy
                # p_prob = pred_prob[i].squeeze().cpu().numpy()
                p_basic = pred_binary_basic[i].squeeze().cpu().numpy()
                t = masks[i].squeeze().cpu().numpy()

                if t.sum() > 0:
                    hd_basic = compute_hd95(p_basic, t)
                    test_hd95 += hd_basic
                    valid_samples += 1

    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    avg_hd95 = test_hd95 / valid_samples if valid_samples > 0 else 1

    print(f'Test Results -> Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, HD95: {avg_hd95:.4f}')

    swanlab.log({
        "test/loss": test_loss,
        "test/dice": test_dice,
        "test/hd95": avg_hd95
    })

    # 可视化预测结果 (对比基础和精细化)
    visualize_predictions_comparison(model, test_loader, device, refiner, num_samples=10)


# 图标可视化 - 对比基础和精细化预测
def visualize_predictions_comparison(model, test_loader, device, refiner, num_samples, threshold=0.5):
    image_num = 1
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(test_loader))
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        # 处理模型输出
        if isinstance(outputs, tuple):
            seg_pred, boundary_pred = outputs
        else:
            seg_pred = outputs
            boundary_pred = None

        pred = extract_pred(seg_pred)

        num_samples = min(num_samples, images.size(0))
        indices = list(range(num_samples))

        plt.figure(figsize=(16, 4 * num_samples))

        for i, idx in enumerate(indices):
            base = i * 4

            # 1. 原始图像
            plt.subplot(num_samples, 4, base + 1)
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            plt.imshow(img)
            plt.title('Original Image', fontsize=10)
            plt.axis('off')

            # 2. 真实掩码
            plt.subplot(num_samples, 4, base + 2)
            plt.imshow(masks[idx].cpu().squeeze(), cmap='gray')
            plt.title('Ground Truth', fontsize=10)
            plt.axis('off')

            # 4. 精细化预测
            plt.subplot(num_samples, 4, base + 3)
            pred_prob = pred[idx].squeeze().cpu().numpy()
            refined_mask = refiner(pred_prob)
            plt.imshow(refined_mask, cmap='gray')
            plt.title('Predicted Mask', fontsize=10)
            plt.axis('off')

            # 5. 叠加对比 (绿色=GT, 红色=精细化预测)
            plt.subplot(num_samples, 4, base + 4)
            plt.imshow(img)
            # GT 用绿色
            plt.imshow(masks[idx].cpu().squeeze(), alpha=0.3, cmap='Greens')
            # 精细化预测用红色
            plt.imshow(refined_mask, alpha=0.3, cmap='Reds')
            plt.title('Overlay (Green=GT, Red=Pred)', fontsize=10)
            plt.axis('off')

            image_num += 1

            if (image_num == 5):
                break

        plt.tight_layout()
        swanlab.log({'predictions_comparison': swanlab.Image(plt)})

if __name__ == '__main__':
    seed_everything(42)
    main()