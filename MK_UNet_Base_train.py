from LoadData import *
from Data_Process import COCOSegmentationDataset
from MK_UNet_Baseline import MK_UNet_Baseline
from Loss import compute_hd95
import matplotlib.pyplot as plt
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
    # 这一行最重要，保证卷积算法也是固定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_pred(outputs, sigmoid_therehold = True):

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
    inter = (probs_bin * target).sum(dim=[1,2,3])  # per-batch
    union = probs_bin.sum(dim=[1,2,3]) + target.sum(dim=[1,2,3])
    dice = ((2 * inter + eps) / (union + eps)).mean().item()
    return dice

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()

            pred = extract_pred(outputs)  # 取主输出并转为概率
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
                loss = criterion(outputs, masks)

                pred = extract_pred(outputs)
                val_loss += loss.item()
                val_dice += dice_coeff_from_probs(pred, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # swanlab.log({
        #     'train/loss': train_loss,
        #     'train/dice': train_dice,
        #     'train/epoch': epoch + 1,
        #     'train/lr': optimizer.param_groups[0]['lr'],
        #     'val/loss': val_loss,
        #     'val/dice': val_dice,
        # }, step=epoch + 1)

        print(f'Epoch {epoch + 1} / {num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

def main():
    # swanlab.init(
    #     project="paper",
    #     experiment_name="MK-Unet-T-epochs200",
    #     config={
    #         "batch_size": 16,
    #         "learning_rate": 1e-4,
    #         "num_epochs": 100,
    #         "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    #     },
    # )

    batch_size = swanlab.config['batch_size']
    device = torch.device(swanlab.config['device'])

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集 (baseline 不使用数据增强)
    train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform, use_augmentation=False)
    val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform, use_augmentation=False)
    test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform, use_augmentation=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=batch_size)

    # 初始化模型
    model = MK_UNet_Baseline().to(device)

    # 设置优化器和学习率
    optimizer = optim.Adam(model.parameters(), lr=swanlab.config['learning_rate'])

    # 训练模型
    train_model(
        model = model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=combined_loss,
        optimizer=optimizer,
        num_epochs=swanlab.config['num_epochs'],
        device=device
    )

    # 在测试集上评估
    model.eval()
    best_threshold = 0.6
    test_loss = 0.0
    test_dice = 0.0
    test_hd95 = 0.0
    valid_samples = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss_output = combined_loss(outputs, masks)

            if isinstance(loss_output, tuple):
                loss = loss_output[0]
            else:
                loss = loss_output

            pred_prob = extract_pred(outputs)

            test_loss += loss.item()
            test_dice += dice_coeff_from_probs(pred_prob, masks)
            pred_binary = (pred_prob > best_threshold).float()
            batch_size = images.size(0)

            for i in range(batch_size):
                # 取出单张图，转为 numpy
                p_prob = pred_binary[i].squeeze().cpu().numpy()
                t = masks[i].squeeze().cpu().numpy()

                # 只有当这张图里真的有肿瘤（GT不为空）时，计算 HD95 才有意义
                if t.sum() > 0:
                    hd_val = compute_hd95(p_prob, t)
                    test_hd95 += hd_val
                    valid_samples += 1

    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    avg_hd95 = test_hd95 / valid_samples if valid_samples > 0 else 1

    print(f'Test Results -> Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, HD95: {avg_hd95:.4f}')
    # swanlab.log({"test/loss": test_loss, "test/dice": test_dice, "test/hd95": avg_hd95})
    # 可视化预测结果
    visualize_predictions(model, test_loader, device, num_samples=10)

# 图标可视化
def visualize_predictions(model, test_loader, device, num_samples, therehold = 0.6):
    image_num = 1
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(test_loader))
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        pred = extract_pred(outputs)
        binary_predictions = (pred > therehold).float()

        num_samples = min(num_samples, images.size(0))
        indices = list(range(num_samples))

        plt.figure(figsize=(16, 4 * num_samples))

        for i, idx in enumerate(indices):
            base = i * 4 # 每行4个子图

            # 原始图像
            plt.subplot(num_samples, 4, base + 1)
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')

            # 真实掩码
            plt.subplot(num_samples, 4, base + 2)
            plt.imshow(masks[idx].cpu().squeeze(), cmap='gray')
            plt.title('True Mask')
            plt.axis('off')

            # 预测掩码
            plt.subplot(num_samples, 4, base + 3)
            plt.imshow(binary_predictions[idx].cpu().squeeze(), cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            # 叠加对比
            plt.subplot(num_samples, 4, base + 4)
            plt.imshow(img) # 先显示原图
            # 添加红色半透明掩码
            plt.imshow(masks[idx].cpu().squeeze(), alpha=0.3, cmap='Greens')
            plt.imshow(binary_predictions[idx].cpu().squeeze(), alpha=0.3, cmap='Reds')
            plt.title('Overlay (Green=GT, Red=Pred)', fontsize=10)
            plt.axis('off')

            image_num += 1

            if (image_num == 5):
                break

        # 防止标题和图像重叠
        plt.tight_layout()

        # 记录图像到SwanLab
        # swanlab.log({'predictions' : swanlab.Image(plt)})

if __name__ == '__main__':
    seed_everything(42)
    main()