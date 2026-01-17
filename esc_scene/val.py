import time
import torch
import torch.nn as nn
from .dataset import create_dataloader


def val(dataloader, model, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    start_time = time.time()

    # 关键：验证阶段不需要计算梯度
    with torch.no_grad():
        for labels, audios in dataloader:
            labels = labels.to(device)
            audios = audios.to(device)

            # 1. 前向传播
            outputs = model(audios)  # 输出形状: [Batch, 4, 1, 1]

            # === 关键修复 ===
            # 必须展平为 [Batch, 4]，否则 Loss 计算会报错
            outputs_flat = outputs.view(outputs.size(0), -1)

            # 2. 计算 Loss (使用展平后的 tensor)
            loss = criterion(outputs_flat, labels)

            # 3. 统计 Loss
            batch_size = audios.size(0)
            running_loss += loss.item() * batch_size

            # 4. 统计准确率 (使用展平后的 tensor)
            # torch.max 返回 (values, indices)，我们只需要 indices (preds)
            _, preds = torch.max(outputs_flat, 1)

            running_corrects += torch.sum(preds == labels).item()
            total_samples += batch_size

    end_time = time.time()

    if total_samples == 0:
        print("#Val# Dataset is empty!")
        return 0., 0., 0.

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    print(f'#Val# loss: {epoch_loss:.4f}, acc: {epoch_acc * 100:.2f}%, time taken: {end_time - start_time:.2f}s')

    return epoch_loss, epoch_acc, end_time - start_time


def val_model(model, dataset_dir, scene, device, batch_size=64):
    print(f'Preparing validation dataset (Scene: {scene})...')

    dataloader, _ = create_dataloader(
        dataset_dir,
        scene,
        split='val',
        batch_size=batch_size
    )

    # 验证集通常不加权，反映真实分布下的性能
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    return val(dataloader, model, criterion, device)