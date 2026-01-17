import time
import torch
import torch.nn as nn
# 确保 dataset.py 包含上一轮提供的 create_dataloader
from .dataset import create_dataloader


def train_epoch(dataloader, model, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0  # 动态统计样本总数，替代硬编码的 1600

    start_time = time.time()

    for labels, audios in dataloader:
        # 1. 数据搬运
        labels = labels.to(device)
        audios = audios.to(device)  # Shape: [Batch, 1, 80, 501]

        # 2. 优化器清零
        optimizer.zero_grad()

        # 3. 前向传播
        outputs = model(audios)

        # 4. 计算损失
        # 确保展平为 [Batch, Num_Classes]
        loss = criterion(outputs.view(outputs.size(0), -1), labels)

        # 5. 反向传播与更新
        loss.backward()
        optimizer.step()

        # 6. 统计指标
        # loss.item() 是当前 batch 的平均 loss，乘以 batch_size 还原为总 loss
        batch_size = audios.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(outputs.view(outputs.size(0), -1), dim=1)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += batch_size

    end_time = time.time()

    # 7. 计算 Epoch 平均指标
    # 防止除以 0 (虽然训练时不太可能)
    if total_samples == 0:
        return 0., 0., 0.

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc, end_time - start_time


def train(dataloader, model, optimizer, criterion, device, epochs):
    total_time = 0.

    print(f"{'Epoch':^7} | {'Time':^7} | {'Train Loss':^12} | {'Train Acc':^12}")
    print("-" * 45)

    for epoch in range(epochs):
        epoch_loss, epoch_acc, epoch_time = train_epoch(dataloader, model, optimizer, criterion, device)
        total_time += epoch_time

        print(f"{epoch + 1:^7d} | {epoch_time:^6.2f}s | {epoch_loss:^12.4f} | {epoch_acc * 100:^11.2f}%")

    print("-" * 45)
    print(f'Total training time: {total_time:.2f} seconds')
    return model


def train_model(model, dataset_dir, scene, device='cpu', epochs=5, lr=0.001, batch_size=64):
    print(f'Preparing dataset (Scene: {scene})...')

    # --- 修改点 1: 解包 dataset 返回的权重 ---
    dataloader, class_weights = create_dataloader(
        dataset_dir,
        scene,
        split='train',
        batch_size=batch_size
    )
    print(f'The dataset is ready! {len(dataloader.dataset)} samples loaded.')

    # --- 修改点 2: 处理类别权重 ---
    # 如果有权重，将其移至 device 并传入 Loss 函数
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Applying Class Weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 确保模型在正确的设备上
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('Start training the model...')
    return train(dataloader, model, optimizer, criterion, device, epochs)