import torch
import os

# 导入我们之前写好的模块
# 确保 esc_scene 文件夹里有 __init__.py (可以是空的)，或者这些文件在当前目录下
from esc_scene.model import Esc50TargetSoundDetector
from esc_scene.train import train_model
from esc_scene.val import val_model
from esc_scene.utils import save_checkpoint, export_onnx


class Config:
    # 基础配置
    lr = 0.001
    batch_size = 64
    epochs = 50
    scene = 'ward'  # 或 'home'

    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径配置 (使用 raw string r'' 避免 Windows 路径转义问题)
    dataset_root = r'F:\Yu_m1ne\dataset\ESC-50-master'
    save_dir = r'./checkpoints'


if __name__ == '__main__':
    # 1. 初始化模型
    print(f"Initializing model on {Config.device}...")
    model = Esc50TargetSoundDetector(num_classes=4)  # 这里的类数要对应 dataset 的定义

    # 2. 训练模型
    # train_model 内部会自动处理 dataset加载、class_weights 和 优化器
    model = train_model(
        model=model,
        dataset_dir=Config.dataset_root,
        scene=Config.scene,
        device=Config.device,
        epochs=Config.epochs,
        lr=Config.lr,
        batch_size=Config.batch_size
    )

    # 3. 验证模型
    print("\nStarting Validation...")
    loss, acc, time_taken = val_model(
        model=model,
        dataset_dir=Config.dataset_root,
        scene=Config.scene,
        device=Config.device,
        batch_size=Config.batch_size
    )

    print(f"\nFinal Result -> Loss: {loss:.4f}, Acc: {acc * 100:.2f}%")

    # 4. 保存与导出
    # 保存 PyTorch 权重 (.pth) 用于后续微调或 Python 推理
    if save_checkpoint(model, Config.save_dir, f'{Config.scene}_model.pth'):
        # 导出 ONNX (.onnx) 用于 ESP32/TensorFlow 转换
        # 注意：这里替代了原来的 save2tf，因为 PyTorch 直接转 TF 很麻烦，ONNX 是标准中间件
        export_onnx(model, Config.device, Config.save_dir, f'{Config.scene}_model.onnx')
    else:
        raise RuntimeError('Error saving model!')
