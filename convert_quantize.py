import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess

# 引入你的数据集模块
from esc_scene.dataset import create_dataloader

# ================= 配置区域 =================
# 1. 模型权重路径 (必须是 FCN 全卷积版权重的路径)
PTH_MODEL_PATH = "checkpoints/ward_model.pth"

# 2. 数据集路径
DATASET_ROOT = r"F:\Yu_m1ne\dataset\ESC-50-master"

# 3. 场景
SCENE = 'ward'

# 4. 输出目录
OUTPUT_DIR = r"./deploy_models"

# 5. 【关键】直接在此处定义 ESP32 最终需要的尺寸
# 这样导出的 ONNX 原生就是这个尺寸，不会报错
TARGET_HEIGHT = 32
TARGET_WIDTH = 32
INPUT_SHAPE = (1, 1, TARGET_HEIGHT, TARGET_WIDTH)


# ===========================================

# --- 再次声明 FCN 模型结构 (确保导出时用的是对的结构) ---
class Esc50NanoSoundDetector(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # GAP
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x


def step1_export_onnx_32x32(onnx_save_path):
    print(f"\n[1/3] Exporting ONNX with shape {INPUT_SHAPE}...")

    # 1. 初始化模型
    model = Esc50NanoSoundDetector(num_classes=4)

    # 2. 加载权重
    try:
        # map_location='cpu' 防止找不到 GPU 报错
        state_dict = torch.load(PTH_MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load weights ({e}). Exporting with random weights (Architecture check only).")

    model.eval()

    # 3. 创建 32x32 的 Dummy Input
    # 这是解决 "ResizeInputTensorStrict" 错误的根本方法
    dummy_input = torch.randn(*INPUT_SHAPE)

    # 4. 导出
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        # 不要使用 dynamic_axes，让形状固定为 32x32，对 TFLite Micro 最友好
    )
    print(f"✅ ONNX saved to: {onnx_save_path}")


def step2_onnx_to_tf(onnx_path, tf_save_path):
    print("\n[2/3] Converting ONNX to TensorFlow...")

    # onnx2tf 命令
    # 注意：这里不再需要 -ois 参数，因为 ONNX 本身已经是 32x32 了
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", tf_save_path,
        "--non_verbose",
        "-osd"  # 依然需要 Output Signature Defs
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("✅ TensorFlow SavedModel generated.")
    except subprocess.CalledProcessError:
        print("❌ onnx2tf conversion failed.")
        sys.exit(1)


def representative_dataset_gen(dataset_dir, scene, num_samples=100):
    """生成 32x32 的校准数据"""
    print(f"Generating calibration data (Resize to {TARGET_HEIGHT}x{TARGET_WIDTH})...")
    dataloader, _ = create_dataloader(dataset_dir, scene, split='val', batch_size=1)

    for i, (label, audio_tensor) in enumerate(dataloader):
        if i >= num_samples: break

        # 1. Resize PyTorch Tensor [1, 1, 80, 501] -> [1, 1, 32, 32]
        # 必须与 step1 中的导出尺寸一致
        audio_tensor = F.interpolate(
            audio_tensor,
            size=(TARGET_HEIGHT, TARGET_WIDTH),
            mode='bilinear',
            align_corners=False
        )

        # 2. Transpose to NHWC [1, 32, 32, 1]
        # onnx2tf 会自动处理模型维度的 NCHW->NHWC 转换
        # 所以校准数据也必须给 NHWC
        data_nhwc = audio_tensor.permute(0, 2, 3, 1).numpy()

        yield [data_nhwc.astype(np.float32)]


def step3_convert_tflite(tf_path, tflite_path):
    print("\n[3/3] Quantizing to Int8 TFLite...")

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    except Exception as e:
        print(f"❌ Failed to load TF model: {e}")
        return

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(DATASET_ROOT, SCENE)

    # 强制全整型
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"\n🎉 SUCCESS! Model saved to: {tflite_path}")
        print(f"📊 Final Size: {size_kb:.2f} KB")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    onnx_path = os.path.join(OUTPUT_DIR, "model_32x32.onnx")
    tf_path = os.path.join(OUTPUT_DIR, "model_tf")
    tflite_path = os.path.join(OUTPUT_DIR, "ward_model_int8.tflite")

    # 执行全流程
    step1_export_onnx_32x32(onnx_path)
    step2_onnx_to_tf(onnx_path, tf_path)
    step3_convert_tflite(tf_path, tflite_path)