import os
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 引入你的数据集加载代码
from esc_scene.dataset import create_dataloader


def evaluate_tflite_model(tflite_path, dataset_dir, scene):
    print(f"Loading TFLite model: {tflite_path}")

    # 1. 加载 TFLite 解释器
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # 2. 获取输入输出详情
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 获取模型期望的输入尺寸 (例如: 32x32)
    input_h = input_details['shape'][1]
    input_w = input_details['shape'][2]

    # 输入节点的量化参数
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    print(f"\nModel Input Details:")
    print(f"  Shape: {input_details['shape']}")
    print(f"  Type: {input_details['dtype']}")
    print(f"  Quantization: Scale={input_scale}, ZeroPoint={input_zero_point}")

    # 3. 准备验证集
    dataloader, _ = create_dataloader(dataset_dir, scene, split='val', batch_size=1)

    all_preds = []
    all_labels = []

    print(f"\nStarting evaluation on {len(dataloader)} samples...")

    for i, (label, audio_tensor) in enumerate(dataloader):
        # audio_tensor 原始形状: [1, 1, 80, 501] (PyTorch Float32)

        # ------------------------------------------------------
        # 步骤 A: 维度转换 (NCHW -> NHWC)
        # ------------------------------------------------------
        # [1, 1, 80, 501] -> [1, 80, 501, 1]
        input_data = audio_tensor.numpy()
        input_data = np.transpose(input_data, (0, 2, 3, 1))

        # ------------------------------------------------------
        # [新增步骤]: 调整尺寸 (Resize) 到模型需要的 32x32
        # ------------------------------------------------------
        # 如果当前尺寸不是 32x32，则强行缩放
        if input_data.shape[1] != input_h or input_data.shape[2] != input_w:
            # 使用 TensorFlow 的 resize 功能 (支持双线性插值)
            input_data = tf.image.resize(input_data, [input_h, input_w])
            # 转回 numpy 以便进行下面的数学计算
            input_data = input_data.numpy()

        # ------------------------------------------------------
        # 步骤 B: 手动量化 (Float32 -> Int8)
        # ------------------------------------------------------
        if input_details['dtype'] == np.int8:
            input_data = (input_data / input_scale) + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)

        # ------------------------------------------------------
        # 步骤 C: 推理
        # ------------------------------------------------------
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])

        # ------------------------------------------------------
        # 步骤 D: 处理输出
        # ------------------------------------------------------
        if output_details['dtype'] == np.int8:
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        pred_label = np.argmax(output_data)

        all_preds.append(pred_label)
        all_labels.append(label.item())

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} samples...")

    # 4. 生成报告 (保持不变)
    target_names = ['Target 1', 'Target 2', 'Target 3', 'Background']
    if scene == 'ward':
        target_names = ['crying_baby', 'sneezing', 'coughing', 'background']
    elif scene == 'home':
        target_names = ['dog', 'crying_baby', 'door_knock', 'background']

    print("\n" + "=" * 60)
    print("TFLite Quantized Model Evaluation Report")
    print("=" * 60)

    print(classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        labels=[0, 1, 2, 3],
        zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nOverall Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    pass