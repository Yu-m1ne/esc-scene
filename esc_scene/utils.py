import torch
import os


def save_checkpoint(model, save_dir, filename='esc50_model.pth'):
    """保存 PyTorch 权重 (.pth)"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    try:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to save model: {e}")
        return False


def export_onnx(model, device, save_dir, filename='esc50_model.onnx'):
    """
    导出为 ONNX 格式 (ESP32/嵌入式部署标准格式)
    输入形状: [1, 1, 80, 501]
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # 切换到 eval 模式至关重要
    model.eval()

    # 创建一个虚拟输入 (Batch=1, Channel=1, Freq=80, Time=501)
    dummy_input = torch.randn(1, 1, 80, 501).to(device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,  # 兼容性较好的版本
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX model exported to: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to export ONNX: {e}")
        return False