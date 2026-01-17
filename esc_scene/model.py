import torch
import torch.nn as nn


class Esc50TargetSoundDetector(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1: [B, 1, 80, 501] -> [B, 8, 40, 125]
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),

            # Layer 2: [B, 8, 40, 125] -> [B, 16, 20, 31]
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),

            # Layer 3: [B, 16, 20, 31] -> [B, 32, 20, 31]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Global Average Pooling: [B, 32, 20, 31] -> [B, 32, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # === 终极修改：使用 1x1 卷积代替全连接层 ===
        # 不再使用 Flatten() 和 Linear()
        # 输入: [B, 32, 1, 1]
        # 卷积核: 1x1
        # 输出: [B, 32, 1, 1] -> [B, num_classes, 1, 1]
        self.classifier = nn.Sequential(
            # 相当于原来的 Linear(32, 32)
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            # 相当于原来的 Linear(32, num_classes)
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = self.classifier(x)

        # 输出形状是 [B, 4, 1, 1]
        # 我们不需要在这里 Flatten，直接返回 4D 张量
        # TFLite C++ 读取时，内存是连续的，所以读取 data.int8[0]~[3] 依然有效
        return x

# ==========================================
# 验证脚本
# ==========================================
if __name__ == '__main__':
    model = Esc50TargetSoundDetector(num_classes=4)

    # 1. 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    # 2. 估算 int8 体积
    # 1 param = 1 byte. TFLite header ≈ 2000 bytes.
    est_size_kb = (total_params + 2000) / 1024
    print(f"Estimated int8 TFLite Size: ~{est_size_kb:.2f} KB")

    # 3. 维度测试
    dummy_input = torch.randn(1, 1, 80, 501)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")