"""
ESC-50 Target Sound Detector
============================

这是一个基于 ESC-50 数据集的特定场景声音检测


"""

# 1. 版本与元数据信息
__version__ = "0.1.0"
__author__ = "Yu-m1ne"
__email__ = "Yu_m1ne@outlook.com"
__license__ = "Apache-2.0"

# 2. 导入核心组件
from .dataset import create_dataloader
from .model import Esc50TargetSoundDetector
from .train import train_model
from .val import val_model

# 3. 定义公开 API (__all__)
# 这决定了用户使用 `from package import *` 时会导入哪些内容，
# 同时也向 IDE 和阅读代码的人表明哪些是这个包的"公开接口"。
__all__ = [
    "create_dataloader",
    "Esc50TargetSoundDetector",
    "train_model",
    "val_model",
]