import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T  # 用于 Resize
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class Esc50TargetDataset(Dataset):
    def __init__(self, root_dir, scene, split='train', target_sr=16000, fixed_length=80000):
        """
        Args:
            root_dir (str): ESC-50 数据集根目录
            scene (str): 场景模式 ('ward' 或 'home')
            split (str): 'train' 或 'val'
            target_sr (int): 目标采样率 (默认 16k)
            fixed_length (int): 目标采样点长度 (16k * 5s = 80000)
        """
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.fixed_length = fixed_length

        # 1. 定义目标类别映射
        if scene == 'ward':
            self.target_dict = {'crying_baby': 0, 'sneezing': 1, 'coughing': 2}
        elif scene == 'home':
            self.target_dict = {'dog': 0, 'crying_baby': 1, 'door_knock': 2}
        else:
            raise ValueError(f"Scene '{scene}' is not supported.")

        # 2. 加载并筛选文件列表
        self.file_list, self.labels = self._load_file_list(split)

    def _load_file_list(self, split):
        csv_file = os.path.join(self.root_dir, 'meta', 'esc50.csv')
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Meta file not found: {csv_file}")

        df = pd.read_csv(csv_file)

        # 划分训练集和验证集 (Fold 5 用于验证)
        if split == 'train':
            df = df[df['fold'] != 5]
        else:
            df = df[df['fold'] == 5]

        file_list = df['filename'].tolist()
        categories = df['category'].tolist()

        # 将非目标类别标记为 3 (Background)
        processed_labels = [self.target_dict.get(c, 3) for c in categories]

        return file_list, processed_labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, 'audio', self.file_list[idx])

        # 加载音频
        try:
            audio, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # 返回全0张量防止崩溃
            return self.labels[idx], torch.zeros(1, self.fixed_length)

        # 1. 重采样
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio = resampler(audio)

        # 2. 转单声道
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # 3. 固定长度 (截断或补零)
        current_len = audio.shape[-1]
        if current_len > self.fixed_length:
            audio = audio[..., :self.fixed_length]
        elif current_len < self.fixed_length:
            pad_amount = self.fixed_length - current_len
            audio = F.pad(audio, (0, pad_amount))

        return self.labels[idx], audio


class AudioCollate:
    """
    Collate 函数：负责特征提取、Resize 和 数据增广
    """

    def __init__(self, sample_rate=16000, training=True):
        self.training = training  # 标记是否为训练模式

        # 基础 Mel 变换 (保持高精度提取)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            power=2.0
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

        # 【关键修改】强制缩放到 32x32，对齐 TFLite 模型
        self.resize = T.Resize((32, 32))

        # 【增广 1】频域/时域遮罩 (SpecAugment)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def __call__(self, batch):
        labels = [item[0] for item in batch]
        audios = [item[1] for item in batch]

        # 1. 堆叠波形 [Batch, 1, Time]
        audio_batch = torch.stack(audios)

        # 【增广 2】高斯噪声注入 (仅训练时)
        # 在波形上加一点点随机噪声，提高鲁棒性
        if self.training and torch.rand(1) < 0.5:
            noise = torch.randn_like(audio_batch) * 0.005
            audio_batch += noise

        # 2. 提取特征 -> MelSpec [Batch, 1, 80, 501]
        melspec = self.mel_transform(audio_batch)
        melspec = self.to_db(melspec)

        # 【增广 3】应用 SpecAugment (仅训练时)
        # 随机遮挡部分频率或时间段，强迫模型不依赖单一特征
        if self.training:
            # 50% 概率应用
            if torch.rand(1) < 0.8:
                melspec = self.freq_mask(melspec)
                melspec = self.time_mask(melspec)

        # 3. 【关键】Resize 到 32x32
        # 输出: [Batch, 1, 32, 32]
        melspec = self.resize(melspec)

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return labels_tensor, melspec


def create_dataloader(root_dir, scene, split='train', batch_size=64, num_workers=4):
    """
    创建 DataLoader，如果 split='train'，会自动应用加权采样器来解决类别不平衡。
    """
    dataset = Esc50TargetDataset(root_dir, scene, split)

    # 根据 split 设置是否开启增广
    is_training = (split == 'train')
    collate_fn = AudioCollate(sample_rate=16000, training=is_training)

    sampler = None
    shuffle = False

    if is_training:
        print(f"--- Configuring WeightedRandomSampler for {scene} ---")
        # 1. 获取所有标签
        targets = dataset.labels
        targets_tensor = torch.tensor(targets, dtype=torch.long)

        # 2. 统计各类别样本数
        class_counts = torch.bincount(targets_tensor)
        print(f"Original Class Counts: {class_counts.tolist()}")

        # 3. 计算权重 (样本越少，权重越大)
        # 加上 1e-6 防止除以 0
        class_weights = 1.0 / (class_counts.float() + 1e-6)

        # 4. 为每个样本分配权重
        sample_weights = class_weights[targets_tensor]

        # 5. 创建采样器
        # replacement=True 允许重复抽样 (这是过采样的关键)
        # num_samples 设置为 len(dataset) 保证一个 epoch 的迭代次数与原来大致相同
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

        # 使用 sampler 时，shuffle 必须为 False
        shuffle = False
    else:
        # 验证集不需要采样器，也不需要 Shuffle (为了评估一致性)
        shuffle = False

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # 如果有 sampler，这里必须是 False
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 注意：不再返回 class_weights，因为 Sampler 已经解决了不平衡
    # 外部训练循环中的 CrossEntropyLoss 不需要再设置 weight 参数
    return dataloader, None