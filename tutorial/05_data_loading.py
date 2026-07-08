"""
╔══════════════════════════════════════════════════════════════╗
║        PyTorch 速成课 · 第5课：数据加载 DataLoader          ║
║        「怎么高效地把数据喂给模型」— Dataset & DataLoader     ║
╚══════════════════════════════════════════════════════════════╝

📌 核心概念：
  训练模型时，数据不会一次性全加载到内存（显存放不下！）。
  PyTorch 用 Dataset + DataLoader 实现：
    Dataset：定义「怎么读一个样本」（getitem）
    DataLoader：负责「批量打包、打乱、多线程预取」

  运行：python tutorial/05_data_loading.py
  练习：practice/05_custom_dataset.py
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ================================================================
# 1. TensorDataset — 最简单的 Dataset
# ================================================================
sep("1. TensorDataset — 把已有 Tensor 包装成 Dataset")

# 假设我们有 feature 和 label 两个 tensor
X = torch.randn(100, 5)        # 100 个样本，每个5维
y = torch.randint(0, 3, (100,))  # 100 个标签，3 分类

dataset = TensorDataset(X, y)

# Dataset 实现了 __len__ 和 __getitem__
print(f"len(dataset) = {len(dataset)}")         # 100
print(f"dataset[0]   = {dataset[0]}")            # 返回 tuple: (feature, label)
print(f"  feature: {dataset[0][0][:3]}...")      # 前3维
print(f"  label:   {dataset[0][1]}")

# TensorDataset 可以直接喂给 DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True)
batch_x, batch_y = next(iter(loader))
print(f"\n一个 batch 的 shape: x={batch_x.shape}, y={batch_y.shape}")


# ================================================================
# 2. 自定义 Dataset — 实现 __len__ + __getitem__
# ================================================================
sep("2. 自定义 Dataset — 只需实现两个方法")

class MyDataset(Dataset):
    """通用的自定义 Dataset 模板"""
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: 原始数据（numpy array, list, 文件路径列表…）
            labels: 标签
            transform: 可选的 transform 函数
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回第 idx 个样本。
        ⚠️ 这是 DataLoader 多线程调用的入口，要保证线程安全！
        """
        x = self.data[idx]
        y = self.labels[idx]

        # 数据预处理可以放在这里
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y

# 使用
raw_data = np.random.randn(100, 5).astype(np.float32)
raw_labels = np.random.randint(0, 3, 100)

dataset = MyDataset(raw_data, raw_labels)
print(f"len(dataset) = {len(dataset)}")
print(f"dataset[0] = (shape={dataset[0][0].shape}, label={dataset[0][1]})")


# ================================================================
# 3. DataLoader — 核心参数
# ================================================================
sep("3. DataLoader — 批处理、打乱、多进程")

dataset = TensorDataset(torch.randn(100, 5), torch.randint(0, 3, (100,)))

# 核心参数一览
loader = DataLoader(
    dataset,
    batch_size=16,       # 每批多少个样本
    shuffle=True,         # 每个 epoch 随机打乱
    num_workers=0,        # 子进程数（0 = 主进程加载；>0 可加速但要注意线程安全）
    drop_last=False,      # 最后不足一个 batch 是否丢弃
    pin_memory=False,     # 锁页内存，加速 CPU→GPU 传输（有 GPU 时推荐 True）
)

print(f"batch_size=16:")
for batch_idx, (x, y) in enumerate(loader):
    print(f"  Batch {batch_idx}: x.shape={list(x.shape)}, y.shape={list(y.shape)}")
print(f"  共 {len(loader)} 个 batch (100/16 = 6.25，不丢末尾就是7个)")

# drop_last=True 的效果
loader_drop = DataLoader(dataset, batch_size=16, drop_last=True)
print(f"\ndrop_last=True: {len(loader_drop)} 个 batch (末尾不足16的丢弃)")


# ================================================================
# 4. 数据预处理 — torchvision.transforms
# ================================================================
sep("4. 数据预处理 — transforms (torchvision)")

try:
    import torchvision.transforms as T

    # transforms.Compose 把多个 transform 串起来
    transform = T.Compose([
        T.Resize((224, 224)),    # 统一尺寸
        T.RandomHorizontalFlip(), # 随机水平翻转（数据增强！）
        T.ToTensor(),             # PIL Image / numpy → Tensor + 归一化到[0,1]
        T.Normalize(              # 标准化：减去均值，除以标准差
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    print("transforms.Compose 示例:")
    print("  Resize(224) → RandomHorizontalFlip → ToTensor → Normalize")
    print("  (这是 ImageNet 预训练模型的标准预处理流程)")
except ImportError:
    print("torchvision.transforms 需要 torchvision 库")
    print("(已安装，此提示不应出现)")

# 你也可以写自己的 transform（就是普通函数）
normalize = lambda x: (x - x.mean()) / x.std()
x = torch.randn(100, 5)
print(f"\n自定义 transform: 标准化后 mean={normalize(x).mean():.6f}, std={normalize(x).std():.6f}")


# ================================================================
# 5. 常见数据集场景
# ================================================================
sep("5. 常见 Dataset 场景")

print("""
  场景 1 — 数据全在内存（numpy/tensor）:
    → TensorDataset(X, y) 直接用

  场景 2 — 数据在文件系统（图片、文本）:
    → 自定义 Dataset，__getitem__ 里读文件
      def __getitem__(self, idx):
          path = self.file_list[idx]
          img = Image.open(path)
          return self.transform(img), self.labels[idx]

  场景 3 — 数据太大放不下内存:
    → 同上，按需从磁盘读取即可，DataLoader 会自动并行

  场景 4 — torchvision 内置数据集:
    → torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    → torchvision.datasets.ImageFolder(root='./images/')  # 按文件夹分类

  场景 5 — 文本数据:
    → torchtext / HuggingFace datasets
""")


# ================================================================
# 6. 训练循环中的 DataLoader
# ================================================================
sep("6. 训练循环中的典型用法")

dataset = TensorDataset(torch.randn(100, 5), torch.randint(0, 3, (100,)))
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("典型的训练 epoch 循环：")
print("""
  for epoch in range(num_epochs):
      model.train()
      for batch_idx, (x, y) in enumerate(train_loader):
          x, y = x.to(device), y.to(device)   # 搬到 GPU
          optimizer.zero_grad()
          pred = model(x)
          loss = criterion(pred, y)
          loss.backward()
          optimizer.step()

      # 每个 epoch 后验证
      model.eval()
      with torch.no_grad():
          for x, y in val_loader:
              ...
""")


# ================================================================
# 7. 📝 核心速查
# ================================================================
sep("7. 📝 核心速查")

print("""
  Dataset：
    TensorDataset(X, y)           → 数据已在内存中
    class MyDataset(Dataset):     → 自定义，实现 __len__ + __getitem__

  DataLoader 关键参数：
    batch_size=32                 → 批大小
    shuffle=True                  → 每个 epoch 打乱（训练时 True，验证时 False）
    num_workers=4                 → 多进程加载（Windows下可能需设为0）
    drop_last=True                → 丢弃最后不完整的 batch（训练时常用，BatchNorm需要）
    pin_memory=True               → 加速 CPU→GPU（有 GPU 时开）

  transforms (torchvision):
    transforms.Compose([...])     → 串联多个 transform
    transforms.ToTensor()         → 转 Tensor + 归一化
    transforms.Normalize(mean,std)→ 标准化
    transforms.RandomHorizontalFlip() → 数据增强

  👉 下一步：去 practice/05_custom_dataset.py 刷题！
""")
