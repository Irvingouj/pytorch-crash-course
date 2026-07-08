"""
╔══════════════════════════════════════════════════════════════╗
║         PyTorch 速成课 · 第6课：训练流程实战                 ║
║         「把前面所有知识串起来」— 从零训练一个分类器          ║
╚══════════════════════════════════════════════════════════════╝

📌 核心概念：
  这一课把前面学的 Tensor、Autograd、nn.Module、Loss、Optimizer、DataLoader
  串成一个完整的训练流程。我们用合成数据训练一个简单的分类器，
  让你看到「模型从瞎猜到学会」的全过程。

  运行：python tutorial/06_training_loop.py
  练习：practice/06_training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ================================================================
# 1. 准备数据（合成数据，方便演示）
# ================================================================
sep("1. 准备数据 — 合成一个三分类数据集")

torch.manual_seed(42)

# 生成 3 类数据，每类围绕不同的中心点
def make_data(n_per_class=200, n_features=2):
    X_list, y_list = [], []
    centers = [(0, 0), (5, 5), (0, 8)]  # 3个类的中心
    for i, (cx, cy) in enumerate(centers):
        # 每个类围绕中心点生成，加上随机噪声
        X = torch.randn(n_per_class, n_features)
        X[:, 0] += cx
        X[:, 1] += cy
        X_list.append(X)
        y_list.append(torch.full((n_per_class,), i))
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y

X, y = make_data()
print(f"数据集: X.shape={X.shape}, y.shape={y.shape}")
print(f"类别分布: {torch.bincount(y).tolist()}")

# 划分训练/验证集
n_train = int(0.8 * len(X))
indices = torch.randperm(len(X))
X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
X_val, y_val = X[indices[n_train:]], y[indices[n_train:]]

print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")

# 包装成 DataLoader
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=64, shuffle=False
)


# ================================================================
# 2. 定义模型
# ================================================================
sep("2. 定义模型 — 一个简单 MLP")

class Classifier(nn.Module):
    def __init__(self, input_dim=2, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
print(f"模型在: {device}")
print(f"模型结构:\n{model}")
print(f"参数量: {sum(p.numel() for p in model.parameters())}")


# ================================================================
# 3. 定义 Loss 和 Optimizer
# ================================================================
sep("3. 配置 Loss & Optimizer")

criterion = nn.CrossEntropyLoss()   # 多分类
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"Loss: CrossEntropyLoss")
print(f"Optimizer: Adam(lr=0.01)")


# ================================================================
# 4. 训练循环
# ================================================================
sep("4. 训练 — 每个 epoch 训练 + 验证")

def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回平均 loss 和准确率"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # 四步走
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * x.size(0)
        correct += (pred.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型，返回 loss 和准确率"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        total_loss += loss.item() * x.size(0)
        correct += (pred.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


# 开始训练！
num_epochs = 30
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

print(f"\n训练完成！最终验证准确率: {val_accs[-1]:.3f}")


# ================================================================
# 5. 可视化训练过程
# ================================================================
sep("5. 可视化 — Loss 曲线 & 准确率曲线")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train')
ax1.plot(val_losses, label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, label='Train')
ax2.plot(val_accs, label='Val')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=100)
print("图片已保存到 training_curves.png")
print("(如果没弹出窗口，这是正常的，图片已保存到文件)")


# ================================================================
# 6. 可视化决策边界
# ================================================================
sep("6. 决策边界 — 看看模型学到了什么")

@torch.no_grad()
def plot_decision_boundary(model, X, y, device):
    model.eval()

    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 200),
        torch.linspace(y_min, y_max, 200),
        indexing='ij'
    )
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1).to(device)
    pred = model(grid).argmax(dim=1).cpu().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, pred, alpha=0.3, cmap='viridis')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=30)
    ax.set_title('Decision Boundary')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.tight_layout()
    plt.savefig('decision_boundary.png', dpi=100)
    print("图片已保存到 decision_boundary.png")

plot_decision_boundary(model, X, y, device)


# ================================================================
# 7. 📝 核心速查
# ================================================================
sep("7. 📝 核心速查")

print("""
  完整训练流程 = 以下步骤的循环：

  ① 准备数据
      Dataset + DataLoader(batch_size, shuffle)

  ② 定义模型
      class MyModel(nn.Module):
          ...

  ③ 选 Loss + Optimizer
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.001)

  ④ 训练循环
      for epoch in range(num_epochs):
          # 训练
          model.train()
          for x, y in train_loader:
              x, y = x.to(device), y.to(device)
              optimizer.zero_grad()
              pred = model(x)
              loss = criterion(pred, y)
              loss.backward()
              optimizer.step()

          # 验证
          model.eval()
          with torch.no_grad():
              for x, y in val_loader:
                  ...

  训练 vs 验证的关键区别：
    train: model.train() + optimizer.zero_grad + loss.backward + optimizer.step
    eval:  model.eval()  + torch.no_grad()                        (没有梯度/更新)

  👉 下一步：去 practice/06_training.py 刷题！
""")
