"""
╔══════════════════════════════════════════════════════════════╗
║        PyTorch 速成课 · 第3课：神经网络模块 nn.Module        ║
║         「像搭乐高一样搭网络」— 一切网络层的基类              ║
╚══════════════════════════════════════════════════════════════╝

📌 核心概念：
  nn.Module 是所有神经网络的基类。你把网络的每一层定义为它的属性，
  然后实现 forward() 定义数据怎么流。PyTorch 自动帮你：
  - 收集所有可训练参数 (parameters())
  - 处理反向传播（你只要定义前向！）
  - 管理设备迁移 (to(device))
  - 保存/加载 (state_dict())

  运行：python tutorial/03_nn_module.py
  练习：practice/03_network_building.py
"""

import torch
import torch.nn as nn


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ================================================================
# 1. 最简单的 nn.Module
# ================================================================
sep("1. 最简网络 — 继承 nn.Module，实现 __init__ + forward")

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 把层定义为 self 的属性，nn.Module 会自动发现它们
        self.fc1 = nn.Linear(10, 20)   # 输入10维 → 输出20维
        self.fc2 = nn.Linear(20, 5)    # 输入20维 → 输出5维
        self.relu = nn.ReLU()

    def forward(self, x):
        # forward 定义数据流，你只管前向，backward 自动生成！
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
print(model)

# 看看参数长什么样
print(f"\nfc1.weight shape: {model.fc1.weight.shape}")  # (20, 10)
print(f"fc1.bias shape:   {model.fc1.bias.shape}")      # (20,)


# ================================================================
# 2. 前向传播 — 像调函数一样调模型
# ================================================================
sep("2. 前向传播 — model(x) 等价于 model.forward(x)")

x = torch.randn(3, 10)    # batch_size=3, input_dim=10
out = model(x)
print(f"输入 shape: {x.shape}")
print(f"输出 shape: {out.shape}   # (3, 5)，每行是一个样本的预测")

# ⚠️ 永远不要直接调 model.forward(x)！
# 调 model(x) 会走 __call__，它会处理 hooks 等额外逻辑


# ================================================================
# 3. 常用层一览
# ================================================================
sep("3. 常用层速览")

# --- 线性层（全连接）---
linear = nn.Linear(in_features=4, out_features=3)
print(f"nn.Linear(4, 3): weight={linear.weight.shape}, bias={linear.bias.shape}")

# --- 激活函数 ---
print(f"nn.ReLU()     : max(0, x)")
print(f"nn.Sigmoid()  : 1/(1+e^{-x})  → 压缩到 (0,1)")
print(f"nn.Tanh()     : 双曲正切 → 压缩到 (-1,1)")
print(f"nn.Softmax(dim=1): e^x / Σe^x → 概率分布（多分类最后一步）")

# 演示 softmax
logits = torch.tensor([[2.0, 1.0, 0.1]])
print(f"\nlogits = {logits}")
print(f"Softmax = {torch.softmax(logits, dim=1)}   # 变成概率，和为1")

# --- Dropout（防过拟合）---
dropout = nn.Dropout(p=0.5)  # 训练时随机丢弃50%神经元
# 训练模式：会随机置零
dropout.train()
x = torch.ones(1, 6)
print(f"\nDropout.train(): {dropout(x)}  # ~一半是0，其余是2（保持期望不变）")
# 评估模式：不做任何事
dropout.eval()
print(f"Dropout.eval():  {dropout(x)}  # 原样输出")

# --- BatchNorm ---
bn = nn.BatchNorm1d(num_features=4)
print(f"\nnn.BatchNorm1d(4): 对每个特征做标准化，让训练更稳定")


# ================================================================
# 4. Sequential — 简单网络不用写类
# ================================================================
sep("4. nn.Sequential — 「搭积木」式写法")

# 层按顺序执行，不需要写 forward
seq_net = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

x = torch.randn(3, 10)
out = seq_net(x)
print(f"Sequential 网络: {seq_net}")
print(f"输出 shape: {out.shape}")

# 也可以用 OrderedDict 给每层起名字
from collections import OrderedDict
named_net = nn.Sequential(OrderedDict([
    ('hidden', nn.Linear(10, 20)),
    ('activation', nn.ReLU()),
    ('output', nn.Linear(20, 5)),
]))
print(f"\n命名版 Sequential: {named_net}")
print(f"通过名字访问: {named_net.hidden}")


# ================================================================
# 5. parameters() — 模型有哪些可训练参数
# ================================================================
sep("5. parameters() — 自动收集所有可训练参数")

model = SimpleNet()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数数量: {total_params}")
print(f"可训练参数: {trainable_params}")

# 打印每层参数
for name, param in model.named_parameters():
    print(f"  {name:25s}  shape={str(param.shape):15s}  requires_grad={param.requires_grad}")


# ================================================================
# 6. train() vs eval() — 模式切换
# ================================================================
sep("6. train() / eval() — 影响 Dropout、BatchNorm 等层的行为")

model = nn.Sequential(
    nn.Linear(4, 4),
    nn.BatchNorm1d(4),
    nn.Dropout(0.5),
)

# 训练模式
model.train()
x = torch.randn(2, 4)
out_train = model(x)
print(f"train 模式: Dropout 生效，BatchNorm 用当前 batch 的统计量")

# 评估模式
model.eval()
with torch.no_grad():
    out_eval = model(x)
print(f"eval  模式: Dropout 关闭，BatchNorm 用训练时的 running mean/var")

print(f"\n输出在两种模式下不同: {not torch.allclose(out_train, out_eval)}")


# ================================================================
# 7. 自定义层 — 当标准层不够用
# ================================================================
sep("7. 自定义层 — 继承 nn.Module，实现任意计算")

class MyCustomLayer(nn.Module):
    """一个简单的自定义层：y = w * x + b，但权重永远为正"""
    def __init__(self, in_features, out_features):
        super().__init__()
        # 用 Parameter 包装，会被自动收集
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # 保证权重为正（只是一个演示，实际中可能用 softplus）
        return x @ torch.abs(self.weight).T + self.bias

layer = MyCustomLayer(4, 3)
x = torch.randn(2, 4)
out = layer(x)
print(f"MyCustomLayer: {x.shape} -> {out.shape}")
print(f"参数被自动收集: {list(layer.parameters())}")

# nn.Parameter vs 普通 Tensor：
# - nn.Parameter 是 Tensor 的子类
# - 被赋值为 Module 属性时，自动加入 parameters()
# - 普通 Tensor 不会被收集


# ================================================================
# 8. 一个完整的 MLP（多层感知机）示例
# ================================================================
sep("8. 完整示例 — 一个可用的 MLP")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

mlp = MLP(input_dim=100, hidden_dims=[64, 32], output_dim=10, dropout=0.3)
x = torch.randn(16, 100)
out = mlp(x)
print(f"MLP(100→64→32→10): 输入 {x.shape} → 输出 {out.shape}")
print(f"总参数: {sum(p.numel() for p in mlp.parameters()):,}")


# ================================================================
# 9. 📝 核心速查
# ================================================================
sep("9. 📝 核心速查")

print("""
  写网络三步：
    1. class MyNet(nn.Module):
    2. __init__: 定义层 (self.conv = nn.Conv2d(...))
    3. forward(self, x): 定义数据流

  关键 API：
    nn.Linear(in, out)       → 全连接层
    nn.ReLU() / Sigmoid()    → 激活函数
    nn.Dropout(p)            → 随机丢弃
    nn.BatchNorm1d(features) → 批标准化
    nn.Sequential(*layers)   → 顺序组合，不用写 forward

  模型操作：
    model(x)               → 前向传播（别直接调 .forward）
    model.parameters()     → 所有可训练参数
    model.train() / .eval()→ 切换模式
    model.to(device)       → 移动设备

  nn.Parameter vs Tensor：
    Parameter 自动收集到 parameters()；Tensor 不会

  👉 下一步：去 practice/03_network_building.py 刷题！
""")
