"""
╔══════════════════════════════════════════════════════════════╗
║      PyTorch 速成课 · 第4课：损失函数 & 优化器               ║
║      「怎么衡量模型好坏」&「怎么让模型变好」                   ║
╚══════════════════════════════════════════════════════════════╝

📌 核心概念：
  损失函数（Loss Function）：一个数值，衡量「预测」和「真实标签」之间的差距。
  优化器（Optimizer）：根据损失算出来的梯度，更新模型参数让损失越来越小。

  训练的本质：
    1. 前向传播 → 得到预测值
    2. 计算 loss = loss_fn(预测值, 真实值)
    3. loss.backward() → 每个参数 .grad 自动填充
    4. optimizer.step() → 用 .grad 更新参数

  运行：python tutorial/04_loss_optimizer.py
  练习：practice/04_loss_optim.py
"""

import torch
import torch.nn as nn
import torch.optim as optim


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ================================================================
# 1. 损失函数 — 衡量预测和真实值的差距
# ================================================================
sep("1. 损失函数一览")

# --- MSELoss：回归任务，预测连续值 ---
mse = nn.MSELoss()
pred = torch.tensor([2.5, 0.0, 2.1])
target = torch.tensor([3.0, -0.5, 2.0])
loss_mse = mse(pred, target)
print(f"MSELoss:   预测={pred.tolist()}")
print(f"           真实={target.tolist()}")
print(f"           损失={loss_mse:.4f}   # (2.5-3)² + (0-(-0.5))² + (2.1-2)² 再除以3")

# --- CrossEntropyLoss：分类任务 ---
# ⚠️ 输入是 raw logits（未经 softmax!），target 是类别索引（不是 one-hot!）
ce = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 1.0, 0.1],    # batch=2, classes=3
                       [0.5, 2.0, 0.3]])
labels = torch.tensor([0, 1])              # 第一个样本是第0类，第二个是第1类
loss_ce = ce(logits, labels)
print(f"\nCrossEntropyLoss:")
print(f"  logits (2个样本, 3类):\n{logits}")
print(f"  labels: {labels}")
print(f"  损失 = {loss_ce:.4f}")
print(f"  内部: Softmax + NLLLoss，所以不要在外面再加 Softmax！")

# --- BCELoss / BCEWithLogitsLoss：二分类 ---
# 推荐 BCEWithLogitsLoss（数值更稳定，自带 Sigmoid）
bce = nn.BCEWithLogitsLoss()
pred_binary = torch.tensor([1.5, -2.0, 0.5])   # logits
target_binary = torch.tensor([1.0, 0.0, 1.0])  # 0 或 1
loss_bce = bce(pred_binary, target_binary)
print(f"\nBCEWithLogitsLoss: {loss_bce:.4f}  (内部自动 Sigmoid)")


# ================================================================
# 2. 优化器 — 用梯度更新参数
# ================================================================
sep("2. 优化器 — SGD 和 Adam")

# 先造一个简单模型和一个假的优化问题
model = nn.Linear(2, 1)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y_true = torch.tensor([[5.0], [11.0]])   # 目标：y = 2*x1 + 1.5*x2 左右

# --- SGD (随机梯度下降) ---
# lr 是最重要的超参数！
sgd = optim.SGD(model.parameters(), lr=0.01)
print(f"SGD: 最经典的优化器，lr=0.01")
print(f"  更新规则: w = w - lr * grad")

# 看一次更新
pred = model(x)
loss = nn.MSELoss()(pred, y_true)
loss.backward()

print(f"  更新前 weight: {model.weight.data}")
sgd.step()
print(f"  更新后 weight: {model.weight.data}")

# 清零梯度（必须！否则下次会累加）
sgd.zero_grad()

# --- Adam (自适应矩估计) ---
# 大多数情况下的首选，自动调整学习率
model2 = nn.Linear(2, 1)
adam = optim.Adam(model2.parameters(), lr=0.01)
print(f"\nAdam: 自适应学习率，大多数任务的首选")
print(f"  自动为每个参数调整学习率")
print(f"  对稀疏梯度效果好，收敛快")

pred = model2(x)
loss = nn.MSELoss()(pred, y_true)
loss.backward()
adam.step()
adam.zero_grad()


# ================================================================
# 3. 完整的训练步骤 — 四次操作为一个循环
# ================================================================
sep("3. 一个训练 step 的完整流程")

model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

x = torch.tensor([[1.0, 2.0]])
y_true = torch.tensor([[5.0]])

print("一个训练 step 的四步走：\n")

# ① 清零梯度
optimizer.zero_grad()
print(f"① optimizer.zero_grad() — 清零上次的梯度")

# ② 前向传播
pred = model(x)
print(f"② pred = model(x)        — 前向传播，pred={pred.data.tolist()}")

# ③ 计算损失
loss = criterion(pred, y_true)
print(f"③ loss = criterion(...)  — 计算损失, loss={loss.item():.4f}")

# ④ 反向传播 + 参数更新
loss.backward()
optimizer.step()
print(f"④ loss.backward() + optimizer.step() — 反传+更新")

print(f"\n循环此四步直到收敛，就是训练！")


# ================================================================
# 4. 学习率调度器 — 训练过程中动态调整 lr
# ================================================================
sep("4. 学习率调度器 lr_scheduler")

model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# StepLR：每 step_size 个 epoch，lr 乘以 gamma
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

print("StepLR(step_size=2, gamma=0.5): 每2步学习率减半")
for epoch in range(1, 6):
    print(f"  Epoch {epoch}: lr = {scheduler.get_last_lr()}")
    scheduler.step()
    # 实际训练中这里会是：train_one_epoch() → scheduler.step()

# 其他常用调度器：
print("""
  其他常用调度器：
    ReduceLROnPlateau  → 验证loss不降时自动降lr（最常用！）
    CosineAnnealingLR  → 余弦退火（训练末期常用）
    OneCycleLR         → 先升后降（训练大模型常用）
""")


# ================================================================
# 5. 优化器的高级用法 — 不同参数不同学习率
# ================================================================
sep("5. 不同参数不同 lr — 分组优化")

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(10, 20)
        self.classifier = nn.Linear(20, 2)

    def forward(self, x):
        return self.classifier(torch.relu(self.features(x)))

model = MyModel()

# 不同层用不同学习率
optimizer = optim.SGD([
    {'params': model.features.parameters(), 'lr': 0.001},
    {'params': model.classifier.parameters(), 'lr': 0.01},
], lr=0.01)  # 不指定组的默认 lr

print("参数分组：")
for i, group in enumerate(optimizer.param_groups):
    print(f"  组 {i}: lr={group['lr']}, params={len(group['params'])} 个")

# 典型场景：finetune 时 backbone 用小 lr，分类头用大 lr


# ================================================================
# 6. 常见 Loss 选型指南
# ================================================================
sep("6. Loss 选型指南")

print("""
  任务类型              推荐 Loss             输入要求
  ─────────────────────────────────────────────────────
  回归（预测房价）       nn.MSELoss()           pred 和 target 同形状
  回归（有离群值）       nn.L1Loss() / SmoothL1Loss
  二分类                 nn.BCEWithLogitsLoss() pred 是 logits, target 是 0/1
  多分类（互斥）         nn.CrossEntropyLoss()  pred 是 logits, target 是类别索引
  多标签分类             nn.BCEWithLogitsLoss() pred 是 logits, target 是 0/1 向量
  对比学习               nn.TripletMarginLoss()
  GAN / 生成             nn.BCELoss() (需手动加 Sigmoid)
""")


# ================================================================
# 7. 📝 核心速查
# ================================================================
sep("7. 📝 核心速查")

print("""
  训练四步：
    optimizer.zero_grad()     # ① 清零梯度
    loss = criterion(y_pred, y_true)  # ② + ③ 前向+算loss
    loss.backward()           # ④ 反向传播
    optimizer.step()          # ⑤ 更新参数

  Loss:
    nn.MSELoss()              → 回归
    nn.CrossEntropyLoss()     → 多分类（输入logits！）
    nn.BCEWithLogitsLoss()    → 二分类（自带Sigmoid，数值稳定）

  优化器:
    optim.SGD(params, lr)     → 经典
    optim.Adam(params, lr)    → 首选，自适应

  调度器:
    lr_scheduler.StepLR       → 固定步数衰减
    lr_scheduler.ReduceLROnPlateau → 监控指标，不降就减lr

  👉 下一步：去 practice/04_loss_optim.py 刷题！
""")
