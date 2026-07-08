"""
╔══════════════════════════════════════════════════════════════╗
║       PyTorch 速成课 · 第7课：GPU 训练 & 模型保存加载       ║
║       「让模型飞起来 & 把辛苦训练的成果存下来」               ║
╚══════════════════════════════════════════════════════════════╝

📌 核心概念：
  GPU 训练：把模型和数据搬到 GPU 上，运算速度提升 10-100 倍。
  保存/加载：state_dict 是模型的「灵魂」— 所有参数的字典。
             保存 state_dict（推荐）比保存整个模型更灵活。

  运行：python tutorial/07_gpu_save_load.py
  练习：practice/07_gpu_checkpoint.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import tempfile


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ================================================================
# 1. 检测 GPU 可用性
# ================================================================
sep("1. GPU 是否可用？")

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用:    {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 数量:     {torch.cuda.device_count()}")
    print(f"当前 GPU:     {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本:    {torch.version.cuda}")
else:
    print("(无 GPU — 以下演示在 CPU 上也完全能跑，逻辑一模一样)")


# ================================================================
# 2. 设备管理 — .to(device) 一把梭
# ================================================================
sep("2. 设备管理 — .to(device) 的最佳实践")

# 最佳实践：定义一个 device 变量，到处用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# 创建张量时直接指定设备
x = torch.randn(3, 4, device=device)
print(f"tensor 在: {x.device}")

# 或者创建后搬过去
x = torch.randn(3, 4).to(device)
print(f".to(device) 后: {x.device}")

# 模型也一样
model = nn.Linear(10, 5).to(device)
print(f"模型参数在: {next(model.parameters()).device}")

# ⚠️ 常见错误：tensor 和 model 在不同设备上不能一起运算！
# 正确做法：把数据也搬到同一个 device


# ================================================================
# 3. 保存模型 — state_dict（推荐方式）
# ================================================================
sep("3. 保存模型 — state_dict 是业界标准")

# 创建一个训练好的"模型"（简单演示）
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

# state_dict 是一个 OrderedDict，存了所有参数
print("state_dict 的 keys:")
for key in model.state_dict().keys():
    param = model.state_dict()[key]
    print(f"  {key:30s}  shape={list(param.shape)}")

# 保存：torch.save(state_dict, path)
save_path = "/tmp/demo_model.pth"
torch.save(model.state_dict(), save_path)
print(f"\n✓ 模型已保存到: {save_path}")


# ================================================================
# 4. 加载模型 — 两步走
# ================================================================
sep("4. 加载模型 — 先建壳，再填参数")

# 方式 A：加载到同结构模型（最常用）
model2 = nn.Sequential(   # ← 先建一个结构完全一样的壳
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)
model2.load_state_dict(torch.load(save_path, weights_only=True))
print("✓ 方式A: 同结构模型加载成功")

# 方式 B：加载到不同设备
model_cpu = nn.Sequential(
    nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5),
)
# map_location：把 GPU 上保存的模型加载到 CPU
state_dict = torch.load(save_path, map_location="cpu", weights_only=True)
model_cpu.load_state_dict(state_dict)
print("✓ 方式B: map_location='cpu' 加载成功")

# 验证加载是否一致
x = torch.randn(3, 10)
with torch.no_grad():
    out1 = model(x)
    out2 = model2(x)
print(f"  验证: 加载前后输出一致 = {torch.allclose(out1, out2)} ✓")


# ================================================================
# 5. 保存完整 checkpoint（训练中断恢复）
# ================================================================
sep("5. 保存完整 Checkpoint — 训练中断恢复")

model = nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 模拟训练到第 42 个 epoch
current_epoch = 42
best_val_acc = 0.873

checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_acc,
    # 你还可以加超参数、随机种子等
}

ckpt_path = "/tmp/checkpoint.pth"
torch.save(checkpoint, ckpt_path)
print("Checkpoint 包含:")
for k, v in checkpoint.items():
    if isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
    else:
        print(f"  {k}: {v}")
print(f"\n✓ Checkpoint 保存到: {ckpt_path}")

# 恢复训练
# 先新建各部分（结构要一致）
model2 = nn.Linear(10, 5)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.5)

ckpt = torch.load(ckpt_path, weights_only=True)
model2.load_state_dict(ckpt['model_state_dict'])
optimizer2.load_state_dict(ckpt['optimizer_state_dict'])
scheduler2.load_state_dict(ckpt['scheduler_state_dict'])
resume_epoch = ckpt['epoch']
best_val_acc = ckpt['best_val_acc']

print(f"✓ 从 Epoch {resume_epoch+1} 恢复训练，历史最佳 acc = {best_val_acc}")


# ================================================================
# 6. 保存整个模型（不推荐，但很方便）
# ================================================================
sep("6. 保存整个模型 — 方便但不推荐")

model = nn.Linear(10, 5)

# 保存整个模型对象
torch.save(model, "/tmp/full_model.pth")
print("✓ 整个模型保存到 /tmp/full_model.pth")

# 加载整个模型
model_loaded = torch.load("/tmp/full_model.pth", weights_only=False)
print("✓ 整个模型加载成功（但依赖原始代码结构，换环境可能出错）")
print("  ⚠️ 不推荐：跨项目/跨版本时容易出问题")


# ================================================================
# 7. GPU 训练最佳实践
# ================================================================
sep("7. GPU 训练最佳实践清单")

print("""
  ✅ 数据准备阶段：
     1. pin_memory=True  → DataLoader 加上，加速 CPU→GPU
     2. non_blocking=True → .to(device, non_blocking=True) 异步传输

  ✅ 训练循环：
     3. 统一 device 变量 → device = torch.device("cuda" if cuda else "cpu")
     4. model.to(device)   → 模型搬到 GPU（只需一次）
     5. x, y = x.to(device), y.to(device)  → 每次 batch 搬数据

  ✅ 多 GPU：
     6. nn.DataParallel(model)  → 单机多卡，最简单（有 overhead）
     7. nn.DistributedDataParallel → 多机多卡，性能最优（推荐生产用）

  ✅ 混合精度训练（省显存、加速）：
     8. torch.cuda.amp → 自动混合精度，几乎零代码改动
        with torch.cuda.amp.autocast():
            output = model(x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)

  ✅ 显存管理：
     9. torch.cuda.empty_cache() → 释放未使用的显存碎片
     10. del tensor; torch.cuda.empty_cache() → 确保持有者的引用也释放

  ✅ 常见坑：
     - model 和 data 不在同一设备 → RuntimeError
     - 忘记 .to(device)  → 默认 CPU，训练慢
     - batch_size 太大 → CUDA out of memory
     - 跨 GPU 加载 → map_location 解决
""")


# ================================================================
# 8. 📝 核心速查
# ================================================================
sep("8. 📝 核心速查")

print("""
  设备管理：
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor.to(device)

  保存（推荐）：
    torch.save(model.state_dict(), "model.pth")

  加载：
    model = YourModel()
    model.load_state_dict(torch.load("model.pth", weights_only=True))

  保存/恢复 checkpoint：
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'checkpoint.pth')

    ckpt = torch.load('checkpoint.pth')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

  GPU 加速关键：
    ① DataLoader(pin_memory=True)
    ② model.to(device)
    ③ x, y = x.to(device), y.to(device)
    ④ torch.cuda.amp 混合精度

  👉 下一步：去 practice/07_gpu_checkpoint.py 刷题！
  🎉 学完所有7课，你已经掌握了 PyTorch 的核心！
""")

# 清理临时文件
os.remove("/tmp/demo_model.pth")
os.remove("/tmp/checkpoint.pth")
os.remove("/tmp/full_model.pth")
