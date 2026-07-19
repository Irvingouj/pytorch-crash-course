"""Build tutorial/07_gpu_save_load_lab.ipynb in the style of 04/05/06."""
import nbformat as nbf


def build(path):
    nb = nbf.v4.new_notebook()
    cells = []

    def md(src):
        cells.append(nbf.v4.new_markdown_cell(src))

    def code(src):
        cells.append(nbf.v4.new_code_cell(src))

    # ================================================================
    # Title
    # ================================================================
    md("""# PyTorch GPU 训练 & 模型保存加载实验课

## 两个独立但都关键的主题

这一课讲两件事，它们让训练从「能跑」变成「能用」：

### ① GPU 训练 — 让模型飞起来
GPU 有几千个小核心并行算矩阵乘法，训练速度比 CPU 快 **10~100 倍**。从 CPU 切到 GPU，PyTorch 里只需要 **3 行代码**。

### ② 保存 / 加载 — 把成果存下来
训练一个模型可能要几小时几天。如果中途崩了，或者想在别处用它，必须能把参数存盘、再读回来。核心是 **`state_dict`**。

```
训练完的模型
     │
     │  torch.save(model.state_dict(), 'model.pth')
     ▼
  model.pth  (磁盘文件，几十 MB ~ 几 GB)
     │
     │  model.load_state_dict(torch.load('model.pth'))
     ▼
另一个进程 / 另一台机器 —— 拿到一模一样的模型
```

配合 `tutorial/07_gpu_save_load.py` 学习。练习在 `practice/07_gpu_checkpoint.py`。""")

    # ================================================================
    # Setup
    # ================================================================
    code("""import torch
import torch.nn as nn
import torch.optim as optim
import os
import tempfile

print(f\"PyTorch version: {torch.__version__}\")
print(f\"CUDA available:  {torch.cuda.is_available()}\")
print(\"用到的模块: torch, torch.nn, torch.optim, os, tempfile\")""")

    # ================================================================
    # Part 1 header
    # ================================================================
    md("""---
# Part 1 · GPU 训练""")

    # ================================================================
    # Section 1: device detection
    # ================================================================
    md("""## 1. 检测 GPU — 一行搞定

PyTorch 用一个 `torch.device` 对象代表计算设备。最常用的写法：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

有 NVIDIA GPU → `cuda`；没有 → `cpu`。**代码完全一样**，只是算得快慢不同。

> ⚠️ **MPS（Apple Silicon）**：M1/M2/M3 Mac 的 GPU 用 `torch.device("mps")`。本课为了简单只讲 `cuda`/`cpu`，Mac 用户可以把 `cuda` 换成 `mps`。""")

    code("""# --- 检测设备 ---
print(\"=\" * 60)
print(\"GPU 可用性\")
print(\"=\" * 60)

print(f\"torch.cuda.is_available() = {torch.cuda.is_available()}\")

if torch.cuda.is_available():
    print(f\"GPU 数量:  {torch.cuda.device_count()}\")
    print(f\"GPU 名称:  {torch.cuda.get_device_name(0)}\")
    print(f\"CUDA 版本: {torch.version.cuda}\")
else:
    print(\"(无 CUDA GPU — 以下演示在 CPU 上跑，逻辑一模一样)\")

# 这台机器用的设备
device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
print(f\"\\n→ 本 notebook 用 device = {device}\")""")

    # ================================================================
    # Section 2: to(device)
    # ================================================================
    md("""## 2. `.to(device)` — 一把梭的最佳实践

不管是 tensor 还是 model，搬设备的 API 都是同一个：`.to(device)`。

```
tensor:   x = torch.randn(3, 4).to(device)
model:    model = MyModel().to(device)
```

**唯一的铁律**：参与同一个运算的所有 tensor 和 model，必须在**同一个设备**上。否则报错：

```
RuntimeError: Expected all tensors to be on the same device,
               but found at least two devices, cuda:0 and cpu!
```""")

    code("""# --- .to(device) 的几种用法 ---
print(\"=\" * 60)
print(\".to(device) — 把数据 / 模型搬到设备\")
print(\"=\" * 60)

# ① 创建时直接指定设备（最高效）
x1 = torch.randn(3, 4, device=device)
print(f\"① 创建时指定: x1.device = {x1.device}\")

# ② 创建后搬过去
x2 = torch.randn(3, 4).to(device)
print(f\"② .to(device): x2.device = {x2.device}\")

# ③ 模型也是一样
model = nn.Linear(10, 5).to(device)
print(f\"③ 模型参数在: {next(model.parameters()).device}\")

# ④ 铁律验证：同设备才能运算
try:
    bad = torch.randn(3, device='cpu')   # 在 CPU
    result = model(bad)                  # model 在 device，bad 在 cpu → 报错
except RuntimeError as e:
    print(f\"\\n④ 不同设备运算 → RuntimeError ❌\")
    print(f\"   {str(e)[:80]}...\")

# 正确：数据也搬过去
x_in = torch.randn(3, 10).to(device)
out = model(x_in)
print(f\"\\n⑤ 数据也 .to(device) 后 → 运算成功 ✅ (out.device={out.device})\")""")

    # ================================================================
    # Section 3: training loop on device
    # ================================================================
    md("""## 3. 完整训练循环里的设备处理

GPU 训练的标准模板——注意数据搬移的位置：

```python
for x, y in train_loader:
    x, y = x.to(device), y.to(device)   # ← 每个 batch 搬一次（模型不用重复搬）
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

**为什么模型 `.to(device)` 只做一次，数据却每个 batch 都搬？**
- 模型参数搬过去就一直在 GPU 上，不用动
- 数据每次是新的 batch，必须从 CPU（DataLoader）搬到 GPU""")

    code("""# --- GPU 训练模板 ---
print(\"=\" * 60)
print(\"GPU 训练模板（CPU 上也照跑）\")
print(\"=\" * 60)

torch.manual_seed(0)
X = torch.randn(200, 10)
y = (X[:, 0] + X[:, 1] > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True) \\
    if False else None  # placeholder, real loader below

from torch.utils.data import DataLoader, TensorDataset
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)  # ← 模型搬一次
opt = optim.Adam(model.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss()

print(f\"模型在: {next(model.parameters()).device}\")
print(\"训练 5 个 batch：\")
for i, (x_b, y_b) in enumerate(loader):
    x_b, y_b = x_b.to(device), y_b.to(device)   # ← 数据每 batch 搬一次
    opt.zero_grad()
    loss = crit(model(x_b), y_b)
    loss.backward()
    opt.step()
    if i < 5:
        print(f\"  batch {i}: x.device={x_b.device}, loss={loss.item():.4f}\")
    if i >= 5:
        break
print(\"✅ 数据搬移只在 batch 循环里做，模型搬一次就够\")""")

    # ================================================================
    # Section 4: speed comparison
    # ================================================================
    md("""## 4. GPU vs CPU 速度对比（概念）

GPU 的优势在**大矩阵乘法**——核心多、带宽高。但在**小模型 + 小数据**上，CPU 反而可能更快（因为数据 CPU→GPU 来回搬的开销大于计算收益）。

| 场景 | 推荐 |
|------|------|
| 小模型 / 调试 | CPU 就行 |
| 大矩阵乘法（Transformer、CNN） | GPU 快 10~100 倍 |
| 巨大 batch | GPU 几乎必须 |

> 下面这个演示在无 GPU 的机器上会跑 CPU vs CPU（看不出差异）。有 GPU 时把 `device='cuda'` 打开就能看到真实加速。""")

    code("""# --- 矩阵乘法计时（演示概念）---
print(\"=\" * 60)
print(\"大矩阵乘法 — CPU 计时\")
print(\"=\" * 60)
import time

def time_matmul(n, device, repeats=5):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    # 预热
    for _ in range(2):
        _ = a @ b
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    if device.type == 'cuda':
        torch.cuda.synchronize()   # GPU 异步，必须同步才能准确计时
    return (time.perf_counter() - t0) / repeats

for n in [512, 1024, 2048]:
    t_cpu = time_matmul(n, torch.device('cpu'))
    line = f\"  {n}×{n} 矩阵乘: CPU {t_cpu*1000:7.1f} ms\"
    if torch.cuda.is_available():
        t_gpu = time_matmul(n, device)
        line += f\" | GPU {t_gpu*1000:7.1f} ms | 加速 {t_cpu/t_gpu:.1f}x\"
    print(line)

print(\"\\n→ 矩阵越大，GPU 优势越明显（小矩阵反而被数据搬运开销抵消）\")""")

    # ================================================================
    # Part 2 header
    # ================================================================
    md("""---
# Part 2 · 模型保存与加载""")

    # ================================================================
    # Section 5: state_dict
    # ================================================================
    md("""## 5. `state_dict` — 模型的「灵魂」

一个 `nn.Module` 有两层东西：

| 层 | 是什么 | 例子 |
|----|--------|------|
| **结构** | 网络长什么样（哪些层、怎么连） | `nn.Linear(10, 5)` |
| **参数** | 每层具体的权重数值 | `weight: [5×10 matrix]`, `bias: [5 vector]` |

`state_dict` 就是**参数那一层**——一个 `OrderedDict`，key 是层名，value 是参数 tensor。

```python
model.state_dict()
# OrderedDict([
#   ('0.weight', tensor(...)),
#   ('0.bias',   tensor(...)),
#   ('2.weight', tensor(...)),
#   ('2.bias',   tensor(...)),
# ])
```

**为什么存 `state_dict` 而不是整个 model？** 见第 8 节。""")

    code("""# --- state_dict 长啥样 ---
print(\"=\" * 60)
print(\"state_dict — 模型的所有参数\")
print(\"=\" * 60)

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

sd = model.state_dict()
print(f\"state_dict 类型: {type(sd).__name__}\")
print(f\"包含 {len(sd)} 个参数张量:\\n\")
for key, param in sd.items():
    print(f\"  {key:<20s} shape={list(param.shape)}  dtype={param.dtype}\")

# 注意：ReLU 没有参数，所以不在 state_dict 里
print(\"\\n💡 ReLU 没参数，所以不在 state_dict 里\")
print(f\"   总参数量: {sum(p.numel() for p in sd.values())}\")""")

    # ================================================================
    # Section 6: save state_dict
    # ================================================================
    md("""## 6. 保存 — 一行 `torch.save`

```python
torch.save(model.state_dict(), 'model.pth')
```

- `.pth` / `.pt` 是约定俗成的扩展名（其实是 pickle 文件）
- **只存参数，不存结构**。加载时必须先有一个结构完全相同的空模型
- 文件大小 ≈ 参数量 × 4 bytes（float32）""")

    code("""# --- 保存 state_dict ---
print(\"=\" * 60)
print(\"保存 — torch.save(state_dict, path)\")
print(\"=\" * 60)

# 用临时目录，不污染仓库
save_dir = tempfile.mkdtemp()
save_path = os.path.join(save_dir, 'demo_model.pth')

torch.save(model.state_dict(), save_path)
file_size = os.path.getsize(save_path)
n_params = sum(p.numel() for p in model.parameters())

print(f\"保存到: {save_path}\")
print(f\"文件大小: {file_size} bytes  (参数 {n_params} × 4 = {n_params*4} bytes 预期)\")""")

    # ================================================================
    # Section 7: load state_dict
    # ================================================================
    md("""## 7. 加载 — 两步走：先建壳，再填参数

```python
# ① 先建一个结构完全相同的空模型
model2 = MyModel()    # 这就是「壳」

# ② 把参数从磁盘灌进去
model2.load_state_dict(torch.load('model.pth', weights_only=True))
```

⚠️ **PyTorch 2.6+ 加载要用 `weights_only=True`**（安全考虑，防止恶意 pickle）。旧代码里看到不带这个参数的，是历史遗留。

`map_location` 参数用来跨设备加载：GPU 上存的想加载到 CPU，就 `map_location='cpu'`。""")

    code("""# --- 加载 state_dict ---
print(\"=\" * 60)
print(\"加载 — load_state_dict\")
print(\"=\" * 60)

# ① 先建一个结构一样的壳
model2 = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

# ② 灌参数
state_dict = torch.load(save_path, weights_only=True)
model2.load_state_dict(state_dict)
print(\"✓ 加载成功\")

# ③ 验证：两个模型输出一致
x_test = torch.randn(3, 10)
with torch.no_grad():
    out1 = model(x_test)
    out2 = model2(x_test)
print(f\"\\n原模型 vs 加载模型 输出一致: {torch.allclose(out1, out2)} ✅\")

# ④ 跨设备加载：map_location
sd_cpu = torch.load(save_path, map_location='cpu', weights_only=True)
print(f\"\\nmap_location='cpu' 加载: {type(sd_cpu).__name__} with {len(sd_cpu)} keys\")

# ⑤ 结构不匹配会报错
print(\"\\n结构不匹配的报错演示:\")
wrong_model = nn.Sequential(nn.Linear(10, 5))   # 少了一层
try:
    wrong_model.load_state_dict(state_dict)
except RuntimeError as e:
    print(f\"  RuntimeError ❌\")
    print(f\"  {str(e)[:120]}...\")
    print(\"  → 加载前必须保证模型结构和保存时完全一致\")""")

    # ================================================================
    # Section 8: state_dict vs whole model
    # ================================================================
    md("""## 8. 为什么存 `state_dict` 而不是整个 model？

PyTorch 有两种保存方式：

| 方式 | 代码 | 优点 | 缺点 |
|------|------|------|------|
| **存 state_dict**（推荐）| `torch.save(model.state_dict(), path)` | 灵活、跨版本安全、能改结构 | 加载时要先建模型 |
| 存整个 model | `torch.save(model, path)` | 加载一行 `torch.load(path)` | **依赖原始类定义**，换环境易崩 |

存整个 model 用的是 Python pickle，它会序列化**类的完整路径**。你在项目 A 里定义了 `class MyModel`，把整个 model 存了，拿到项目 B（没有 `MyModel` 类）就加载不出来。**生产环境永远用 state_dict**。""")

    code("""# --- 存整个 model vs 存 state_dict ---
print(\"=\" * 60)
print(\"存整个 model（不推荐）vs 存 state_dict（推荐）\")
print(\"=\" * 60)

# 方式 A：存整个 model
full_path = os.path.join(save_dir, 'full_model.pth')
torch.save(model, full_path)
print(f\"存整个 model: {os.path.getsize(full_path)} bytes\")

model_full = torch.load(full_path, weights_only=False)   # 整个 model 要 weights_only=False
print(f\"加载: {type(model_full).__name__}, 直接能用\")

# 方式 B：存 state_dict（推荐）
sd_path = os.path.join(save_dir, 'model_state.pth')
torch.save(model.state_dict(), sd_path)
print(f\"\\n存 state_dict: {os.path.getsize(sd_path)} bytes\")

print(\"\\n💡 对比:\")
print(\"   存整个 model:   方便，但换环境（没类定义）就崩 ❌\")
print(\"   存 state_dict:  要先建壳，但灵活、跨版本安全 ✅  ← 生产用这个\")""")

    # ================================================================
    # Section 9: checkpoint
    # ================================================================
    md("""## 9. 完整 Checkpoint — 训练中断恢复

真实训练动辄几十小时。如果跑到一半断了（断电、OOM、手动停），从头再来太亏。**Checkpoint** 就是把训练的全部状态存盘，断了能接着练。

一个完整的 checkpoint 不仅仅是模型参数，还包括：

```python
checkpoint = {
    'epoch': 42,                              # 训练到第几轮
    'model_state_dict': model.state_dict(),   # 模型参数
    'optimizer_state_dict': optimizer.state_dict(),   # 优化器动量等状态
    'scheduler_state_dict': scheduler.state_dict(),   # lr 调度器状态
    'best_val_acc': 0.873,                    # 最佳指标
    # 还可以加：超参数、随机种子、数据集版本...
}
```

**为什么连 optimizer 也要存？** 因为 Adam 这种优化器内部有**动量**（历史梯度的指数平均）。如果不存，恢复训练后动量归零，前几步会乱跳。""")

    code("""# --- 完整 checkpoint 保存 + 恢复 ---
print(\"=\" * 60)
print(\"Checkpoint — 保存训练全状态 + 恢复\")
print(\"=\" * 60)

model = nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 模拟训练到第 42 个 epoch
current_epoch = 42
best_val_acc = 0.873

# ① 保存 checkpoint
checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_acc,
}
ckpt_path = os.path.join(save_dir, 'checkpoint.pth')
torch.save(checkpoint, ckpt_path)

print(\"保存的 checkpoint 内容:\")
for k, v in checkpoint.items():
    if isinstance(v, dict):
        print(f\"  {k:<25s} dict with {len(v)} keys\")
    else:
        print(f\"  {k:<25s} {v}\")

# ② 恢复训练 —— 先新建各组件（结构要一致）
print(\"\\n恢复训练:\")
model_r = nn.Linear(10, 5)
opt_r = optim.Adam(model_r.parameters(), lr=0.001)
sch_r = optim.lr_scheduler.StepLR(opt_r, step_size=10, gamma=0.5)

ckpt = torch.load(ckpt_path, weights_only=True)
model_r.load_state_dict(ckpt['model_state_dict'])
opt_r.load_state_dict(ckpt['optimizer_state_dict'])
sch_r.load_state_dict(ckpt['scheduler_state_dict'])
resume_epoch = ckpt['epoch']
best_acc = ckpt['best_val_acc']

print(f\"  ✓ 从 Epoch {resume_epoch + 1} 接着练\")
print(f\"  ✓ 历史最佳 acc = {best_acc}\")
print(f\"  ✓ 模型 + 优化器 + 调度器状态全部恢复\")""")

    # ================================================================
    # Section 10: best practice checkpoint during training
    # ================================================================
    md("""## 10. 训练中保存 Checkpoint 的标准模式

实际训练循环里，通常**每次验证 loss 创新低就保存一次**（best checkpoint），再定期保存一次（latest checkpoint 防中断）。

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train(...)
    val_loss = evaluate(...)

    # 保存 best（验证创新低）
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, 'best.pth')

    # 定期存 latest（防中断，每个 epoch 或每 N 个 epoch）
    torch.save({...}, 'latest.pth')
```""")

    code("""# --- 训练循环里的 checkpoint 模式 ---
print(\"=\" * 60)
print(\"训练循环里的 checkpoint 模式\")
print(\"=\" * 60)

# 模拟一个简短的训练 + best checkpoint
torch.manual_seed(0)
X = torch.randn(100, 5); y = (X[:, 0] > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
opt = optim.Adam(model.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss()

best_path = os.path.join(save_dir, 'best.pth')
best_val_loss = float('inf')

for epoch in range(10):
    # train
    model.train()
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        opt.zero_grad(); crit(model(x_b), y_b).backward(); opt.step()
    # eval
    model.eval()
    with torch.no_grad():
        val_loss = crit(model(X.to(device)), y.to(device)).item()
    # save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_loss': val_loss}, best_path)
        marker = ' ← 保存 best'
    else:
        marker = ''
    print(f\"  Epoch {epoch}: val_loss={val_loss:.4f}{marker}\")

print(f\"\\n✓ best checkpoint 在 epoch 创新低时保存，best_val_loss={best_val_loss:.4f}\")
print(\"  推理时加载 best.pth 即可\")""")

    # ================================================================
    # Section 11: GPU best practices
    # ================================================================
    md("""## 11. GPU 训练最佳实践清单

| 类别 | 做法 | 作用 |
|------|------|------|
| **数据传输** | `DataLoader(pin_memory=True)` | 锁页内存，加速 CPU→GPU |
| | `x.to(device, non_blocking=True)` | 异步传输，不阻塞 CPU |
| **模型** | `model.to(device)` 只做一次 | 参数一直在 GPU |
| **多 GPU** | `nn.DataParallel(model)` | 单机多卡，简单但有开销 |
| | `nn.DistributedDataParallel` | 多机多卡，性能最优（生产推荐） |
| **混合精度** | `torch.cuda.amp.autocast()` + `GradScaler` | 省显存、加速 ~2x，几乎零改动 |
| **显存管理** | `torch.cuda.empty_cache()` | 释放未用显存碎片 |
| | `del tensor` | 先解除引用再 empty_cache 才有效 |

### 混合精度示例（省显存 + 加速）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with autocast():                       # ← 自动用 float16 算
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()          # ← 缩放防 float16 梯度下溢
    scaler.step(optimizer)
    scaler.update()
```""")

    code("""# --- 验证最佳实践代码能跑（CPU 上 amp 会自动 fallback）---
print(\"=\" * 60)
print(\"GPU 训练最佳实践 — 验证代码可运行\")
print(\"=\" * 60)

device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
print(f\"device = {device}\")

# 演示完整的 GPU-ready 训练片段（CPU 上也能跑）
model = nn.Linear(10, 2).to(device)
opt = optim.Adam(model.parameters(), lr=0.01)

# pin_memory 在 CPU 上无副作用，non_blocking 同理
dummy_loader = DataLoader(
    TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))),
    batch_size=8, pin_memory=True,
)

model.train()
for x_b, y_b in dummy_loader:
    x_b = x_b.to(device, non_blocking=True)   # ← 最佳实践
    y_b = y_b.to(device, non_blocking=True)
    opt.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x_b), y_b)
    loss.backward()
    opt.step()
print(\"✅ pin_memory + non_blocking + .to(device) 模板跑通\")
print(\"   (有 GPU 时这套写法能拿到最大吞吐)\")

# 清理临时文件
import shutil
shutil.rmtree(save_dir)
print(f\"\\n🧹 已清理临时目录 {save_dir}\")""")

    # ================================================================
    # Cheat sheet
    # ================================================================
    md("""## 📝 核心速查

### 设备管理

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)           # 模型搬一次
for x, y in loader:
    x, y = x.to(device), y.to(device)  # 数据每 batch 搬
```

### 保存 / 加载（推荐方式）

```python
# 保存
torch.save(model.state_dict(), "model.pth")

# 加载
model = MyModel()                              # 先建结构相同的壳
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

### 完整 Checkpoint（训练恢复）

```python
# 保存
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
}, "checkpoint.pth")

# 恢复
ckpt = torch.load("checkpoint.pth", weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
```

### 跨设备加载

```python
# GPU 上存的，加载到 CPU
state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)
```

### 常见误区

| ❌ 错误 | ✅ 正确 |
|---------|---------|
| 存整个 model `torch.save(model, ...)` | 存 state_dict `torch.save(model.state_dict(), ...)` |
| 加载不带 `weights_only=True`（PyTorch 2.6+） | `torch.load(..., weights_only=True)` |
| model 和 data 在不同 device | 都 `.to(device)` 到同一个 |
| 验证 loss 创新低不保存 | 存 best checkpoint，推理用它 |
| 只存 model 不存 optimizer | checkpoint 要含 optimizer（Adam 有动量） |
| batch_size 太大 OOM | 降 batch_size 或用混合精度 |

### GPU 加速关键四件套

```python
loader = DataLoader(ds, batch_size=32, pin_memory=True)   # ① pin_memory
model = model.to(device)                                   # ② 模型搬 GPU
x = x.to(device, non_blocking=True)                        # ③ 数据异步搬
# ④ 混合精度（大模型）: torch.cuda.amp.autocast + GradScaler
```

👉 下一步：去 `practice/07_gpu_checkpoint.py` 刷题！

🎉 **学完所有 7 课，你已经掌握了 PyTorch 的核心！**""")

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "pytorch-crash-course (3.12.11)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.11",
        },
    }
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5

    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {path} with {len(cells)} cells")


build("tutorial/07_gpu_save_load_lab.ipynb")
