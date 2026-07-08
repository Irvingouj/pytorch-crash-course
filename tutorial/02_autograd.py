"""
╔══════════════════════════════════════════════════════════════╗
║           PyTorch 速成课 · 第2课：自动求导 Autograd          ║
║           「神经网络训练的心脏」— 梯度是怎样自动算出来的      ║
╚══════════════════════════════════════════════════════════════╝

📌 核心概念：
  训练神经网络 = 反复做一件事：算梯度 → 更新参数。
  PyTorch 的 autograd 帮你自动算梯度 — 你只需要定义好「前向计算」，
  它就能反向传播算出每个参数的梯度。

  关键思路：每个 Tensor 都可以选择记录自己的「来源」，
  形成一个计算图（computation graph）。调用 .backward() 时沿图反向传播。

  运行：python tutorial/02_autograd.py
  练习：practice/02_autograd.py
"""

import torch


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ================================================================
# 1. requires_grad — 开启梯度追踪
# ================================================================
sep("1. requires_grad — 「请帮我记住计算过程」")

# 默认创建的 tensor 不需要梯度
x = torch.tensor([1.0, 2.0, 3.0])
print(f"默认 requires_grad = {x.requires_grad}")

# 显式声明：我需要梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"声明后 requires_grad = {x.requires_grad}")

# 也可以用 .requires_grad_() 原地设置
y = torch.randn(3).requires_grad_(True)
print(f"y.requires_grad = {y.requires_grad}")


# ================================================================
# 2. 前向计算 → backward → 拿到梯度
# ================================================================
sep("2. 前向 → backward → .grad")

x = torch.tensor([2.0, 3.0], requires_grad=True)

# 前向：定义计算
y = x ** 2 + 3 * x + 1   # y = x² + 3x + 1
# 手动求导：dy/dx = 2x + 3, 在 x=[2,3] 处应为 [7, 9]

print(f"x = {x}")
print(f"y = {y}   (y = x² + 3x + 1)")

# 反向传播：注意 y 必须是标量才能直接 .backward()！
# 如果不是标量，需要传一个 grad_tensor（后面会讲）
loss = y.sum()            # 变成标量
loss.backward()

print(f"x.grad = {x.grad}    # 手动验证：dy/dx = 2x+3, 在[2,3]处 = [7,9] ✓")

# ⚠️ 重要：grad 是累积的！再 backward 一次会叠加
# loss.backward()
# print(x.grad)  # 会变成 [14, 18]！
# 所以每次 backward 前通常要清零：optimizer.zero_grad()


# ================================================================
# 3. 计算图 — 链式法则自动执行
# ================================================================
sep("3. 链式法则 — 多层计算自动求导")

x = torch.tensor(1.0, requires_grad=True)

# 一个稍微复杂的计算：y = sin(x² + 1)
u = x ** 2 + 1      # u = x² + 1
y = torch.sin(u)     # y = sin(u)

y.backward()

# 手动链式法则：dy/dx = cos(u) * 2x
manual_grad = torch.cos(u) * 2 * x
print(f"自动求导 x.grad = {x.grad:.6f}")
print(f"手动链式  2x·cos(x²+1) = {manual_grad:.6f}")
print(f"一致: {torch.allclose(x.grad, manual_grad)} ✓")


# ================================================================
# 4. 非标量输出如何 backward
# ================================================================
sep("4. 非标量输出 — 传入 gradient 参数")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2   # y 是向量 [1, 4, 9]，不是标量

# 向量不能直接 backward()，需要告诉 PyTorch 「每个输出的权重」
# 传入一个和 y 同形状的 gradient：
y.backward(gradient=torch.ones_like(y))

# 等价于：先 y.sum() 再 backward()
print(f"y = {y}")
print(f"x.grad = {x.grad}   # dy/dx = 2x = [2, 4, 6] ✓")

# 如果 gradient 不是全 1：
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.backward(gradient=torch.tensor([1.0, 0.0, 0.0]))
print(f"gradient=[1,0,0] -> x.grad = {x.grad}   # 只关心第一个输出")


# ================================================================
# 5. 暂停梯度追踪 — torch.no_grad()
# ================================================================
sep("5. torch.no_grad() — 推理/评估时关闭追踪")

x = torch.tensor(2.0, requires_grad=True)

# 正常计算会建图
y = x ** 2
print(f"正常计算: y.requires_grad = {y.requires_grad}")

# no_grad 上下文里：不建图，省内存，速度快
with torch.no_grad():
    y2 = x ** 2
    print(f"no_grad内: y2.requires_grad = {y2.requires_grad}")
    print(f"         y2 = {y2}  # 值照算，梯度不追踪")

# 典型场景：评估模型时
# model.eval()
# with torch.no_grad():
#     for batch in dataloader:
#         outputs = model(batch)


# ================================================================
# 6. detach() — 从计算图中「摘」出来
# ================================================================
sep("6. detach() — 切断梯度流")

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2          # y 连着 x
z = y.detach()      # z 和 y 值一样，但离开了计算图

print(f"y = {y}, requires_grad = {y.requires_grad}")
print(f"z = {z}, requires_grad = {z.requires_grad}")

# detach 的典型用途：固定某些参数不更新
# 或者：拿中间结果做日志/可视化但不影响梯度


# ================================================================
# 7. 梯度清零 & 多次 backward
# ================================================================
sep("7. 梯度累积 — 每次 backward 前记得清零")

x = torch.tensor(1.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第 {i+1} 次 backward 后 x.grad = {x.grad}")

print(f"\n⚠️ 梯度是累加的！每次 +2.0。训练时记得 optimizer.zero_grad()")

# 正确做法：
x = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"清零后第 {i+1} 次: x.grad = {x.grad}")
    x.grad.zero_()   # ← 清零！


# ================================================================
# 8. 高阶求导 — create_graph=True
# ================================================================
sep("8. 高阶求导 — 保留计算图用于二次求导")

x = torch.tensor(2.0, requires_grad=True)

# 一阶导
y = x ** 3          # y = x³
# create_graph=True：保留计算图，以便对梯度再求导
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"y = x³")
print(f"一阶导 dy/dx = 3x² = {grad1} (在 x=2 处应为 12) ✓")

# 二阶导：对一阶导再求导
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"二阶导 d²y/dx² = 6x = {grad2} (在 x=2 处应为 12) ✓")


# ================================================================
# 9. 📝 核心速查
# ================================================================
sep("9. 📝 核心速查")

print("""
  requires_grad=True    → 开启梯度追踪
  .backward()           → 反向传播，结果存在 .grad 里
  .grad                 → 梯度值（只有叶子节点才有）
  .grad.zero_()         → 清零梯度（每个训练 step 前必做！）
  torch.no_grad():      → 暂停追踪（推理/评估用）
  .detach()             → 从计算图摘除（值不变）
  create_graph=True     → 保留图用于高阶求导

  典型训练片段：
      optimizer.zero_grad()   # 清零
      loss.backward()         # 反传
      optimizer.step()        # 更新参数

  👉 下一步：去 practice/02_autograd.py 刷题！
""")
