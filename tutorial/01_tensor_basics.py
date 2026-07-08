"""
╔══════════════════════════════════════════════════════════════╗
║           PyTorch 速成课 · 第1课：Tensor 基础               ║
║           张量 — PyTorch 世界的「一等公民」                   ║
╚══════════════════════════════════════════════════════════════╝

📌 核心概念：
  Tensor（张量）就是多维数组。标量是 0 维，向量是 1 维，矩阵是 2 维...
  PyTorch 的 Tensor 和 NumPy 的 ndarray 几乎一一对应，
  但 Tensor 多了两个杀手锏：GPU 加速 + 自动求导。

  先跑一遍本文件：
      python tutorial/01_tensor_basics.py

  学完之后去 practice/01_tensor_ops.py 刷题巩固。
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
#  helper: print code + result together, REPL-style
# ---------------------------------------------------------------------------
def show(code: str, result):
    """打印一行代码及其结果，模拟 Python REPL 的视觉效果。"""
    lines = code.split('\n')
    for line in lines:
        print(f"  >>> {line}")
    print(f"  {result}")
    print()


def section(title: str):
    print(f"\n{'─' * 54}")
    print(f"  {title}")
    print(f"{'─' * 54}\n")


# ================================================================
# 1. 创建 Tensor — 多种「工厂函数」
# ================================================================
section("1. 创建 Tensor — 工厂函数一览")

# 从 Python 列表
show("torch.tensor([1, 2, 3])",
     torch.tensor([1, 2, 3]))

# 从 NumPy（零拷贝，共享内存）
show("torch.from_numpy(np.array([4., 5., 6.]))",
     torch.from_numpy(np.array([4., 5., 6.])))

# 全零 / 全一
show("torch.zeros(2, 3)",
     torch.zeros(2, 3))
show("torch.ones(2, 3)",
     torch.ones(2, 3))

# 随机
show("torch.rand(2, 3)   # 均匀分布 [0,1)",
     torch.rand(2, 3))
show("torch.randn(2, 3)  # 标准正态 N(0,1)",
     torch.randn(2, 3))

# 等差数列
show("torch.arange(0, 10, 2)",
     torch.arange(0, 10, 2))

# 等间距
show("torch.linspace(0, 1, 5)",
     torch.linspace(0, 1, 5))

# 单位矩阵
show("torch.eye(3)",
     torch.eye(3))

# 像某个 tensor 一样（继承 shape/dtype/device）
a = torch.ones(2, 3)
show("torch.zeros_like(a)",
     torch.zeros_like(a))

# ═══════════════════════════════════════════════════════════════
#  🆕 torch.stack — 沿新维度堆叠（vs torch.cat 沿已有维度拼接）
# ═══════════════════════════════════════════════════════════════
section("1b. stack vs cat — 堆叠与拼接的区别")

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])

show("torch.stack([x, y])              # 新增第0维, (2,2)",
     torch.stack([x, y]))

show("torch.stack([x, y], dim=1)       # 沿第1维堆叠, (2,2)",
     torch.stack([x, y], dim=1))

show("torch.cat([x, y])                # 沿已有维度拼接, (4,)",
     torch.cat([x, y]))

print("  💡 stack 凭空生出一个维度，cat 只沿已有维度延长")
print("     shape (N,) + stack(dim=0) → (2, N)")
print("     shape (N,) + stack(dim=1) → (N, 2)")
print("     shape (N,) + cat          → (2N,)")


# ================================================================
# 2. Tensor 属性
# ================================================================
section("2. Tensor 属性 — shape / dtype / device")

t = torch.randn(3, 4)
print(f"  t.shape   (尺寸)  = {t.shape}      # torch.Size 即 tuple")
print(f"  t.dtype   (类型)  = {t.dtype}      # 默认 float32")
print(f"  t.device  (设备)  = {t.device}     # 默认 CPU")
print(f"  t.ndim    (几维)  = {t.ndim}")
print(f"  t.numel() (总元素) = {t.numel()}")
print()


# ================================================================
# 3. 索引与切片（和 NumPy 一模一样）
# ================================================================
section("3. 索引 & 切片 — NumPy 语法原样复用")

t = torch.arange(12).reshape(3, 4)
show("t",
     t)
show("t[0, 0]     # 单个元素",
     t[0, 0])
show("t[0]        # 第一行",
     t[0])
show("t[:, 1]     # 第二列",
     t[:, 1])
show("t[:2, 1:]   # 前两行，第2列起",
     t[:2, 1:])
show("t[-1]       # 最后一行",
     t[-1])

# 花式索引：用列表/张量指定位置
show("t[[0, 2]]   # 取第0行和第2行",
     t[[0, 2]])

# 布尔索引
show("t[t > 5]    # 筛选 >5 的元素",
     t[t > 5])

# ═══════════════════════════════════════════════════════════════
#  🆕 二维花式索引 — 提取对角线、特定坐标
# ═══════════════════════════════════════════════════════════════
print("  ── 二维花式索引：两个列表配对提取 ──\n")

show("t",
     t)

show("t[[0, 1, 2], [0, 1, 2]]   # 每对 (row,col) 配对: (0,0)(1,1)(2,2)",
     t[[0, 1, 2], [0, 1, 2]])

show("t[torch.arange(3), torch.arange(3)]   # 同上，arange 更简洁",
     t[torch.arange(3), torch.arange(3)])

print("  💡 这就是不用 torch.diag() 取对角线的方法。")
print("     所有 row 和 col 一一配对：t[row_indices, col_indices]")
print()


# ================================================================
# 4. 变形 — reshape / view / transpose / permute
# ================================================================
section("4. 变形 — 只换「看法」，数据不变")

t = torch.arange(6)
show("t",
     t)

# reshape / view
show("t.reshape(2, 3)",
     t.reshape(2, 3))
show("t.reshape(-1, 2)   # -1 表示「帮我算」",
     t.reshape(-1, 2))

# unsqueeze
show("t.unsqueeze(0)     # 第0维前插一维, shape:",
     t.unsqueeze(0).shape)
show("t.unsqueeze(1)     # 第1维前插一维, shape:",
     t.unsqueeze(1).shape)

# squeeze
x = torch.randn(1, 3, 1, 4)
show("x.squeeze()        # 干掉所有大小为1的维, shape:",
     f"{x.shape} → {x.squeeze().shape}")

# 转置
m = torch.randn(2, 3)
show("m.T                # 仅限2D, shape:",
     f"{m.shape} → {m.T.shape}")

# permute
t3d = torch.randn(2, 3, 4)
show("t3d.permute(2, 0, 1)  # 通用维度重排, shape:",
     f"{t3d.shape} → {t3d.permute(2, 0, 1).shape}")


# ================================================================
# 5. 基本运算 — 逐元素 & 矩阵乘法 & 广播
# ================================================================
section("5a. 逐元素运算 & 矩阵乘法")

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

show("a + b   # 逐元素加",
     a + b)
show("a * b   # 逐元素乘（不是点积！）",
     a * b)
show("a ** 2",
     a ** 2)

# 矩阵乘法（三种写法等价）
A = torch.randn(2, 3)
B = torch.randn(3, 4)
show("(A @ B).shape      # 推荐的通用写法",
     (A @ B).shape)
show("torch.mm(A, B).shape    # 仅限 2D",
     torch.mm(A, B).shape)
show("torch.matmul(A, B).shape  # 通用但啰嗦",
     torch.matmul(A, B).shape)


# ═══════════════════════════════════════════════════════════════
#  🆕 不用 @ 实现矩阵乘法 — 用广播拆解 matmul
# ═══════════════════════════════════════════════════════════════
section("5b. 手动矩阵乘法 — 用广播拆解 matmul")

print("  目标: A (2,3) @ B (3,4) → (2,4)\n")

A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
B = torch.tensor([[1., 2., 3., 4.],
                  [5., 6., 7., 8.],
                  [9., 10., 11., 12.]])

show("A",
     A)
show("B",
     B)

# Step 1: 对齐维度
show("A.unsqueeze(-1).shape   # (2, 3) → (2, 3, 1)",
     A.unsqueeze(-1).shape)
show("B.unsqueeze(0).shape    # (3, 4) → (1, 3, 4)",
     B.unsqueeze(0).shape)

# Step 2: broadcast 乘法
step = A.unsqueeze(-1) * B.unsqueeze(0)  # (2, 3, 4)
show("(A.unsqueeze(-1) * B.unsqueeze(0)).shape  # 广播!, shape:",
     step.shape)

# Step 3: 沿中间维求和
result = step.sum(dim=1)  # (2, 4)
show("step.sum(dim=1).shape  # 沿 dim=1 求和 → (2, 4)",
     result.shape)

show("torch.allclose(result, A @ B)  # 验证与 @ 一致",
     torch.allclose(result, A @ B))

print("  💡 三步公式：unsqueeze → broadcast * → sum(dim=匹配维)")
print("     A(m,n) @ B(n,p) = (A.unsqueeze(-1) * B.unsqueeze(0)).sum(dim=1)")
print()


# ═══════════════════════════════════════════════════════════════
#  广播与聚合
# ═══════════════════════════════════════════════════════════════
section("5c. 广播 (broadcasting) — 不同形状也能运算")

x = torch.randn(3, 1)   # (3, 1)
y = torch.randn(1, 4)   # (1, 4)
show("x.shape, y.shape",
     f"{x.shape}, {y.shape}")
show("(x + y).shape     # (3,1) + (1,4) → broadcast → (3,4)",
     (x + y).shape)

# 行广播 vs 列广播
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
row_vec = torch.tensor([10, 20])              # shape (2,)
col_vec = torch.tensor([10, 20, 30])          # shape (3,)

show("matrix",
     matrix)
show("matrix + row_vec.unsqueeze(1)   # (2,3) + (2,1) → 每一行 + 不同值",
     matrix + row_vec.unsqueeze(1))
show("matrix + col_vec                # (2,3) + (3,)  → 每一列 + [10,20,30]",
     matrix + col_vec)


# ═══════════════════════════════════════════════════════════════
#  🆕 Min-Max 归一化 — .min() + .max() 沿维度
# ═══════════════════════════════════════════════════════════════
section("5d. 聚合 — sum / mean / min / max 沿维度")

t = torch.randn(3, 4)
show("t",
     t)
show("t.sum()",
     t.sum())
show("t.sum(dim=0)    # 沿行求和 → (4,)",
     t.sum(dim=0))
show("t.mean(dim=1)   # 沿列求均值 → (3,)",
     t.mean(dim=1))
show("t.max()",
     t.max())
show("t.argmax(dim=1) # 每行最大值的索引",
     t.argmax(dim=1))

# min / max 沿维度
show("t.min(dim=1)    # 返回 (values, indices)",
     t.min(dim=1))
show("t.max(dim=1)",
     t.max(dim=1))

# 完整的 Min-Max 归一化（逐行）
print("  ── Min-Max 归一化：把每行缩放到 [0, 1] ──\n")
data = torch.tensor([[1., 2., 3.],
                     [5., 5., 5.]])

x_min = data.min(dim=1, keepdim=True).values
x_max = data.max(dim=1, keepdim=True).values

show("data",
     data)
show("x_min = data.min(dim=1, keepdim=True).values",
     x_min)
show("x_max = data.max(dim=1, keepdim=True).values",
     x_max)
show("(data - x_min) / (x_max - x_min)",
     (data - x_min) / (x_max - x_min))

print("  ⚠️  全相同行 → 分母为 0！需要额外处理（比如返回 0.5）")
print()

# ═══════════════════════════════════════════════════════════════
#  🆕 广播创建图案 — 棋盘格等
# ═══════════════════════════════════════════════════════════════
section("5e. 广播创建图案 — 用 arange + unsqueeze 生成坐标网格")

print("  技巧：arange(n).unsqueeze(1) 得到列向量 (n,1)")
print("        arange(n).unsqueeze(0) 得到行向量 (1,n)")
print("        两者相加 → (n,n) 的坐标和矩阵\n")

rows = torch.arange(3).unsqueeze(1)   # shape (3, 1)
cols = torch.arange(3).unsqueeze(0)   # shape (1, 3)

show("rows = torch.arange(3).unsqueeze(1)",
     rows)
show("cols = torch.arange(3).unsqueeze(0)",
     cols)
show("rows + cols   # 每格 = 行号 + 列号",
     rows + cols)
show("(rows + cols) % 2   # 棋盘格！",
     (rows + cols) % 2)

print("  💡 这个技巧是创建棋盘、注意力掩码等结构化张量的基础")
print()


# ================================================================
# 6. 原地操作 & 类型转换
# ================================================================
section("6. 原地操作 & 类型转换")

a = torch.randn(2, 3)

# 带 _ 后缀的是原地操作（in-place），会修改原张量
b = a.clone()  # 备份
b.add_(1)
show("a = torch.randn(2, 3)\nb = a.clone()\nb.add_(1)   # in-place, b 被修改",
     f"a unchanged: {a[0,0]:.4f}   b modified: {b[0,0]:.4f}")

# 类型转换
show("a.float().dtype",
     a.float().dtype)
show("a.long().dtype  # 向0取整，慎用",
     a.long().dtype)
show("a.bool().dtype",
     a.bool().dtype)

# 设备迁移
if torch.cuda.is_available():
    print(f"  a.cuda()  device = {a.cuda().device}")
else:
    print("  (当前环境无 GPU，跳过 .cuda() 演示)")
print()


# ================================================================
# 7. NumPy 互通 — 共享内存，零拷贝
# ================================================================
section("7. NumPy 互通 — 共享内存，零拷贝")

t = torch.ones(3)
n = t.numpy()
show("t = torch.ones(3)\nn = t.numpy()",
     n)

# 修改 tensor 会影响 numpy！
t.add_(1)
show("t.add_(1)  # 修改 tensor 后…\nn",
     n)  # numpy 同步变化

# 反过来
n2 = np.array([2., 4., 6.])
t2 = torch.from_numpy(n2)
show("n2 = np.array([2., 4., 6.])\nt2 = torch.from_numpy(n2)",
     t2)

n2[0] = 99
show("n2[0] = 99  # 修改 numpy 后…\nt2",
     t2)  # tensor 同步变化

print("  💡 .numpy() 和 torch.from_numpy() 共享同一块内存，零拷贝！")
print()


# ================================================================
# 8. einops — rearrange / reduce / repeat 三大金刚
# ================================================================
section("8. einops — 字符串描述张量变换，告别猜谜")

from einops import rearrange, reduce, repeat

# --- 8a. rearrange ---
print("  ── 8a. rearrange — 万能变形 ──\n")

imgs = torch.randn(4, 3, 32, 32)
show("imgs.shape   # (B, C, H, W)",
     list(imgs.shape))

flat = rearrange(imgs, 'b c h w -> b (c h w)')
show("rearrange(imgs, 'b c h w -> b (c h w)').shape  # Flatten 空间",
     list(flat.shape))

patches = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=8, p2=8)
show("rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=8, p2=8).shape",
     list(patches.shape))

x = torch.randn(2, 3, 4, 5)
show("rearrange(x, 'a b c d -> a c b d').shape  # 交换第2维和第3维",
     list(rearrange(x, 'a b c d -> a c b d').shape))

x = torch.randn(3, 4)
show("rearrange(x, 'h w -> h w 1').shape   # 增维",
     list(rearrange(x, 'h w -> h w 1').shape))

x_sqz = torch.randn(3, 1, 4)
show("rearrange(x_sqz, 'h 1 w -> h w').shape  # 减维（前提该维=1）",
     list(rearrange(x_sqz, 'h 1 w -> h w').shape))

# --- 8b. reduce ---
print("\n  ── 8b. reduce — 按维度名聚合 ──\n")

x = torch.randn(2, 3, 4)
show("x.shape",
     list(x.shape))
show("reduce(x, 'b c d -> c d', 'mean').shape  # 沿 b 求均值",
     list(reduce(x, 'b c d -> c d', 'mean').shape))
show("reduce(x, 'b c d -> b d', 'max').shape   # 沿 c 求最大值",
     list(reduce(x, 'b c d -> b d', 'max').shape))
show("reduce(x, 'b c d -> c', 'sum').shape     # 沿 b 和 d 求和",
     list(reduce(x, 'b c d -> c', 'sum').shape))
show("reduce(x, 'b c d -> ', 'min').shape      # 全局最小值 → 标量",
     reduce(x, 'b c d -> ', 'min').shape)

# --- 8c. repeat ---
print("\n  ── 8c. repeat — 沿指定维度复制 ──\n")

x = torch.tensor([1., 2., 3.])
show("x.shape",
     list(x.shape))

out = repeat(x, 'd -> b d', b=2)
show("repeat(x, 'd -> b d', b=2).shape    # (3,) → (2, 3)",
     f"{list(out.shape)}\n  {out}")

out = repeat(x, 'd -> d c', c=4)
show("repeat(x, 'd -> d c', c=4).shape    # (3,) → (3, 4)",
     list(out.shape))

out = repeat(x, 'd -> b d c', b=2, c=3)
show("repeat(x, 'd -> b d c', b=2, c=3).shape  # (3,) → (2, 3, 3)",
     list(out.shape))

# --- 8d. 对照表 ---
print("\n  ── 8d. 经典模式对照表 ──\n")
print("""
    原版 PyTorch                   einops
    ────────────────────────────────────────────────────
    x.reshape(B, -1)               rearrange(x, 'b c h w -> b (c h w)')
    x.permute(0,2,1)               rearrange(x, 'b c d -> b d c')
    x.mean(dim=1)                  reduce(x, 'b d -> b', 'mean')
    x.unsqueeze(0).repeat(3,1,1)   repeat(x, 'h w -> b h w', b=3)

    ① 可读性：字符串比数字索引清晰 100 倍
    ② 防 Bug：维度写错直接报错，不会偷偷跑通
    ③ 框架无关：NumPy / JAX / TensorFlow 也能用
""")


# ================================================================
# 9. 总结速查
# ================================================================
section("9. 📝 核心速查表")

print("""
  创建：tensor / zeros / ones / rand / randn / arange / eye / stack / cat
  属性：.shape / .dtype / .device / .ndim / .numel()
  索引：[row, col]  / [start:end]  / [[rows]]  / [[rows], [cols]]  / [mask]
  变形：reshape(-1,n) / view / squeeze / unsqueeze / permute / T
  运算：+ - * / **  （逐元素）  @ mm matmul  （矩阵乘）
  聚合：sum / mean / min / max / argmax (dim=..., keepdim=...)
  广播：自动对齐，从最后一维往前，维度=1 的自动扩展
  原地：方法名加 _  如 add_()  mul_()
  NumPy：.numpy() / torch.from_numpy()  →  共享内存，零拷贝！
  einops：rearrange / reduce / repeat  →  用字符串描述，清晰防错！

  🎯 下一步：python practice/01_tensor_ops.py
""")
