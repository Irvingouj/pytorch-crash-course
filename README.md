# 🔥 PyTorch 速成课 — 深入浅出 · LeetCode 风格

PyTorch crash course with a **tutorial + practice** format inspired by LeetCode:  
read the tutorial to understand the API and concepts, then implement the stubs in practice to lock it in.

```
tutorial/          ← Run me first!  (7 lessons, ~10 min each)
practice/          ← Then do me!    (7 problem sets, ~60 LeetCode-style questions)
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖（只需要 uv）
uv sync

# 2. 学一节教程
python tutorial/01_tensor_basics.py

# 3. 刷对应的练习
python practice/01_tensor_ops.py
#    ↑ 全部报错很正常 — 你的任务是把它们全部修成 ✅！
```

---

## 📖 课程大纲

| # | 教程 (tutorial) | 练习 (practice) | 核心内容 |
|---|---|---|---|
| 1 | `01_tensor_basics.py` | `01_tensor_ops.py` | 张量创建、变形、索引、广播、NumPy 互通、**einops** |
| 2 | `02_autograd.py` | `02_autograd.py` | requires_grad、backward、计算图、梯度清零 |
| 3 | `03_nn_module.py` | `03_network_building.py` | nn.Module、Linear、Sequential、Dropout、BatchNorm |
| 4 | `04_loss_optimizer.py` | `04_loss_optim.py` | MSELoss、CrossEntropy、SGD、Adam、lr_scheduler |
| 5 | `05_data_loading.py` | `05_custom_dataset.py` | Dataset、DataLoader、transforms、collate_fn |
| 6 | `06_training_loop.py` | `06_training.py` | 完整训练/验证循环、Early Stopping、Gradient Clipping |
| 7 | `07_gpu_save_load.py` | `07_gpu_checkpoint.py` | GPU 迁移、state_dict、checkpoint 保存恢复 |

---

## 🎯 使用方式

### 学习路径

```
  tutorial/01 → practice/01  ✅ 全部通过？
  tutorial/02 → practice/02  ✅ 全部通过？
  ...                        ✅
  tutorial/07 → practice/07  🎉 恭喜毕业！
```

### 练习文件结构

每个 practice 文件的结构如下：

```python
def test_q1():
    # 实现函数（你的任务）
    def create_range_tensor(start, end, step):
        # YOUR CODE HERE
        pass

    # 测试输入
    out = create_range_tensor(0, 6, 2)
    # 期望值被「封印」在单独命名的函数里
    assert torch.equal(out, _expected_q1())

# ... 更多 test_qN() ...

# ---------------------------------------------------------------------------
#  期望值区域（翻到这里才能看到答案！）
# ---------------------------------------------------------------------------
def _expected_q1():
    return torch.tensor([0, 2, 4])
```

**为什么要这样组织？**

传统的练习文件会把期望值直接写在 assert 里：

```python
def test_q1():
    def some_function(x):
        pass  # ← 你在实现

    assert some_function(x) == torch.tensor([0, 2, 4])  # ← 答案直接写在旁边！
```

这导致你在实现函数时，一眼就能看到期望值，练习效果大打折扣。

我们的改进：**把每个期望值「封印」进单独命名的 `_expected_qN()` 函数，统一放在文件底部**。
这样一来，你在 `test_qN()` 里只看到 `_expected_qN()` 这个调用，实际的期望值被隐藏在了文件末尾。

- ✅ VS Code ▶ 按钮仍然出现在 `test_qN()` 旁边，一键运行
- ✅ 实现和答案在视觉上彻底分离
- ✅ 想看答案？自己翻到文件底部 — 但这需要你主动做出选择

你的任务：**只改 `# YOUR CODE HERE` 区域，让所有 assert 通过**。

---

## 🛠 技术栈

- **Python 3.12+**
- **PyTorch 2.x** (CPU 即可学习全部内容)
- **uv** — 快速 Python 包管理器

---

## 📝 设计理念

> **深入浅出** — 每个概念先用直觉讲清楚「为什么」，再给出代码。

- **Tutorial**：带丰富注释的可执行脚本，把概念拆到最细
- **Practice**：LeetCode 式的函数填空，期望值被封印在文件底部的 `_expected_qN()` 函数中，答案不会在你实现时泄露
- **Progressive**：从 tensor 创建到完整训练流程，每一课都建立在上一课基础上

---

## 🎓 学完之后

你就能：
- 用 PyTorch 搭建任意网络结构
- 写出标准训练/验证循环
- 理解 autograd 和计算图
- 处理 Dataset/DataLoader
- 保存/加载模型和 checkpoint
- 把模型搬到 GPU 训练

下一步可以继续学：CNN、RNN/LSTM、Transformer、TorchScript、分布式训练…
