"""
PyTorch 速成课 · 练习 1: Tensor 操作

每道题都是一个 pytest test。VS Code 的 ▶ 按钮会出现在 def test_qN() 旁边。
"""

import numpy as np
import torch
from einops import rearrange, reduce, repeat


def test_q1():
    # 第1题：创建 Tensor
    def create_range_tensor(start: int, end: int, step: int) -> torch.Tensor:
        """
        返回从 start 到 end（不含），步长为 step 的 1D Tensor。
        示例: create_range_tensor(0, 6, 2) -> tensor([0, 2, 4])
        """
        tensor = torch.arange(start, end, step)
        return tensor

    out = create_range_tensor(0, 6, 2)
    assert torch.equal(out, _expected_q1())


def test_q2():
    # 第2题：创建全零/全一张量
    def make_chessboard(n: int) -> torch.Tensor:
        """
        返回一个 n×n 的棋盘矩阵（0 和 1 交替，左上角为 0）。
        """
        tensor = torch.zeros((n, n))
        # TODO: this 100% can be done without loops
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 1:
                    tensor[i, j] = 1
        return tensor

    expected = _expected_q2()
    assert torch.equal(make_chessboard(3), expected)


def test_q3():
    # 第3题：张量变形
    def reshape_to_square(t: torch.Tensor) -> torch.Tensor:
        """
        将一个 1D Tensor 变形为方阵。假设元素个数是完全平方数。
        """
        # YOUR CODE HERE



        pass

    out = reshape_to_square(torch.tensor([1, 2, 3, 4]))
    assert torch.equal(out, _expected_q3())


def test_q4():
    # 第4题：矩阵乘法
    def matmul_manual(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        用 torch 操作实现矩阵乘法 A @ B（不允许直接用 @ 或 torch.mm/torch.matmul）。
        """
        # YOUR CODE HERE
        pass

    A = torch.tensor([[1., 2.], [3., 4.]])
    B = torch.tensor([[5., 6.], [7., 8.]])
    assert torch.equal(matmul_manual(A, B), _expected_q4())


def test_q5():
    # 第5题：广播加法
    def broadcast_add_column_vector(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        把 vec 加到 matrix 的每一列上。matrix: (m, n), vec: (m,)。
        """
        # YOUR CODE HERE
        pass

    matrix = torch.tensor([[1, 2], [3, 4]])
    vec = torch.tensor([10, 20])
    assert torch.equal(broadcast_add_column_vector(matrix, vec), _expected_q5())


def test_q6():
    # 第6题：统计计算
    def normalize_rows(t: torch.Tensor) -> torch.Tensor:
        """
        对每一行做 Min-Max 归一化。全相同行返回 0.5。
        """
        # YOUR CODE HERE
        pass

    x = torch.tensor([[1., 2., 3.], [5., 5., 5.]])
    expected = _expected_q6()
    assert torch.allclose(normalize_rows(x), expected)


def test_q7():
    # 第7题：索引与掩码
    def replace_negatives_with_zero(t: torch.Tensor) -> torch.Tensor:
        """
        将 t 中所有负值替换为 0。不允许用循环。
        """
        # YOUR CODE HERE
        pass

    out = replace_negatives_with_zero(torch.tensor([-1, 2, -3, 4]))
    assert torch.equal(out, _expected_q7())


def test_q8():
    # 第8题：高级索引
    def get_diagonal_elements(t: torch.Tensor) -> torch.Tensor:
        """
        返回方阵 t 的对角线元素（用索引，不允许用 torch.diag）。
        """
        # YOUR CODE HERE
        pass

    out = get_diagonal_elements(torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(out, _expected_q8())


def test_q9():
    # 第9题：Torch ↔ NumPy
    def numpy_to_torch_add_one(arr: np.ndarray) -> torch.Tensor:
        """
        接收 NumPy 数组，转成共享内存的 torch.Tensor，所有元素加 1，返回。
        """
        # YOUR CODE HERE
        pass

    arr = np.array([1., 2., 3.])
    out = numpy_to_torch_add_one(arr)
    assert torch.equal(out, _expected_q9_tensor())
    assert np.allclose(arr, _expected_q9_numpy())


def test_q10():
    # 第10题：维度操作
    def stack_along_new_axis(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        将两个同形状 tensor 沿第 0 维堆叠。
        """
        # YOUR CODE HERE
        pass

    out = stack_along_new_axis(torch.tensor([1, 2]), torch.tensor([3, 4]))
    assert torch.equal(out, _expected_q10())


def test_q11():
    # 第11题：einops rearrange — 图像 Flatten
    def flatten_images(imgs: torch.Tensor) -> torch.Tensor:
        """
        用 einops.rearrange 将 (batch, channel, height, width) 展平为 (batch, channel*height*width)。
        """
        # YOUR CODE HERE
        pass

    imgs = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    out = flatten_images(imgs)
    assert out.shape == (2, 48)
    assert torch.equal(out[0], imgs[0].flatten())


def test_q12():
    # 第12题：einops rearrange — 维度交换
    def swap_time_and_batch(x: torch.Tensor) -> torch.Tensor:
        """
        用 einops.rearrange 将 (batch, time, features) 变成 (time, batch, features)。
        """
        # YOUR CODE HERE
        pass

    x = torch.randn(4, 10, 32)
    out = swap_time_and_batch(x)
    assert out.shape == (10, 4, 32)
    assert torch.equal(out[0, 0], x[0, 0])


def test_q13():
    # 第13题：einops reduce — 按维度求均值
    def mean_over_spatial(x: torch.Tensor) -> torch.Tensor:
        """
        用 einops.reduce 对空间维度 (h, w) 求均值。
        """
        # YOUR CODE HERE
        pass

    x = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    assert torch.allclose(mean_over_spatial(x), x.mean(dim=(2, 3)))


def test_q14():
    # 第14题：einops repeat — 扩展维度
    def repeat_to_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        用 einops.repeat 将单个样本 (features,) 扩展成 (batch_size, features)。
        """
        # YOUR CODE HERE
        pass

    out = repeat_to_batch(torch.tensor([1, 2, 3]), 2)
    assert torch.equal(out, _expected_q14())


# ---------------------------------------------------------------------------
#  Expected values (sealed away so you don't see the answer while implementing).
# ---------------------------------------------------------------------------

def _expected_q1():
    return torch.tensor([0, 2, 4])

def _expected_q2():
    return torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])

def _expected_q3():
    return torch.tensor([[1, 2], [3, 4]])

def _expected_q4():
    return torch.tensor([[19., 22.], [43., 50.]])

def _expected_q5():
    return torch.tensor([[11, 12], [23, 24]])

def _expected_q6():
    return torch.tensor([[0., 0.5, 1.], [0.5, 0.5, 0.5]])

def _expected_q7():
    return torch.tensor([0, 2, 0, 4])

def _expected_q8():
    return torch.tensor([1, 4])

def _expected_q9_tensor():
    return torch.tensor([2., 3., 4.])

def _expected_q9_numpy():
    return np.array([2., 3., 4.])

def _expected_q10():
    return torch.tensor([[1, 2], [3, 4]])

def _expected_q14():
    return torch.tensor([[1, 2, 3], [1, 2, 3]])


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
