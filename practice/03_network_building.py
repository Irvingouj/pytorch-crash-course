"""
PyTorch 速成课 · 练习 3：搭建神经网络

每道题都是一个 pytest test。VS Code 的 ▶ 按钮会出现在 def test_qN() 旁边。
"""

import torch
import torch.nn as nn


def test_q1():
    # 第1题：单层网络
    class SimpleLinear(nn.Module):
        """
        一个简单的线性层：输入 10 维 -> 输出 5 维。
        """
        def __init__(self):
            super().__init__()
            # YOUR CODE HERE
            pass

        def forward(self, x):
            # YOUR CODE HERE
            pass

    model = SimpleLinear()
    out = model(torch.randn(4, 10))
    assert out.shape == (4, 5)
    assert isinstance(model, nn.Module)


def test_q2():
    # 第2题：两层 MLP
    class TwoLayerMLP(nn.Module):
        """
        两层全连接 + ReLU：input_dim -> hidden_dim -> ReLU -> output_dim。
        """
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            # YOUR CODE HERE
            pass

        def forward(self, x):
            # YOUR CODE HERE
            pass

    model = TwoLayerMLP(20, 64, 3)
    out = model(torch.randn(8, 20))
    assert out.shape == (8, 3)


def test_q3():
    # 第3题：用 Sequential 搭建
    def build_sequential_network(input_dim: int, output_dim: int) -> nn.Sequential:
        """
        用 nn.Sequential 搭建：input_dim -> 64 -> ReLU -> 32 -> ReLU -> output_dim。
        """
        # YOUR CODE HERE
        pass

    model = build_sequential_network(10, 5)
    assert isinstance(model, nn.Sequential)
    out = model(torch.randn(2, 10))
    assert out.shape == (2, 5)


def test_q4():
    # 第4题：带 Dropout 的网络
    class DropoutNet(nn.Module):
        """
        input_dim -> 128 -> ReLU -> Dropout(0.5) -> 64 -> ReLU -> Dropout(0.2) -> output_dim。
        """
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            # YOUR CODE HERE
            pass

        def forward(self, x):
            # YOUR CODE HERE
            pass

    model = DropoutNet(10, 3)
    x = torch.randn(4, 10)
    out_train = model(x)
    model.eval()
    out_eval = model(x)
    assert out_train.shape == (4, 3)
    assert out_eval.shape == (4, 3)


def test_q5():
    # 第5题：带 BatchNorm 的网络
    class BatchNormNet(nn.Module):
        """
        input_dim -> Linear(hidden) -> BatchNorm1d -> ReLU -> Linear(output_dim)。
        """
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            # YOUR CODE HERE
            pass

        def forward(self, x):
            # YOUR CODE HERE
            pass

    model = BatchNormNet(10, 32, 5)
    out = model(torch.randn(4, 10))
    assert out.shape == (4, 5)


def test_q6():
    # 第6题：共享权重（Siamese 风格）
    class SharedWeightNet(nn.Module):
        """
        一个 encoder 被两个输入共享，编码后 concat，再过 Linear 输出。
        """
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            # YOUR CODE HERE
            pass

        def forward(self, x1, x2):
            # YOUR CODE HERE
            pass

    model = SharedWeightNet(10, 16, 3)
    out = model(torch.randn(4, 10), torch.randn(4, 10))
    assert out.shape == (4, 3)


def test_q7():
    # 第7题：自定义参数
    class LearnableScale(nn.Module):
        """
        y = scale * x + bias，scale 和 bias 是可训练 nn.Parameter。
        """
        def __init__(self, dim: int):
            super().__init__()
            # YOUR CODE HERE
            pass

        def forward(self, x):
            # YOUR CODE HERE
            pass

    model = LearnableScale(5)
    out = model(torch.randn(3, 5))
    params = list(model.parameters())
    assert out.shape == (3, 5)
    assert len(params) == 2
    assert torch.allclose(params[0], _expected_q7_scale())
    assert torch.allclose(params[1], _expected_q7_bias())


def test_q8():
    # 第8题：Residual Block（残差块）
    class ResidualBlock(nn.Module):
        """
        out = ReLU(Linear(ReLU(Linear(x))) + x)。
        """
        def __init__(self, dim: int):
            super().__init__()
            # YOUR CODE HERE
            pass

        def forward(self, x):
            # YOUR CODE HERE
            pass

    model = ResidualBlock(16)
    out = model(torch.randn(4, 16))
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    assert out.shape == (4, 16)
    assert linear_count >= 2


# ---------------------------------------------------------------------------
#  Expected values (sealed away so you don't see the answer while implementing).
# ---------------------------------------------------------------------------

def _expected_q7_scale():
    return torch.ones(5)

def _expected_q7_bias():
    return torch.zeros(5)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
