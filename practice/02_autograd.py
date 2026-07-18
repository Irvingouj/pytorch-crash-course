"""
PyTorch 速成课 · 练习 2：自动求导

每道题都是一个 pytest test。VS Code 的 ▶ 按钮会出现在 def test_qN() 旁边。
"""

import torch


def test_q1():
    # 第1题：计算梯度
    def compute_gradient_at_point(x_val: float) -> float:
        """
        计算 f(x) = x³ + 2x² - 5x + 1 在 x = x_val 处的导数。
        要求：使用 PyTorch autograd，不允许手动算导数公式。
        """
        x_tensor = torch.tensor(x_val, requires_grad=True)
        y = x_tensor ** 3 + 2 * x_tensor ** 2 - 5 * x_tensor + 1
        y.backward()
        return x_tensor.grad.item()

    out = compute_gradient_at_point(2.0)
    assert abs(out - _expected_q1()) < 1e-5, f"Q1: expected {_expected_q1()}, got {out}"


def test_q2():
    # 第2题：多元函数偏导数
    def compute_partial_derivatives(x_val: float, y_val: float):
        """
        计算 f(x, y) = x²·y + 3x·y² 在 (x_val, y_val) 处的两个偏导数。
        """
        
        x_tensor = torch.tensor(x_val, requires_grad=True)
        y_tensor = torch.tensor(y_val, requires_grad=True)
        f = x_tensor ** 2 * y_tensor + 3 * x_tensor * y_tensor ** 2
        f.backward()
        return x_tensor.grad.item(), y_tensor.grad.item()

    dx, dy = compute_partial_derivatives(1.0, 2.0)
    assert abs(dx - _expected_q2_dx()) < 1e-5, f"Q2: df/dx expected {_expected_q2_dx()}, got {dx}"
    assert abs(dy - _expected_q2_dy()) < 1e-5, f"Q2: df/dy expected {_expected_q2_dy()}, got {dy}"


def test_q3():
    # 第3题：验证链式法则
    def chain_rule_demo(x_val: float) -> float:
        """
        令 u = x² + 1, y = sin(u)，用 autograd 计算 dy/dx。
        """
        x_tensor = torch.tensor(x_val, requires_grad=True)
        u = x_tensor ** 2 + 1
        y = torch.sin(u)
        y.backward()
        return x_tensor.grad.item()

    out = chain_rule_demo(1.0)
    assert abs(out - _expected_q3()) < 1e-5, f"Q3: expected {_expected_q3()}, got {out}"


def test_q4():
    # 第4题：梯度不追踪
    def no_grad_computation(x: torch.Tensor) -> torch.Tensor:
        """
        对 x 做平方运算，但不要追踪梯度。
        """
        with torch.no_grad():
            return x*x

    x = torch.tensor(3.0, requires_grad=True)
    out = no_grad_computation(x)
    assert out.requires_grad is False
    assert out.item() == _expected_q4()


def test_q5():
    # 第5题：detach 的使用
    def detach_and_modify(x: torch.Tensor) -> torch.Tensor:
        """
        对 x 做 y = x * 2 + 3，然后返回 y 的 detached 版本。
        """
        return (x*2 + 3).detach()

    x = torch.tensor(2.0, requires_grad=True)
    out = detach_and_modify(x)
    assert out.requires_grad is False
    assert out.item() == _expected_q5()


def test_q6():
    # 第6题：梯度清零
    def accumulate_and_clear(x: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        重复 backward，但每次 backward 前要清零梯度，返回最终 x.grad。
        """
        for i in range(n_steps):
            if x.grad is not None:
                x.grad.zero_()
            y = x ** 2
            y.backward()

        return x.grad

    x = torch.tensor(2.0, requires_grad=True)
    out = accumulate_and_clear(x, 4)
    assert abs(out.item() - _expected_q6()) < 1e-5


def test_q7():
    # 第7题：向量 backward
    def vector_backward_demo() -> torch.Tensor:
        """
        对 y = x² 使用 gradient=[1, 1, 1] 做 backward，返回 x.grad。
        """
        x = torch.tensor([1.,2.,3.],requires_grad=True)
        y = x**2
        y.backward(gradient=torch.ones_like(y))
        return x.grad

    out = vector_backward_demo()
    assert torch.allclose(out, _expected_q7())


def test_q8():
    # 第8题：高阶导数
    def second_derivative(x_val: float) -> float:
        """
        计算 f(x) = x⁴ 在 x = x_val 处的二阶导数。
        """
        x = torch.tensor(x_val,requires_grad=True)
        y = x ** 4
        grad1 = torch.autograd.grad(y,x, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1,x)[0]
        return grad2

    out = second_derivative(2.0)
    assert abs(out - _expected_q8()) < 1e-5


# ---------------------------------------------------------------------------
#  Expected values (sealed away so you don't see the answer while implementing).
# ---------------------------------------------------------------------------

def _expected_q1():
    return 15.0

def _expected_q2_dx():
    return 16.0

def _expected_q2_dy():
    return 13.0

def _expected_q3():
    return torch.cos(torch.tensor(2.0)).item() * 2

def _expected_q4():
    return 9.0

def _expected_q5():
    return 7.0

def _expected_q6():
    return 4.0

def _expected_q7():
    return torch.tensor([2.0, 4.0, 6.0])

def _expected_q8():
    return 48.0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
