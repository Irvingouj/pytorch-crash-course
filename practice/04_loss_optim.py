"""
PyTorch 速成课 · 练习 4：损失函数 & 优化器

每道题都是一个 pytest test。VS Code 的 ▶ 按钮会出现在 def test_qN() 旁边。
"""

import torch
import torch.nn as nn
import torch.optim as optim


def test_q1():
    # 第1题：手动计算 MSE
    def manual_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        不用 nn.MSELoss，手动实现 mean((pred - target)^2)。
        """
        # YOUR CODE HERE
        pass

    pred = torch.tensor([1., 2., 3.])
    target = torch.tensor([1., 1., 5.])
    assert torch.allclose(manual_mse(pred, target), _expected_q1())


def test_q2():
    # 第2题：CrossEntropyLoss 的正确使用
    def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        用 nn.CrossEntropyLoss 计算 raw logits 和类别索引 labels 的损失。
        """
        # YOUR CODE HERE
        pass

    logits = torch.tensor([[2., 0.], [0., 2.]])
    labels = torch.tensor([0, 1])
    assert torch.allclose(cross_entropy_loss(logits, labels), _expected_q2())


def test_q3():
    # 第3题：一个完整的训练 step
    def one_training_step(
        model: nn.Module,
        optimizer: optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        """
        zero_grad -> forward -> CrossEntropyLoss -> backward -> step -> loss.item()。
        """
        # YOUR CODE HERE
        pass

    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss = one_training_step(model, optimizer, torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    assert isinstance(loss, float)
    assert not torch.equal(model.weight, before)


def test_q4():
    # 第4题：选择正确的 Loss
    def compute_loss_for_task(task: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        regression -> MSELoss, binary -> BCEWithLogitsLoss, multiclass -> CrossEntropyLoss。
        """
        # YOUR CODE HERE
        pass

    assert compute_loss_for_task("regression", torch.tensor([1.]), torch.tensor([2.])).ndim == 0
    assert compute_loss_for_task("binary", torch.tensor([0.]), torch.tensor([1.])).ndim == 0
    assert compute_loss_for_task("multiclass", torch.tensor([[1., 2.]]), torch.tensor([1])).ndim == 0


def test_q5():
    # 第5题：SGD 更新验证
    def verify_sgd_update():
        """
        手动设置 Linear 权重，做一次 SGD(lr=0.1) 更新，返回更新后的权重 list。
        """
        # YOUR CODE HERE
        pass

    out = verify_sgd_update()
    assert torch.allclose(torch.tensor(out), _expected_q5(), atol=1e-6)


def test_q6():
    # 第6题：Optimizer 参数分组
    def create_two_lr_optimizer(model: nn.Module) -> optim.Optimizer:
        """
        名称中包含 fc1 的参数 lr=0.001，其他参数 lr=0.01。
        """
        # YOUR CODE HERE
        pass

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 1)

    opt = create_two_lr_optimizer(SimpleNet())
    lrs = sorted(group["lr"] for group in opt.param_groups)
    assert lrs == _expected_q6()


def test_q7():
    # 第7题：学习率调度
    def simulate_lr_decay(initial_lr: float, gamma: float, steps: int) -> list:
        """
        用 StepLR(step_size=1, gamma=gamma) 模拟 steps 步学习率变化。
        """
        # YOUR CODE HERE
        pass

    assert simulate_lr_decay(1.0, 0.5, 3) == _expected_q7()


# ---------------------------------------------------------------------------
#  Expected values (sealed away so you don't see the answer while implementing).
# ---------------------------------------------------------------------------

def _expected_q1():
    return torch.tensor(5 / 3)

def _expected_q2():
    logits = torch.tensor([[2., 0.], [0., 2.]])
    labels = torch.tensor([0, 1])
    return nn.CrossEntropyLoss()(logits, labels)

def _expected_q5():
    return torch.tensor([[1.4, 2.4]])

def _expected_q6():
    return [0.001, 0.01]

def _expected_q7():
    return [1.0, 0.5, 0.25, 0.125]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
