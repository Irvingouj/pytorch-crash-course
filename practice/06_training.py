"""
PyTorch 速成课 · 练习 6：训练流程

每道题都是一个 pytest test。VS Code 的 ▶ 按钮会出现在 def test_qN() 旁边。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _tiny_loader():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    y = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_q1():
    # 第1题：补全训练循环
    def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
    ) -> float:
        """
        训练一个 epoch，返回按样本平均的 loss。
        """
        # YOUR CODE HERE
        pass

    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss = train_one_epoch(model, _tiny_loader(), optimizer, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.equal(model.weight, before)


def test_q2():
    # 第2题：补全评估循环
    @torch.no_grad()
    def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
        """
        返回 {"loss": float, "accuracy": float}，不更新参数。
        """
        # YOUR CODE HERE
        pass

    model = nn.Linear(2, 2)
    out = evaluate(model, _tiny_loader(), torch.device("cpu"))
    assert set(out) == {"loss", "accuracy"}
    assert isinstance(out["loss"], float)
    assert 0.0 <= out["accuracy"] <= 1.0


def test_q3():
    # 第3题：Early Stopping
    class EarlyStopping:
        """
        val_loss 改善超过 min_delta 就重置计数，否则计数；计数 >= patience 返回 True。
        """
        def __init__(self, patience: int = 5, min_delta: float = 0.0):
            # YOUR CODE HERE
            pass

        def __call__(self, val_loss: float) -> bool:
            # YOUR CODE HERE
            pass

    stopper = EarlyStopping(patience=2, min_delta=0.1)
    assert stopper(1.0) is False
    assert stopper(0.95) is False
    assert stopper(0.94) is True


def test_q4():
    # 第4题：计算准确率
    def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        返回 argmax 预测准确率，Python float。
        """
        # YOUR CODE HERE
        pass

    logits = torch.tensor([[3., 1.], [1., 4.], [5., 0.]])
    labels = torch.tensor([0, 0, 0])
    assert compute_accuracy(logits, labels) == _expected_q4()


def test_q5():
    # 第5题：训练循环中记录指标
    def train_with_metrics(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_epochs: int,
    ) -> dict:
        """
        返回 train_loss、val_loss、val_acc 三个列表，每个长度为 num_epochs。
        """
        # YOUR CODE HERE
        pass

    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    history = train_with_metrics(model, _tiny_loader(), _tiny_loader(), optimizer, torch.device("cpu"), 2)
    assert set(history) == {"train_loss", "val_loss", "val_acc"}
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["val_acc"]) == 2


def test_q6():
    # 第6题：Gradient Clipping
    def train_one_epoch_with_clip(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        max_norm: float = 1.0,
    ) -> float:
        """
        和 train_one_epoch 一样，但 optimizer.step() 前调用 clip_grad_norm_。
        """
        # YOUR CODE HERE
        pass

    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = train_one_epoch_with_clip(model, _tiny_loader(), optimizer, torch.device("cpu"), max_norm=0.1)
    assert isinstance(loss, float)
    assert loss > 0


# ---------------------------------------------------------------------------
#  Expected values (sealed away so you don't see the answer while implementing).
# ---------------------------------------------------------------------------

def _expected_q4():
    return 2 / 3


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
