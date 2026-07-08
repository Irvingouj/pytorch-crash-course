"""
PyTorch 速成课 · 练习 7：GPU 训练 & 模型保存

每道题都是一个 pytest test。VS Code 的 ▶ 按钮会出现在 def test_qN() 旁边。
"""

import os
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim


def test_q1():
    # 第1题：获取设备
    def get_device() -> torch.device:
        """
        返回最佳可用设备：有 GPU 返回 cuda，否则返回 cpu。
        """
        # YOUR CODE HERE
        pass

    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda"}


def test_q2():
    # 第2题：把模型搬到设备
    def model_to_device(model: nn.Module, device: torch.device) -> nn.Module:
        """
        把模型搬到指定 device，返回模型本身。
        """
        # YOUR CODE HERE
        pass

    model = nn.Linear(2, 1)
    moved = model_to_device(model, torch.device("cpu"))
    assert moved is model
    assert next(model.parameters()).device.type == "cpu"


def test_q3():
    # 第3题：保存 state_dict
    def save_model_state(model: nn.Module, filepath: str):
        """
        用 torch.save 保存 model.state_dict() 到 filepath。
        """
        # YOUR CODE HERE
        pass

    model = nn.Linear(2, 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        save_model_state(model, path)
        assert os.path.exists(path)
        state = torch.load(path, weights_only=True)
        assert "weight" in state


def test_q4():
    # 第4题：加载 state_dict
    def load_model_state(model: nn.Module, filepath: str):
        """
        从 filepath 加载 state_dict 到 model，使用 weights_only=True。
        """
        # YOUR CODE HERE
        pass

    source = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        torch.save(source.state_dict(), path)
        load_model_state(target, path)
    assert torch.allclose(source.weight, target.weight)
    assert torch.allclose(source.bias, target.bias)


def test_q5():
    # 第5题：保存完整 checkpoint
    def save_checkpoint(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        best_acc: float,
        filepath: str,
    ):
        """
        保存 model_state_dict、optimizer_state_dict、epoch、best_acc。
        """
        # YOUR CODE HERE
        pass

    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "checkpoint.pt")
        save_checkpoint(model, optimizer, 3, 0.8, path)
        checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["epoch"] == _expected_q5_epoch()
    assert checkpoint["best_acc"] == _expected_q5_best_acc()
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_q6():
    # 第6题：加载完整 checkpoint
    def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filepath: str) -> tuple:
        """
        从 filepath 加载 checkpoint，恢复 model 和 optimizer，返回 (epoch, best_acc)。
        """
        # YOUR CODE HERE
        pass

    source = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    source_opt = optim.SGD(source.parameters(), lr=0.1)
    target_opt = optim.SGD(target.parameters(), lr=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "checkpoint.pt")
        torch.save({
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": source_opt.state_dict(),
            "epoch": 5,
            "best_acc": 0.9,
        }, path)
        epoch, best_acc = load_checkpoint(target, target_opt, path)
    assert epoch == _expected_q6_epoch()
    assert best_acc == _expected_q6_best_acc()
    assert torch.allclose(source.weight, target.weight)


def test_q7():
    # 第7题：把 batch 搬到设备
    def move_batch_to_device(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple:
        """
        将 (x, y) 搬到 device，返回 (x_on_device, y_on_device)。
        """
        # YOUR CODE HERE
        pass

    x, y = move_batch_to_device(torch.tensor([1]), torch.tensor([2]), torch.device("cpu"))
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"


def test_q8():
    # 第8题：验证模型参数在正确设备上
    def check_model_device(model: nn.Module) -> torch.device:
        """
        返回模型任意参数的 device；如果没有参数，返回 cpu。
        """
        # YOUR CODE HERE
        pass

    assert check_model_device(nn.Linear(2, 1)).type == "cpu"
    assert check_model_device(nn.ReLU()).type == "cpu"


def test_q9():
    # 第9题：完整保存/加载往返测试
    def roundtrip_save_load(model: nn.Module, x: torch.Tensor) -> bool:
        """
        保存 model.state_dict，创建同结构模型并加载，比较两次输出是否一致。
        """
        # YOUR CODE HERE
        pass

    model = nn.Linear(2, 1)
    x = torch.randn(3, 2)
    assert roundtrip_save_load(model, x) is _expected_q9()


# ---------------------------------------------------------------------------
#  Expected values (sealed away so you don't see the answer while implementing).
# ---------------------------------------------------------------------------

def _expected_q5_epoch():
    return 3

def _expected_q5_best_acc():
    return 0.8

def _expected_q6_epoch():
    return 5

def _expected_q6_best_acc():
    return 0.9

def _expected_q9():
    return True


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
