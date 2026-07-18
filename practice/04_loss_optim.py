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
        squre = (pred-target)**2
        return torch.mean(squre)

    pred = torch.tensor([1., 2., 3.])
    target = torch.tensor([1., 1., 5.])
    assert torch.allclose(manual_mse(pred, target), _expected_q1())


def test_q2():
    # 第2题：CrossEntropyLoss 的正确使用
    def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        用 nn.CrossEntropyLoss 计算 raw logits 和类别索引 labels 的损失。
        """
        
        return nn.CrossEntropyLoss()(logits, labels)

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
        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()

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
        if task == "regression":
            return nn.MSELoss()(pred, target)
        elif task == "binary":
            return nn.BCEWithLogitsLoss()(pred, target)
        elif task == "multiclass":
            return nn.CrossEntropyLoss()(pred, target)
        pass

    assert compute_loss_for_task("regression", torch.tensor([1.]), torch.tensor([2.])).ndim == 0
    assert compute_loss_for_task("binary", torch.tensor([0.]), torch.tensor([1.])).ndim == 0
    assert compute_loss_for_task("multiclass", torch.tensor([[1., 2.]]), torch.tensor([1])).ndim == 0


def test_q5():
    # 第5题：SGD 更新验证
    def verify_sgd_update(
        model: nn.Linear,
        optimizer: optim.Optimizer,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        对一个回归样本做一次训练 step，并返回更新后的 weight 张量。

        依次完成：zero_grad -> forward -> MSELoss -> backward -> step。
        不要新建模型或优化器：它们以及训练数据都由测试明确给出。
        """
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        optimizer.step()
        return model.weight.detach()

    # 这个例子刻意不用随机数：你应该能手算 prediction、gradient 和 SGD 更新。
    # bias=False 让本题只聚焦于 weight 的更新。
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([[5.0]])

    out = verify_sgd_update(model, optimizer, x, target)
    assert torch.allclose(out, _expected_q5(), atol=1e-6)


def test_q6():
    # 第6题：Optimizer 参数分组（不同模块用不同学习率）
    #
    # 真实场景：迁移学习中，预训练的 backbone 想用小学习率「微调」，
    # 而新加的分类头想用大学习率「快速学习」。这正是 param_groups 的用途。
    #
    # 要求：backbone 参数 lr=0.001，分类头（classifier）参数 lr=0.01。
    def create_two_lr_optimizer(model: nn.Module) -> optim.Optimizer:
        """
        把 model 里属于 backbone 的参数放进 lr=0.001 的组，
        属于 classifier 的参数放进 lr=0.01 的组，返回一个 SGD 优化器。

        提示：
          - 用 model.named_parameters() 拿到 (name, param) 对
          - 按 name 是否以 "classifier" 开头来分流
          - optim.SGD 接受一个 param_groups 的 list
        """
        # YOUR CODE HERE
        pass

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            # 模拟一个「预训练 backbone」+「新分类头」的结构
            self.backbone_fc1 = nn.Linear(2, 3)
            self.backbone_fc2 = nn.Linear(3, 4)
            self.classifier = nn.Linear(4, 2)

        def forward(self, x):
            x = torch.relu(self.backbone_fc1(x))
            x = torch.relu(self.backbone_fc2(x))
            return self.classifier(x)

    model = SimpleNet()
    opt = create_two_lr_optimizer(model)

    # ---- 结构断言（可见）：恰好两组，学习率正确 ----
    assert len(opt.param_groups) == 2, "应该有 2 个 param_groups"
    assert {g["lr"] for g in opt.param_groups} == {0.001, 0.01}, \
        "学习率应为 {0.001, 0.01}"

    # ---- 正确性断言：与隐藏的参考答案对比每个参数落在哪个 lr 组 ----
    expected = _expected_q6(model)  # 返回 {param_data_ptr: expected_lr}
    for group in opt.param_groups:
        for p in group["params"]:
            assert expected[p.data_ptr()] == group["lr"], \
                f"某参数应在 lr={expected[p.data_ptr()]} 组，但被分到了 lr={group['lr']} 组"


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

def _expected_q6(model: nn.Module) -> dict:
    # 返回 {param.data_ptr(): 应该所在的 lr}，答案对做题人不可见。
    return {
        p.data_ptr(): (0.01 if n.startswith("classifier") else 0.001)
        for n, p in model.named_parameters()
    }

def _expected_q7():
    return [1.0, 0.5, 0.25, 0.125]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
