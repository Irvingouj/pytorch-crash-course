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
        squre = (pred - target) ** 2
        return torch.mean(squre)

    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 1.0, 5.0])
    assert torch.allclose(manual_mse(pred, target), _expected_q1())


def test_q2():
    # 第2题：CrossEntropyLoss 的正确使用
    def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        用 nn.CrossEntropyLoss 计算 raw logits 和类别索引 labels 的损失。
        """

        return nn.CrossEntropyLoss()(logits, labels)

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
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
    loss = one_training_step(
        model, optimizer, torch.randn(4, 2), torch.tensor([0, 1, 0, 1])
    )
    assert isinstance(loss, float)
    assert not torch.equal(model.weight, before)


def test_q4():
    # 第4题：选择正确的 Loss
    def compute_loss_for_task(
        task: str, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
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

    assert (
        compute_loss_for_task(
            "regression", torch.tensor([1.0]), torch.tensor([2.0])
        ).ndim
        == 0
    )
    assert (
        compute_loss_for_task("binary", torch.tensor([0.0]), torch.tensor([1.0])).ndim
        == 0
    )
    assert (
        compute_loss_for_task(
            "multiclass", torch.tensor([[1.0, 2.0]]), torch.tensor([1])
        ).ndim
        == 0
    )


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
        params = model.named_parameters()
        classifier_params = []
        backbone_params = []
        for name, p in params:
            if name.startswith("classfier"):
                classifier_params.append(p)
            if name.startswith("backbone"):
                backbone_params.append(p)

        opt_b = optim.SGD(
            [
                {"params": classifier_params, "lr": 0.01},
                {"params": backbone_params, "lr": 0.001},
            ]
        )

        return opt_b

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
    assert {g["lr"] for g in opt.param_groups} == {0.001, 0.01}, (
        "学习率应为 {0.001, 0.01}"
    )

    # ---- 正确性断言：与隐藏的参考答案对比每个参数落在哪个 lr 组 ----
    expected = _expected_q6(model)  # 返回 {param_data_ptr: expected_lr}
    for group in opt.param_groups:
        for p in group["params"]:
            assert expected[p.data_ptr()] == group["lr"], (
                f"某参数应在 lr={expected[p.data_ptr()]} 组，但被分到了 lr={group['lr']} 组"
            )


def test_q7():
    # 第7题：学习率调度（StepLR）
    #
    # 真实场景：训练 100 个 epoch，希望 lr 每个 epoch 都乘以 gamma（逐步降速）。
    # 这正是 StepLR(step_size=1) 的用途 —— 每个 epoch 衰减一次。
    #
    # 学习率调度器（scheduler）本身不持有 lr —— 它是「挂」在一个 optimizer 上的，
    # 每次调用 scheduler.step() 就去修改那个 optimizer 的 lr。所以「scheduler
    # 离开 optimizer 就没意义」。本题把这套东西都摆在台面上：测试会先帮你
    # 建好 optimizer 和绑定的 StepLR，再传进你的函数，你只要把衰减后的 lr
    # 序列记录下来。
    #
    # 你要实现 record_lr_schedule(optimizer, scheduler, steps) -> list[float]：
    #   - optimizer 和 scheduler 已经配好对了（scheduler 绑在这个 optimizer 上），
    #     你不需要、也不应该新建它们。
    #   - 你要做的只是「记录」：把「初始 lr」加上「每次 scheduler.step() 之后的 lr」
    #     依次收进一个 list 返回，总长度 = steps + 1（初始 + 每步一个）。
    #
    # 从 optimizer 身上读当前 lr 的办法：
    #     optimizer.param_groups[0]["lr"]
    # （一个 optimizer 可以有多组参数、每组一个 lr；这里只有一组，所以取 [0]。）
    #
    # ⚠️ 必须真的调用 scheduler.step()，不能手写 initial_lr * gamma**t。
    #
    # 示例：optimizer 初始 lr = 1.0、scheduler 是 StepLR(step_size=1, gamma=0.5)
    #       record_lr_schedule(optimizer, scheduler, steps=3)
    #   t=0: 初始 lr               = 1.0
    #   t=1: scheduler.step() 一次 → 0.5
    #   t=2: 再 step 一次          → 0.25
    #   t=3: 再 step 一次          → 0.125
    #   返回 → [1.0, 0.5, 0.25, 0.125]   （长度 = steps + 1 = 4）
    def record_lr_schedule(
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.StepLR,
        steps: int,
    ) -> list:
        """
        optimizer 和 scheduler 已配好对。记录 lr 衰减序列并返回。

        Args:
            optimizer: 已绑在 scheduler 上的优化器（lr 从这里读）
            scheduler: StepLR(step_size=1, gamma=...)（调它的 step()）
            steps:     衰减次数（返回长度 = steps + 1，含初始值）

        Returns:
            [lr_0, lr_1, ..., lr_steps]，lr_0 为初始 lr，之后每 step 一次追加一个。
        """
        lrs = [float(optimizer.param_groups[0]["lr"])]
        for _ in range(steps):
            scheduler.step()
            lrs.append(float(optimizer.param_groups[0]["lr"]))
        return lrs

    assert _expected_q7(record_lr_schedule)


# ---------------------------------------------------------------------------
#  Expected values (sealed away so you don't see the answer while implementing).
# ---------------------------------------------------------------------------


def _expected_q1():
    return torch.tensor(5 / 3)


def _expected_q2():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
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


def _expected_q7(record_fn) -> bool:
    """
    封装所有测试用例（答案对做题人不可见）。
    接收学生的 record_lr_schedule 函数，跑全部断言，返回 True / 抛 AssertionError。
    """
    import torch as _t
    import torch.optim as _optim

    def _make_pair(initial_lr, gamma):
        """建一对配好的 (optimizer, StepLR)，和题面承诺的设置完全一致。"""
        dummy = _t.tensor([0.0], requires_grad=True)
        opt = _optim.SGD([dummy], lr=initial_lr)
        sched = _optim.lr_scheduler.StepLR(opt, step_size=1, gamma=gamma)
        return opt, sched

    def _ref(initial_lr, gamma, steps):
        """参考答案：用真正的 StepLR 跑出来的 lr 序列。"""
        opt, sched = _make_pair(initial_lr, gamma)
        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(steps):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        return [float(x) for x in lrs]

    def _close(a, b, tol=1e-6):
        return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))

    # ---- 主测试：标准衰减 ----
    opt, sched = _make_pair(1.0, 0.5)
    r = record_fn(opt, sched, 3)
    assert isinstance(r, list), "应该返回 list（不是 tensor）"
    assert len(r) == 4, f"steps=3 应返回 4 个值（含初始），实际 {len(r)}"
    assert all(isinstance(x, float) for x in r), "列表元素应为 float"
    assert _close(r, _ref(1.0, 0.5, 3)), f"标准衰减不符：{_ref(1.0, 0.5, 3)} vs {r}"

    # ---- 边界 1：steps=0 → 只返回初始 lr ----
    opt, sched = _make_pair(0.1, 0.5)
    assert record_fn(opt, sched, 0) == [0.1], "steps=0 应只返回 [initial_lr]"

    # ---- 边界 2：gamma=1.0 → lr 永远不变 ----
    opt, sched = _make_pair(0.3, 1.0)
    assert _close(record_fn(opt, sched, 5), [0.3] * 6), "gamma=1.0 时 lr 不应变"

    # ---- 边界 3：不同 gamma（防硬编码 0.5 的答案）----
    opt, sched = _make_pair(1.0, 0.1)
    assert _close(record_fn(opt, sched, 3), _ref(1.0, 0.1, 3)), "gamma=0.1 不符"

    # ---- 边界 4：不同 initial_lr ----
    opt, sched = _make_pair(2.0, 0.5)
    assert _close(record_fn(opt, sched, 2), [2.0, 1.0, 0.5]), "initial_lr=2.0 不符"

    return True


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
