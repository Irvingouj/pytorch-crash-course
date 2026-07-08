"""
PyTorch 速成课 · 练习 5：自定义数据集

每道题都是一个 pytest test。VS Code 的 ▶ 按钮会出现在 def test_qN() 旁边。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def test_q1():
    # 第1题：实现一个简单的 Dataset
    class RangeDataset(Dataset):
        """
        生成 [0, N-1] 的整数 Dataset，每个样本返回 torch.long tensor。
        """
        def __init__(self, n: int):
            # YOUR CODE HERE
            pass

        def __len__(self):
            # YOUR CODE HERE
            pass

        def __getitem__(self, idx):
            # YOUR CODE HERE
            pass

    ds = RangeDataset(5)
    assert len(ds) == 5
    assert torch.equal(ds[3], _expected_q1())


def test_q2():
    # 第2题：带标签的 Dataset
    class LabeledDataset(Dataset):
        """
        __getitem__ 返回 (data[idx].float(), labels[idx].long())。
        """
        def __init__(self, data, labels):
            # YOUR CODE HERE
            pass

        def __len__(self):
            # YOUR CODE HERE
            pass

        def __getitem__(self, idx):
            # YOUR CODE HERE
            pass

    ds = LabeledDataset(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    x, y = ds[1]
    assert len(ds) == 2
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    assert torch.equal(x, _expected_q2_data())
    assert y.item() == _expected_q2_label()


def test_q3():
    # 第3题：带 transform 的 Dataset
    class TransformDataset(Dataset):
        """
        包装已有 Dataset，对 x 应用 transform，y 原样返回。
        """
        def __init__(self, base_dataset: Dataset, transform):
            # YOUR CODE HERE
            pass

        def __len__(self):
            # YOUR CODE HERE
            pass

        def __getitem__(self, idx):
            # YOUR CODE HERE
            pass

    base = TensorDataset(torch.tensor([[1.], [2.]]), torch.tensor([0, 1]))
    ds = TransformDataset(base, lambda x: x + 10)
    x, y = ds[0]
    assert len(ds) == 2
    assert torch.equal(x, _expected_q3_data())
    assert y.item() == _expected_q3_label()


def test_q4():
    # 第4题：用 DataLoader 收集 batch
    def collect_all_batches(dataset: Dataset, batch_size: int, shuffle: bool) -> list:
        """
        用 DataLoader 把 dataset 按 batch 分组，返回所有 batch 的列表。
        """
        # YOUR CODE HERE
        pass

    ds = TensorDataset(torch.arange(5), torch.arange(5) * 10)
    batches = collect_all_batches(ds, batch_size=2, shuffle=False)
    assert len(batches) == 3
    assert torch.equal(batches[0][0], _expected_q4_batch0())
    assert torch.equal(batches[-1][0], _expected_q4_batch_last())


def test_q5():
    # 第5题：样本数统计
    def count_batches_and_samples(dataset: Dataset, batch_size: int) -> tuple:
        """
        返回 (num_batches, total_samples)，drop_last=False。
        """
        # YOUR CODE HERE
        pass

    ds = TensorDataset(torch.arange(5), torch.arange(5))
    assert count_batches_and_samples(ds, batch_size=2) == _expected_q5()


def test_q6():
    # 第6题：自定义 batch 整理函数
    def my_collate_fn(batch):
        """
        data 和 label 都用 torch.stack 堆叠，返回 (stacked_data, stacked_labels)。
        """
        # YOUR CODE HERE
        pass

    batch = [(torch.tensor([1, 2]), torch.tensor(0)), (torch.tensor([3, 4]), torch.tensor(1))]
    x, y = my_collate_fn(batch)
    assert torch.equal(x, _expected_q6_data())
    assert torch.equal(y, _expected_q6_labels())


def test_q7():
    # 第7题：数据集切分
    def split_dataset(dataset: Dataset, train_ratio: float):
        """
        手动用 torch.randperm 切分，返回两个 TensorDataset。
        """
        # YOUR CODE HERE
        pass

    torch.manual_seed(0)
    ds = TensorDataset(torch.arange(10).float().unsqueeze(1), torch.arange(10))
    train_ds, val_ds = split_dataset(ds, 0.6)
    assert isinstance(train_ds, TensorDataset)
    assert isinstance(val_ds, TensorDataset)
    assert len(train_ds) == _expected_q7_train_len()
    assert len(val_ds) == _expected_q7_val_len()


def test_q8():
    # 第8题：数据标准化 transform
    def create_normalize_transform(mean: torch.Tensor, std: torch.Tensor):
        """
        返回一个函数：x -> (x - mean) / std。
        """
        # YOUR CODE HERE
        pass

    transform = create_normalize_transform(torch.tensor([1., 2.]), torch.tensor([2., 4.]))
    out = transform(torch.tensor([3., 10.]))
    assert torch.allclose(out, _expected_q8())


# ---------------------------------------------------------------------------
#  Expected values (sealed away so you don't see the answer while implementing).
# ---------------------------------------------------------------------------

def _expected_q1():
    return torch.tensor(3, dtype=torch.long)

def _expected_q2_data():
    return torch.tensor([3., 4.])

def _expected_q2_label():
    return 1

def _expected_q3_data():
    return torch.tensor([11.])

def _expected_q3_label():
    return 0

def _expected_q4_batch0():
    return torch.tensor([0, 1])

def _expected_q4_batch_last():
    return torch.tensor([4])

def _expected_q5():
    return (3, 5)

def _expected_q6_data():
    return torch.tensor([[1, 2], [3, 4]])

def _expected_q6_labels():
    return torch.tensor([0, 1])

def _expected_q7_train_len():
    return 6

def _expected_q7_val_len():
    return 4

def _expected_q8():
    return torch.tensor([1., 2.])


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
