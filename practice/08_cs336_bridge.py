"""
PyTorch 速成课 · 练习 8：CS336 桥梁课

只改 YOUR CODE HERE。前 8 题是小练习，第 9 题是综合 assignment。
这里允许 PyTorch built-ins；正式 CS336 Assignment 1 的限制更严格。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_q1():
    # 第1题：把 Unicode 文本变成 UTF-8 byte IDs
    def utf8_byte_ids(text: str) -> torch.Tensor:
        """返回 dtype=torch.long 的一维 tensor。"""
        # YOUR CODE HERE
        pass

    assert torch.equal(utf8_byte_ids("牛A"), _expected_q1())


def test_q2():
    # 第2题：构造 next-token prediction 输入与标签
    def next_token_pairs(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens=[t0,t1,...] -> x=[t0,t1,...-1], y=[t1,t2,...]。"""
        # YOUR CODE HERE
        pass

    x, y = next_token_pairs(torch.tensor([10, 20, 30, 40]))
    assert torch.equal(x, _expected_q2_x())
    assert torch.equal(y, _expected_q2_y())


def test_q3():
    # 第3题：把连续 token stream 切成固定长度 batch
    def make_lm_batch(tokens: torch.Tensor, batch_size: int, context_length: int):
        """使用前 batch_size * context_length + 1 个 token，返回相邻的 x/y。"""
        # YOUR CODE HERE
        pass

    x, y = make_lm_batch(torch.arange(13), batch_size=3, context_length=4)
    assert torch.equal(x, _expected_q3_x())
    assert torch.equal(y, _expected_q3_y())


def test_q4():
    # 第4题：估算一个 tied-embedding Transformer 的参数量
    def transformer_parameter_count(
        vocab_size: int, layers: int, d_model: int, d_ff: int
    ) -> int:
        """
        结构：一份 tied token embedding；每层 Q/K/V/O 四个无 bias Linear，
        SwiGLU 三个无 bias Linear，两个 RMSNorm weight。忽略 RoPE buffer。
        """
        # YOUR CODE HERE
        pass

    assert transformer_parameter_count(100, 2, 16, 32) == _expected_q4()


def test_q5():
    # 第5题：调用 PyTorch 的 causal attention 语义 oracle
    def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """调用 F.scaled_dot_product_attention，并开启 causal masking。"""
        # YOUR CODE HERE
        pass

    q = torch.tensor([[[[1.0], [1.0], [1.0]]]])
    k = torch.tensor([[[[1.0], [1.0], [1.0]]]])
    v = torch.tensor([[[[2.0], [4.0], [9.0]]]])
    assert torch.allclose(causal_attention(q, k, v), _expected_q5())


def test_q6():
    # 第6题：语言模型 cross entropy
    def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits=(B,T,V), targets=(B,T)；展平 B/T 后计算 cross entropy。"""
        # YOUR CODE HERE
        pass

    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])
    targets = torch.tensor([[0, 1]])
    assert torch.allclose(language_model_loss(logits, targets), _expected_q6())


def test_q7():
    # 第7题：top-k sampling 前过滤 logits
    def keep_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """每行只保留最大的 k 个 logits，其余位置改成 -inf。"""
        # YOUR CODE HERE
        pass

    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]])
    assert torch.equal(keep_top_k(logits, 2), _expected_q7())


def test_q8():
    # 第8题：估算 MHA KV cache 大小
    def kv_cache_bytes(
        layers: int,
        batch_size: int,
        num_kv_heads: int,
        sequence_length: int,
        head_dim: int,
        bytes_per_element: int,
    ) -> int:
        """K 和 V 各一份；返回总 bytes。"""
        # YOUR CODE HERE
        pass

    mha = kv_cache_bytes(2, 1, 8, 16, 8, 2)
    gqa = kv_cache_bytes(2, 1, 2, 16, 8, 2)
    assert (mha, gqa, mha // gqa) == _expected_q8()


def test_q9_assignment():
    # 综合作业：用 PyTorch built-ins 组装一个最小 decoder-only LM
    class TinyBuiltinLM(nn.Module):
        """
        要求：
        1. token embedding + learned position embedding
        2. 一个 nn.TransformerEncoderLayer；batch_first=True、norm_first=True、dropout=0
        3. 严格 causal mask，未来 token 不得影响过去位置
        4. Linear language-model head，输出 (B,T,vocab_size)

        这是 CS336 前置题。正式作业不能使用这些高层 nn built-ins。
        """

        def __init__(self, vocab_size: int, context_length: int, d_model: int, num_heads: int):
            super().__init__()
            # YOUR CODE HERE
            pass

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            # YOUR CODE HERE
            pass

    torch.manual_seed(0)
    model = TinyBuiltinLM(vocab_size=32, context_length=8, d_model=16, num_heads=4).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 3, 9]])
    logits_a = model(a)
    logits_b = model(b)

    assert logits_a.shape == (1, 4, 32)
    assert sum(p.numel() for p in model.parameters()) > 0
    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6)


# ---------------------------------------------------------------------------
# Expected values — 需要时再主动翻到这里。
# ---------------------------------------------------------------------------


def _expected_q1():
    return torch.tensor(list("牛A".encode("utf-8")), dtype=torch.long)


def _expected_q2_x():
    return torch.tensor([10, 20, 30])


def _expected_q2_y():
    return torch.tensor([20, 30, 40])


def _expected_q3_x():
    return torch.arange(12).reshape(3, 4)


def _expected_q3_y():
    return torch.arange(1, 13).reshape(3, 4)


def _expected_q4():
    embedding = 100 * 16
    attention_per_layer = 4 * 16 * 16
    swiglu_per_layer = 3 * 16 * 32
    norms_per_layer = 2 * 16
    return embedding + 2 * (attention_per_layer + swiglu_per_layer + norms_per_layer)


def _expected_q5():
    return torch.tensor([[[[2.0], [3.0], [5.0]]]])


def _expected_q6():
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])
    return F.cross_entropy(logits.reshape(-1, 2), torch.tensor([0, 1]))


def _expected_q7():
    return torch.tensor([[float("-inf"), 5.0, float("-inf"), 4.0]])


def _expected_q8():
    mha = 2 * 2 * 1 * 8 * 16 * 8 * 2
    gqa = 2 * 2 * 1 * 2 * 16 * 8 * 2
    return mha, gqa, 4


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
