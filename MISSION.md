# Mission: 从 PyTorch 走到 LLM inference engine

## Why
用可运行的小模型建立 Transformer inference 的真实心智模型，最终能独立实现并解释一个 Rust/CUDA inference engine，而不是只会调用模型 API。

## Success looks like
- 能从 token IDs 解释到 logits 和逐 token decode 的完整数据流
- 能完成 CS336 Assignment 1，并独立写出 CPU-only tiny decoder
- 能解释 prefill、decode、causal attention 与 KV cache 的成本

## Constraints
- 中文解释，notebook 学习，pytest LeetCode 风格练习
- 假设 PyTorch 速成课 1–7 已全部完成，不重复 tensor、autograd、nn.Module 或训练循环基础
- 先用 CPU 和 PyTorch 验证正确性，再进入 Rust/CUDA

## Out of scope
- 当前阶段不做 production serving、分布式推理或自定义 CUDA kernel
