# Transformer 与 Inference Resources

## Knowledge

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2026/)
  主线课程。用 Lecture 2–3 建立 tensor resource accounting 与现代 Transformer 架构，用 Assignment 1 检验实现能力。
- [CS336 Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics)
  正式综合作业：byte-level BPE、Transformer LM、AdamW、训练与生成。本速成课只做前置训练，不提供其实现答案。
- [PyTorch: scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
  用作 causal attention 的语义 oracle；CS336 作业中仍需按 handout 从零实现。
- [Karpathy: nanoGPT](https://github.com/karpathy/nanoGPT)
  较小的 GPT 训练参考实现。在完成本桥梁课后阅读，用来追踪模型与训练数据流。

## Wisdom (Communities)

- [Stanford CS336 GitHub Discussions](https://github.com/stanford-cs336/assignment1-basics/discussions)
  用于核对 handout 歧义与已知问题；不用于寻找作业答案。
