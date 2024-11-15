# 探测潜在空间中的语义熵

## 概述

我们将探测器的训练以[笔记本](./train-latent-probe.ipynb)的形式呈现，笔记本在执行过程中会进行日志记录并生成可视化内容（如损失曲线、性能对比等），以增强对过程的理解。

## 教程

我们的方法涉及提取模型在两个位置（TBG 和 SLT）的隐藏状态，并在此基础上训练线性探测器，以评估模型的语义不确定性或正确性。

在语义熵（SE）生成运行中（参考[模型实现](../semantic_uncertainty/uncertainty/models/huggingface_models.py)），我们会保存模型的隐藏状态。如果你已通过 `wandb` 完成 SE 运行，隐藏状态（保存在 `validation_generations.pkl` 中）以及不确定性测量值（例如 `p_true`、token 的 `log likelihoods` 和 `semantic entropy`）应该已经就绪。这些是运行训练笔记本的唯一前置条件。

我们还支持将探测器（本质上是经过训练的逻辑回归模型）以 pickle 文件的形式保存到 `models` 文件夹（运行时会自动创建）中。之后可以自由使用探测器进行推理——只需对笔记本中的代码稍作修改，便可在某些特定的 token 位置（如 SLT 或 TBG）上对连接的隐藏状态运行探测器（SEP 或 Acc. Pr.），它将输出预测模型语义确定性及生成忠实答案的可能性的标签（或 logits）。

为了教学目的，我们提供了 [示例运行](https://wandb.ai/jiatongg/public_semantic_uncertainty)，基于 Llama-2-7B 模型（短文本生成），其内容与我们论文一致。

有关术语和其他技术细节，请参考[我们的论文](https://arxiv.org/abs/2406.15927)。

## 笔记本结构

该笔记本按以下部分组织：

- `Imports and Downloads` 部分帮助你将 wandb 的运行记录加载到本地存储；
- `Data Preparation` 部分准备训练数据，包含训练和评估代码的封装以及一些可视化工具；
- `Probing Acc/SE from Hidden States (IID)` 部分对 SE 进行二值化，并在同一数据集（不同划分）上实际训练 SEPs 和 Acc. Pr.，即 In-Distribution 设置；
- `Test probes trained with one dataset on others` 部分测试 SEPs 和 Acc. Pr. 在其他数据集上预测模型正确性的表现；
- 其余部分则用于与基线的性能比较及模型保存。

## 引用
```
@misc{kossen2024semanticentropyprobesrobust,
      title={Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs}, 
      author={Jannik Kossen and Jiatong Han and Muhammed Razzak and Lisa Schut and Shreshth Malik and Yarin Gal},
      year={2024},
      eprint={2406.15927},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.15927}, 
}
```
