# 项目名称
## 模型图
<img width="1206" height="768" alt="image" src="https://github.com/user-attachments/assets/91b6c88d-f955-46f4-8e5a-20c75976b69f" />
## 简介
本文提出 **XceptionTemporal1DWithPatches** 模型，用于视频伪造检测。模型结构如图 1 所示，由双分支特征提取、特征融合、时序建模与分类输出四部分组成。

### 双分支特征提取

模型包含两个并行分支：

- **Xception 分支（全局）**：以 Xception 为骨干网络，依次经过 Entry Flow、Middle Flow（8 个 block）和 Exit Flow，通过全局平均池化输出 2048 维全局语义特征 $f_{\text{xcep}}$。该分支负责捕捉整体伪造痕迹，支持加载 ImageNet 预训练权重。
- **Patch 分支（局部）**：将输入图像划分为 $100 \times 100$ 的 patch，通过卷积投影为 512 维嵌入，再经过两层多头自注意力（MHSA + LayerNorm）建模 patch 间关系，最后经全局平均池化与线性投影输出 2048 维局部特征 $f_{\text{patch}}$。该分支负责捕捉局部纹理异常。

### 特征融合

两分支特征拼接为 4096 维，通过融合层 $\text{Linear}(4096 \to 2048)$ 得到联合特征，实现全局语义与局部纹理的互补。

### 时序建模

将融合特征重塑为 $[B \cdot V, T, 2048]$，转置后送入一维时序卷积：

$$
h = \text{AdaptiveAvgPool1d}\Big(\text{Conv1d}_{1024 \to 512}\big(\text{Conv1d}_{2048 \to 1024}(F)\big)\Big)
$$

其中 $F \in \mathbb{R}^{B \cdot V \times 2048 \times T}$ 为融合特征的转置。该模块建模帧间关系，抑制单帧噪声，提升视频级预测的稳定性。

### 分类输出

最终通过全连接层 $\text{FC}(512 \to 256 \to 2)$ 输出真实/伪造的预测概率，并对同一视频的多个向量取平均得到视频级预测。

### 两阶段训练策略

- **阶段 1（8 epochs）**：冻结 Xception 分支，仅训练 Patch 分支与后续模块，快速建立局部特征表示。
- **阶段 2（60 epochs）**：解冻所有参数，联合训练整个模型，充分挖掘全局与局部特征的协同作用。
## 使用方法
