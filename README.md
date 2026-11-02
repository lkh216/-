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
```
# 深度伪造视频检测：Xception + Patch分支 + 1D时序卷积

## 📁 项目结构

- `model.py`：完整模型：Xception + Patch 分支 + 1D 时序卷积
- `model_without_1d.py`：消融：去掉 1D 时序卷积，仅序列平均
- `model_without_patch.py`：消融：去掉 Patch 分支，仅 Xception + 1D 时序
- `model_xception.py`：消融：纯 Xception + 序列平均
- `train_on_ff++.py`：在 FF++ 上训练（完整模型）
- `train_on_celeb-df-v2.py`：在 Celeb-DF-v2 上训练（完整模型）
- `train_without_1d.py`：训练 model_without_1d
- `train_without_patch.py`：训练 model_without_patch
- `train_xception.py`：训练 model_xception
- `test_on_other_datasets.py`：跨数据集测试（Celeb-DF-v1）


## ⚙️ 环境依赖
```bash
pip install torch torchvision pillow numpy scikit-learn tqdm
```

推荐 Python ≥ 3.8，PyTorch ≥ 1.10，CUDA 可用时自动使用 GPU。

## 📂 数据准备

数据集需预先抽帧为 PNG 格式，以 FF++ 为例，目录结构如下：

```
frames/
├── original_sequences/youtube/c23/frames/
│   ├── video_001/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   └── ...
└── manipulated_sequences/
    ├── Deepfakes/c23/frames/
    ├── Face2Face/c23/frames/
    ├── FaceSwap/c23/frames/
    └── NeuralTextures/c23/frames/
```

每个视频目录至少包含 `seq_length × vectors_per_video = 4 × 4 = 16` 帧。

预训练权重：Xception 的 ImageNet 预训练权重（如 `xception-43020ad28.pth`），在脚本中通过 `pretrained_path` 参数指定。

## 🚀 训练

### 1. 在 FF++ 上训练完整模型

```
python train_on_ff++.py
```

关键配置（在 `main()` 中修改）：

表格

| 参数 | 说明 |
| --- | --- |
| ffpp_real_path | 真实视频帧路径 |
| ffpp_fake_paths | 伪造视频帧路径列表（4 种方法） |
| pretrained_path | Xception 预训练权重路径 |
| batch_size | 训练 4 / 测试 8 |
| phase1_epochs / phase2_epochs | 8 / 60 |
| lr | 8e-5 |

训练采用两阶段策略：

- 阶段 1：冻结 Xception，仅训练 Patch 分支、融合层、时序卷积与分类器。
- 阶段 2：解冻全部参数联合训练。

按验证集 AUC 保存最佳模型：`best_model_phase{1,2}_auc_{AUC}.pth`。

### 2. 在 Celeb-DF-v2 上训练

```
python train_on_celeb-df-v2.py
```

配置方式同上，路径改为 Celeb-DF-v2 的 `Celeb-real/frames` 与 `Celeb-synthesis/frames`。默认 `phase1_epochs=3`，`phase2_epochs=60`。

### 3. 消融实验训练

```
python train_without_1d.py      # 去掉 1D 时序卷积
python train_without_patch.py   # 去掉 Patch 分支
python train_xception.py        # 纯 Xception
```

三者的训练流程与 `train_on_ff++.py` 一致（`train_without_patch.py` 使用单阶段训练），最佳模型分别保存为 `best_model_auc_{AUC}.pth`。

## 🧪 测试（跨数据集评估）

```
python test_on_other_datasets.py
```

在 `main()` 中配置：

```
original_roots = [
    "/hy-tmp/celeb-df-v1/Celeb-real/frames",
    "/hy-tmp/celeb-df-v1/YouTube-real/frames"
]
manipulated_roots = [
    "/hy-tmp/celeb-df-v1/Celeb-synthesis/frames"
]
model_path = "/path/to/best_model_phase2_auc_xxx.pth"
batch_size = 16
```

输出指标：Accuracy、AUC、F1、Precision、Recall、混淆矩阵、平均推理时间 / 吞吐量，并按伪造方法分别统计，结果同时保存为 `test_results_{时间戳}.txt`。

## 🧠 模型说明

核心前向流程（`model.py`）

```
输入 [B, vectors, seq_len, C, H, W]
        │
        ├── Xception 编码器 ──► 2048 维特征
        │
        └── Patch 分支（PatchEmbedding + 2×AttentionBlock）──► 2048 维
        │
     concat + Linear(4096→2048) 融合
        │
     重塑为 [B*vectors, seq_len, 2048] → Conv1d 时序建模 → 分类
        │
     对 vectors 维平均 → 最终 logits
```

消融变体对比

表格

| 模型文件 | Xception | Patch 分支 | 1D 时序卷积 |
| --- | --- | --- | --- |
| model.py | ✅ | ✅ | ✅ |
| model_without_1d.py | ✅ | ✅ | ❌（序列平均） |
| model_without_patch.py | ✅ | ❌ | ✅ |
| model_xception.py | ✅ | ❌ | ❌（序列平均） |

## 📌 注意事项

- **数据泄露防护**：数据集按视频名分组后确定性划分（seed=42，split_ratio=0.8），并自动校验训练 / 测试视频无重叠。
- **图像预处理**：统一 `F.interpolate` 到 299×299，使用 ImageNet 均值方差归一化；训练时附加水平翻转与高斯模糊增强。
- **帧数不足**：训练脚本会跳过帧数不足的视频；测试脚本 `test_on_other_datasets.py` 支持黑帧补齐。
- **路径配置**：所有脚本中的绝对路径（`/hy-tmp/...`）需改为本地实际路径后再运行。
- **随机种子**：全局固定为 42，保证可复现。
