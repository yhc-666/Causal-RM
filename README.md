# Causal Reward Model

基于因果推断的奖励模型研究框架，用于研究选择偏差(Selection Bias)和 PU Learning（Positive-Unlabeled Learning，隐反馈/部分标签学习）对奖励模型的影响，以及各种去偏方法的效果。

## 项目结构

```
causal-rm/
├── embeddings/                   # Embedding 存储目录
│   ├── normal/                   # Stage 1 输出 (原始 embedding)
│   └── biased_pu/                # Stage 2 输出 (含偏差/PU)
│
├── rawdata/                      # 原始数据集目录 (HuggingFace datasets 格式)
│   ├── hs/                       # HelpSteer 数据集
│   ├── ufb/                      # UltraFeedback 数据集
│   └── saferlhf/                 # SaferLHF 数据集
│
├── prepare.py                    # Stage 1: Embedding 生成
├── data_prepare.sh               # Stage 1 批处理脚本
│
├── simulate_bias_pu.py           # Stage 2: 偏差/PU 模拟
├── simulate_bias_pu.sh           # Stage 2 批处理脚本
│
├── models_debias/                # Stage 3: 去偏方法 (debias, mask-blind PU setting)
├── models_pu/                    # Stage 3: PU 方法 (PU-only)
├── models_debias_pu/             # Stage 3: 去偏 + PU 方法 (debias + PU)
│
├── merge_rm.py                   # 合并多个 RM 结果
├── merge_rm.sh
│
├── analyze/
│   └── causal_rm.ipynb           # 结果分析 Notebook
│
├── scripts/                      # 实验脚本
│
├── requirements.txt              # Python 依赖
└── README.md
```

## 环境配置

```bash
# 创建 conda 环境
conda create -n rm python=3.11
conda activate rm

# 安装 PyTorch (CUDA 12.4)
pip install torch==2.6.0+cu124 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 安装 Flash Attention (可选，提升推理速度)
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 安装其他依赖
pip install -r requirements.txt
```

## 数据处理流程

```
┌─────────────────────┐
│  rawdata/{dataset}/ │   HuggingFace datasets 格式
└──────────┬──────────┘
           │
           ▼  Stage 1: data_prepare.sh → prepare.py
           │
┌──────────────────────────────────────────────────────┐
│  {model_name}_{data_name}_{train/test}.safetensors   │   Embeddings
└──────────┬───────────────────────────────────────────┘
           │
           ▼  Stage 2: simulate_bias_pu.sh → simulate_bias_pu.py
           │
┌──────────────────────────────────────────────────────┐
│  {model_name}_{data_name}_{alpha}_pu.safetensors     │  Biased + PU
└──────────┬───────────────────────────────────────────┘
           │
           ▼  Stage 3: benchmark_*.py
           │
┌──────────────────────────────────────────────────────┐
│  ./results/{method}/{data_name}/                     │   评估结果
└──────────────────────────────────────────────────────┘
```

---

## Stage 1: Embedding 生成

使用预训练奖励模型提取文本的 embedding 表示。

### 脚本

```bash
bash data_prepare.sh
```

### 参数 (`prepare.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 奖励模型路径 | (需配置) |
| `--output_dir` | embedding 输出目录 | `./embeddings/normal` |
| `--data_name` | 数据集名称 | `ufb` |
| `--subset` | `train` 或 `test` | `train` |
| `--batch_size` | 批处理大小 | `32` |
| `--num_workers` | 数据加载线程数 | `30` |

### 输入

```
./rawdata/{data_name}/    # HuggingFace datasets 格式
```

### 输出

```
{output_dir}/{model_name}_{data_name}_{subset}.safetensors

# 格式:
{
    "embeddings": Tensor[N, D],  # N 个样本, D 维 embedding
    "labels": Tensor[N],         # 连续奖励标签
    "user_id": Tensor[N],        # int64, 每条样本对应的 prompt/group id（用于后续 pointwise/pairwise 分组采样）
}
```

> 说明：`user_id` 会优先使用 rawdata 中的 `prompt_id`（若存在），否则用 `prompt` 文本构造一个稳定的哈希 id。

### 支持的数据集

| 数据集 | 标签类型 | 标签范围 |
|--------|----------|----------|
| `hs` | helpfulness | 0-4 |
| `ufb` | score | 1-10 |
| `saferlhf` | severity_level | 0-3 |
| `armorm` | 多属性 (19维) | 各不相同 |

---

## Stage 2: 偏差/PU 模拟

在 embedding 数据上模拟选择偏差和 PU Learning 场景。

### 脚本

```bash
bash simulate_bias_pu.sh
```

### 参数 (`simulate_bias_pu.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name` | 模型名称 | `FsfairX-LLaMA3-RM-v0.1` |
| `--data_name` | 数据集名称 | `saferlhf` |
| `--data_root` | embedding 输入目录 | `./embeddings/normal` |
| `--output_dir` | 输出目录 | `./embeddings/biased_pu` |
| `--alpha` | 选择偏差强度 (越小偏差越强) | `0.5` |

### 输入

```
{data_root}/{model_name}_{data_name}_train.safetensors
{data_root}/{model_name}_{data_name}_test.safetensors
```

### 输出

```
{output_dir}/{model_name}_{data_name}_{alpha}_pu.safetensors
{output_dir}/{model_name}_{data_name}_{alpha}_pu_stats.yaml   # 统计信息 (样本数/unique user/PU mask 等)

# 格式:
{
    # Embeddings (80/20 train/val split)
    "X_train": Tensor[N_train, D],
    "X_val": Tensor[N_val, D],
    "X_test": Tensor[N_test, D],

    # 原始连续标签
    "y_train": Tensor[N_train],
    "y_val": Tensor[N_val],
    "y_test": Tensor[N_test],

    # 二值标签 (PU 场景)
    "y_train_binary": Tensor[N_train],  # 0/1, mask=False 的正样本被标记为 0
    "y_val_binary": Tensor[N_val],
    "y_test_binary": Tensor[N_test],    # 完整标签 (用于评估)

    # 倾向性分数 (用于 IPS 等去偏方法)
    "propensity_train": Tensor[N_train],
    "propensity_val": Tensor[N_val],

    # 观察掩码 (True = 被观察到/标记为正)
    "mask_train": Tensor[N_train],
    "mask_val": Tensor[N_val],

    # (可选) prompt/group id（若 Stage 1 输入包含 user_id）
    "user_id_train": Tensor[N_train],  # int64
    "user_id_val": Tensor[N_val],      # int64
    "user_id_test": Tensor[N_test],    # int64
}
```

### PU Learning 模拟原理

- **选择偏差**: 低奖励样本的 propensity score 更低，更不容易被"观察到"
  - `propensity = alpha^(max_reward - reward)`
  - `alpha=0.5` 时偏差较弱，`alpha=0.01` 时偏差很强

- **PU Learning**: 只有部分正样本被标记，未被标记的正样本与负样本混在一起
  - mask=True: 样本被观察到，标签可信
  - mask=False: 样本未被观察到，正样本被标记为 0（隐反馈）

---

## Stage 3: 模型训练与评估

使用各种去偏方法训练奖励模型并评估。

### 示例

```bash
# Naive baseline (无去偏)
python models_debias/benchmark_naive.py --data_name saferlhf --alpha 0.1

# IPS (Inverse Propensity Scoring)
python models_debias/benchmark_ips.py --data_name saferlhf --alpha 0.1

# Doubly Robust
python models_debias/benchmark_dr.py --data_name saferlhf --alpha 0.1

# Counter-IF (debias + PU)
python models_debias_pu/benchmark_counterif.py --data_name hs --alpha 0.2

# ReCRec-F (debias + PU)
python models_debias_pu/benchmark_recrec.py --data_name hs --alpha 0.2 --variant F
```

### 通用参数 (`models_*/benchmark_*.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_name` | 数据集名称 | `saferlhf` |
| `--data_root` | 数据目录 | `./embeddings/biased_pu` |
| `--model_name` | 模型名称 | `FsfairX-LLaMA3-RM-v0.1` |
| `--alpha` | 偏差强度 | `0.5` |
| `--output_dir` | 结果输出目录 | `./results/cache/{method}/{data_name}` |

### 去偏方法一览

| 方法 | 类型 | 说明 |
|------|------|------|
| Naive | Baseline | 直接训练，不做去偏 |
| IPS | 偏差校正 | 逆倾向性加权 |
| DR | 偏差校正 | 双重稳健估计 |
| MT-IPS | 偏差校正 | 多任务 IPS |
| MT-DR | 偏差校正 | 多任务 DR |
| SDR | 偏差校正 | 自去偏 |
| Counter-IF | 偏差+PU | 分组插补 + pointwise/pairwise + IPM(Wasserstein) |
| ReCRec-F | 偏差+PU | E-M 推断曝光/偏好：P(y=1)=mu(x)*gamma(x) |
| Co-Teaching | PU/噪声校正 | 双网络互教 |
| DivideMix | PU/噪声校正 | 混合样本学习 |
| CVIB | PU/噪声校正 | 变分信息瓶颈 |
| LabelWave | PU/噪声校正 | 标签波动检测 |

---

## 快速开始

```bash
# 1. 准备原始数据
# 将 HuggingFace datasets 放入 ./rawdata/ 目录

# 2. 生成 Embedding
bash data_prepare.sh

# 3. 模拟偏差和 PU Learning
bash simulate_bias_pu.sh

# 4. 运行 benchmark
python models_debias/benchmark_naive.py --data_name ufb --alpha 0.1
python models_debias/benchmark_ips.py --data_name ufb --alpha 0.1
python models_debias/benchmark_dr.py --data_name ufb --alpha 0.1

# 5. 分析结果
jupyter notebook analyze/causal_rm.ipynb
```
