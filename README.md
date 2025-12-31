# Causal Reward Model

基于因果推断的奖励模型研究框架，用于研究选择偏差(Selection Bias)和标签噪声(Label Noise)对奖励模型的影响，以及各种去偏方法的效果。

## 项目结构

```
causal-rm/
├── embeddings/                   # Embedding 存储目录
│   ├── normal/                   # Stage 1 输出 (原始 embedding)
│   └── biased_noisy/             # Stage 2 输出 (含偏差/噪声)
│
├── rawdata/                      # 原始数据集目录 (HuggingFace datasets 格式)
│   ├── hs/                       # HelpSteer 数据集
│   ├── ufb/                      # UltraFeedback 数据集
│   └── saferlhf/                 # SaferLHF 数据集
│
├── prepare.py                    # Stage 1: Embedding 生成
├── data_prepare.sh               # Stage 1 批处理脚本
│
├── simulate_bias_noisy.py        # Stage 2: 偏差/噪声模拟
├── simulate_bias_noisy.sh        # Stage 2 批处理脚本
│
├── benchmark_*.py                # Stage 3: 各种去偏方法的 benchmark
│   ├── benchmark_naive.py        #   - Naive (无去偏)
│   ├── benchmark_ips.py          #   - IPS (Inverse Propensity Scoring)
│   ├── benchmark_dr.py           #   - DR (Doubly Robust)
│   ├── benchmark_mtips.py        #   - MT-IPS (Multi-Task IPS)
│   ├── benchmark_mtdr.py         #   - MT-DR (Multi-Task DR)
│   ├── benchmark_sdr.py          #   - SDR (Self-Debiasing)
│   ├── benchmark_ome_ips.py      #   - OME-IPS
│   ├── benchmark_ome_dr.py       #   - OME-DR
│   ├── benchmark_co_teaching.py  #   - Co-Teaching (噪声标签)
│   ├── benchmark_codis.py        #   - CoDis
│   ├── benchmark_cvib.py         #   - CVIB
│   ├── benchmark_labelwave.py    #   - LabelWave
│   ├── benchmark_kmeidtm.py      #   - KME-IDTM
│   ├── benchmark_eps_softmax.py  #   - EPS-Softmax
│   └── benchmark_robust_dividemix.py  # - DivideMix
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
           ▼  Stage 2: simulate_bias_noisy.sh → simulate_bias_noisy.py
           │
┌──────────────────────────────────────────────────────┐
│  {model_name}_{data_name}_{alpha}_{r10}_{r01}.safetensors  │  Biased + Noisy
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
}
```

### 支持的数据集

| 数据集 | 标签类型 | 标签范围 |
|--------|----------|----------|
| `hs` | helpfulness | 0-4 |
| `ufb` | score | 1-10 |
| `saferlhf` | severity_level | 0-3 |
| `armorm` | 多属性 (19维) | 各不相同 |

---

## Stage 2: 偏差/噪声模拟

在 embedding 数据上模拟选择偏差和标签噪声。

### 脚本

```bash
bash simulate_bias_noisy.sh
```

### 参数 (`simulate_bias_noisy.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name` | 模型名称 | `FsfairX-LLaMA3-RM-v0.1` |
| `--data_name` | 数据集名称 | `saferlhf` |
| `--data_root` | embedding 输入目录 | `./embeddings/normal` |
| `--output_dir` | 输出目录 | `./embeddings/biased_noisy` |
| `--alpha` | 选择偏差强度 (越小偏差越强) | `0.5` |
| `--r10` | 正→负 翻转概率 | `0.1` |
| `--r01` | 负→正 翻转概率 | `0.1` |

### 输入

```
{data_root}/{model_name}_{data_name}_train.safetensors
{data_root}/{model_name}_{data_name}_test.safetensors
```

### 输出

```
{output_dir}/{model_name}_{data_name}_{alpha}_{r10}_{r01}.safetensors

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

    # 二值标签 (含噪声)
    "y_train_binary": Tensor[N_train],  # 0/1, 在 mask=True 处有噪声
    "y_val_binary": Tensor[N_val],
    "y_test_binary": Tensor[N_test],    # 无噪声 (用于评估)

    # 倾向性分数 (用于 IPS 等去偏方法)
    "propensity_train": Tensor[N_train],
    "propensity_val": Tensor[N_val],

    # 观察掩码 (True = 被观察到)
    "mask_train": Tensor[N_train],
    "mask_val": Tensor[N_val],
}
```

### 偏差模拟原理

- **选择偏差**: 低奖励样本的 propensity score 更低，更不容易被"观察到"
  - `propensity = alpha^(max_reward - reward)`
  - `alpha=0.5` 时偏差较弱，`alpha=0.01` 时偏差很强

- **标签噪声**: 对被观察到的样本按概率翻转二值标签
  - `r10`: P(label 1→0)
  - `r01`: P(label 0→1)

---

## Stage 3: 模型训练与评估

使用各种去偏方法训练奖励模型并评估。

### 示例

```bash
# Naive baseline (无去偏)
python benchmark_naive.py --data_name saferlhf --alpha 0.1 --r10 0.2 --r01 0.1

# IPS (Inverse Propensity Scoring)
python benchmark_ips.py --data_name saferlhf --alpha 0.1 --r10 0.2 --r01 0.1

# Doubly Robust
python benchmark_dr.py --data_name saferlhf --alpha 0.1 --r10 0.2 --r01 0.1
```

### 通用参数 (`benchmark_*.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_name` | 数据集名称 | `saferlhf` |
| `--data_root` | 数据目录 | `./embeddings/biased_noisy` |
| `--model_name` | 模型名称 | `FsfairX-LLaMA3-RM-v0.1` |
| `--alpha` | 偏差强度 | `0.5` |
| `--r10` | 噪声率 (1→0) | `0.1` |
| `--r01` | 噪声率 (0→1) | `0.1` |
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
| Co-Teaching | 噪声校正 | 双网络互教 |
| DivideMix | 噪声校正 | 混合样本学习 |
| CVIB | 噪声校正 | 变分信息瓶颈 |
| LabelWave | 噪声校正 | 标签波动检测 |

---

## 快速开始

```bash
# 1. 准备原始数据
# 将 HuggingFace datasets 放入 ./rawdata/ 目录

# 2. 生成 Embedding
bash data_prepare.sh

# 3. 模拟偏差和噪声
bash simulate_bias_noisy.sh

# 4. 运行 benchmark
python benchmark_naive.py --data_name ufb --alpha 0.1
python benchmark_ips.py --data_name ufb --alpha 0.1
python benchmark_dr.py --data_name ufb --alpha 0.1

# 5. 分析结果
jupyter notebook analyze/causal_rm.ipynb
```
