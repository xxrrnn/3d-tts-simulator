<<<<<<< HEAD
# Straggler识别与注意力熵分析工具

## 概述

本工具集用于识别beam search中的straggler branch，并分析注意力熵是否能够提前指示这些branch是否会被选中。

## Straggler定义

一个branch被认为是straggler，当满足以下所有条件：
1. Token数量 > 80
2. Token数量 > 其他所有branch最大值的2倍
3. 该step有至少2个候选branch

## 工具清单

### 1. `identify_stragglers.py` - 主分析脚本

识别所有straggler并计算它们的注意力熵指标。

**用法**:
```bash
python identify_stragglers.py [--input DIR] [--output FILE] [--report FILE] [--verbose]
```

**参数**:
- `--input`: Workload目录路径（默认: `../wordload/model_workloads/16384_4_1`）
- `--output`: 输出JSON文件（默认: `straggler_analysis.json`）
- `--report`: 输出文本报告（默认: `straggler_report.txt`）
- `--verbose`: 显示详细调试信息

**输出**:
- `straggler_analysis.json`: 包含所有straggler的详细信息（JSON格式）
- `straggler_report.txt`: 易读的文本报告

**示例**:
```bash
# 使用默认设置
python identify_stragglers.py --verbose

# 指定输入目录
python identify_stragglers.py --input /path/to/workloads --verbose
```

### 2. `analyze_straggler_entropy.py` - 深度熵分析

分析注意力熵与straggler选择之间的关系（包含reward）。

**用法**:
```bash
python analyze_straggler_entropy.py
```

**前提条件**: 必须先运行 `identify_stragglers.py` 生成 `straggler_analysis.json`

**输出**:
- 控制台输出: 详细的统计分析
- `straggler_entropy_case_study.txt`: 最高/最低熵的案例研究

### 3. `analyze_straggler_runtime.py` - Runtime特征分析 ⭐

**分析runtime时可用的特征**（不使用reward），用于实际剪枝决策。

**用法**:
```bash
python analyze_straggler_runtime.py
```

**前提条件**: 必须先运行 `identify_stragglers.py` 生成 `straggler_analysis.json`

**输出**:
- 控制台输出: Runtime特征重要性排名
- 推荐的剪枝阈值和策略
- 性能评估（精确率/召回率）

**关键发现**:
- Token长度: 59.6%差异（被选中的更短）
- 高置信度比例: 61.4%差异（被选中的更少）
- 推荐保守策略: 83.3%精确率，41.7%召回率

## 快速开始

```bash
# 1. 进入threshold目录
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold

# 2. 运行straggler识别
python identify_stragglers.py --verbose

# 3. 运行深度熵分析（包含reward）
python analyze_straggler_entropy.py

# 4. 运行runtime特征分析（不使用reward）⭐ 实际应用
python analyze_straggler_runtime.py

# 5. 查看结果
cat STRAGGLER_ANALYSIS_SUMMARY.md      # 完整分析总结
cat RUNTIME_PRUNING_STRATEGY.md        # Runtime剪枝策略（重要！）
```

## 注意力熵指标说明

### 基本指标

- **mean_entropy**: 平均token熵
  - 计算: `-sum(p * log(p))` for each token
  - 含义: 值越大表示模型越不确定
  
- **std_entropy**: 熵的标准差
  - 含义: 值越大表示不同token之间的确定性差异越大
  
- **mean_prob**: 平均token概率
  - 含义: 值越大表示模型整体越自信
  
- **min_prob**: 最小token概率
  - 含义: 最不确定的token的概率

### 派生指标

- **high_confidence_tokens**: 概率 > 0.9 的token数量
- **low_confidence_tokens**: 概率 < 0.5 的token数量
- **total_entropy**: 所有token熵的总和

## 输出文件结构

### straggler_analysis.json
=======
# Straggler Branch 识别工具

识别符合straggler定义的branch，并生成完整的分析报告。

## Straggler定义

一个branch被认为是straggler，需要同时满足以下条件：

1. **Token数量超过80** - `branch_tokens > 80`
2. **超过其他branch最大值的2倍** - `branch_tokens > max(other_branches) * 2`
3. **Branch数量至少为2** - `len(branches) >= 2`

## 使用方法

### 基本用法

```bash
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold

# 使用默认路径
python3 identify_stragglers.py

# 指定输入目录
python3 identify_stragglers.py --input ../wordload/model_workloads_need

# 详细输出
python3 identify_stragglers.py --verbose

# 自定义输出文件
python3 identify_stragglers.py --output my_analysis.json --report my_report.txt
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `../wordload/model_workloads_need` | Workload文件目录 |
| `--output` | `straggler_analysis.json` | JSON输出文件 |
| `--report` | `straggler_report.txt` | 文本报告文件 |
| `--verbose` | False | 详细日志输出 |

## 输出文件

### 1. JSON文件 (`straggler_analysis.json`)

完整的结构化数据，包含每个straggler的详细信息：
>>>>>>> 207a1d0 (A6000 0401)

```json
{
  "summary": {
<<<<<<< HEAD
    "total_workload_files": 30,
    "files_with_stragglers": 15,
    "total_straggler_branches": 23,
    "stragglers_with_token_probs": 23
  },
  "stragglers": [
    {
      "source_file": "...",
      "question_id": "question_0",
      "step": 4,
      "straggler_branch_index": 1,
      "straggler_tokens": 141,
      "straggler_reward": 0.375212,
      "is_selected": false,
      "has_token_probs": true,
      "entropy_metrics": {
        "mean_entropy": 0.0118,
        "std_entropy": 0.0494,
        "mean_prob": 0.9776,
        "min_prob": 0.1048,
        "high_confidence_tokens": 135,
        "low_confidence_tokens": 3
      },
      "step_info": { ... },
      "stats": { ... }
=======
    "total_workload_files": 1957,
    "files_with_stragglers": 638,
    "total_straggler_branches": 1010
  },
  "stragglers": [
    {
      "source_file": "path/to/workload.json",
      "question_id": "question_26",
      "step": 1,
      "straggler_branch_index": 0,
      "straggler_tokens": 260,
      "straggler_reward": 0.1878,
      "is_selected": true,
      "step_info": {
        "step": 1,
        "branch_count": 2,
        "branch_tokens": [260, 63],
        "branch_rewards": [0.1878, 0.1549],
        "selected_branch_index": 0
      },
      "stats": {
        "max_other_tokens": 63,
        "ratio_to_max_other": 4.13,
        "total_branches": 2
      }
>>>>>>> 207a1d0 (A6000 0401)
    }
  ]
}
```

<<<<<<< HEAD
## 关键发现

基于`16384_4_1`数据集的分析：

### 1. Runtime可用特征的预测能力 ⭐

**不使用reward时的最强指标**：

| 特征 | 差异度 | 方向 | 可用性 |
|------|--------|------|--------|
| **Token长度** | 59.6% | 被选中的更短 (281 vs 695) | ✓ 总是可用 |
| **高置信度比例** | 61.4% | 被选中的更少 (33.7% vs 87.2%) | ✓ 如果有token_probs |
| **最小概率** | 172.3% | 被选中的更高 (0.23 vs 0.08) | ✓ 如果有token_probs |
| **长度比例** | 53.0% | 被选中的更小 (3.2x vs 6.8x) | ✓ 总是可用 |

**推荐的Runtime剪枝策略**:
```python
# 保守策略（精确率83.3%，误剪率9.1%）
if tokens > 274 and ratio > 3.6:
    prune()  # 能正确剪枝41.7%的不会被选中的straggler
```

详细信息请查看: **`RUNTIME_PRUNING_STRATEGY.md`**

### 2. 注意力熵的预测能力（需要reward验证）

- **熵差异**: 被选中的straggler比未被选中的高14.3%
- **结论**: 注意力熵提供**弱信号**，不应单独使用

### 3. Reward是最强预测指标（但runtime不可用）

- **Reward差异**: 被选中的straggler比未被选中的高**149.9%**
- **结论**: Reward是主要决定因素，但在runtime时还没有计算


### 3. 反直觉发现

被选中的straggler特征：
- ✗ 更高的熵（更不确定）
- ✗ 更少的高置信度token（33.7% vs 87.2%）
- ✓ 更高的reward（0.24 vs 0.10）
- ✓ 更短的长度（281 vs 695 tokens）

**解释**: 未被选中的超长straggler虽然高置信度，但reward极低，说明模型虽然"自信"但实际质量差。

### 4. 剪枝策略建议

**应该使用的指标**:
- ✓ Reward score (最强指标)
- ✓ Token长度
- ✓ Token长度比例

**建议阈值**:
- `Reward < 0.1 AND tokens > 500` → 高概率不会被选中
- `Reward < 0.05 AND ratio > 5x` → 强烈建议剪枝

## 扩展到其他数据集

```bash
# 分析其他配置
python identify_stragglers.py --input ../wordload/model_workloads/16384_8_2 --verbose
python analyze_straggler_entropy.py
```

## 依赖

- Python 3.6+
- numpy
- json
- pathlib

安装依赖：
```bash
pip install numpy
```

## 相关文档

- `RUNTIME_PRUNING_STRATEGY.md`: **Runtime剪枝策略（不使用reward）** ⭐ 重要
- `STRAGGLER_ANALYSIS_SUMMARY.md`: 完整分析结果总结（包含reward分析）
- `straggler_report.txt`: 完整的文本报告
- `straggler_entropy_case_study.txt`: 案例研究
- `straggler_analysis.json`: 原始数据（JSON格式）

## 常见问题

### Q1: 为什么有些straggler没有entropy数据？

A: 只有被选中继续生成的branch才有完整的token_prob_history。如果workload是用`--include-token-probs`生成的，所有被选中的straggler都应该有entropy数据。

### Q2: 熵值的正常范围是多少？

A: 
- 高置信度: entropy < 0.01 (prob > 0.99)
- 中等置信度: entropy 0.01-0.1
- 低置信度: entropy > 0.1 (prob < 0.9)

### Q3: 如何解读"被选中的straggler有更高的熵"？

A: 这是一个反直觉但重要的发现。它说明：
- 模型的"自信"（低熵）不等于"正确"（高reward）
- 未被选中的超长straggler虽然自信但质量差
- Reward比熵更能反映实际质量

## 作者与维护

位置: `/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold/`

最后更新: 2026-04-01
=======
### 2. 文本报告 (`straggler_report.txt`)

易读的摘要报告，按题目分组：

```
================================================================================
STRAGGLER BRANCH ANALYSIS REPORT
================================================================================

SUMMARY
--------------------------------------------------------------------------------
  total_workload_files: 1957
  files_with_stragglers: 638
  total_straggler_branches: 1010

STRAGGLERS BY QUESTION (40 questions)
--------------------------------------------------------------------------------

question_0 - 12 straggler(s)
  Source: ../wordload/.../question_0_workload.json

  Step 0:
    Straggler branch: 0 (not selected)
    Tokens: 270 (ratio: 3.60x)
    Reward: 0.990256
    All branches (3):
      Branch 0: 270 tokens, reward=0.990256 <-- STRAGGLER
      Branch 1: 75 tokens, reward=0.996802 [SELECTED]
      Branch 2: 74 tokens, reward=0.995485
```

## 分析结果

基于当前数据集的分析：

- **总workload文件**: 1,957个
- **包含straggler的文件**: 638个 (32.6%)
- **Straggler branch总数**: 1,010个

### Straggler统计

| 指标 | 最小值 | 最大值 | 平均值 |
|------|--------|--------|--------|
| Token数量 | 81 | 5,224 | 357.2 |
| 与其他branch最大值的比例 | 2.00x | 98.60x | 4.97x |
| 被选为最终答案的比例 | - | - | 35.5% |

### 关键发现

1. **35.5%的straggler被选为最终答案** - 说明长的branch不一定是错的
2. **平均比例4.97x** - straggler通常是其他branch的5倍长
3. **最大比例98.6x** - 存在极端情况

## 数据结构

### JSON字段说明

#### Summary部分
- `total_workload_files` - 分析的workload文件总数
- `files_with_stragglers` - 包含straggler的文件数
- `total_straggler_branches` - straggler branch总数

#### Stragglers部分（每个straggler）

**位置信息**:
- `source_file` - 源workload文件路径
- `question_id` - 题目ID
- `step` - 步骤编号

**Straggler信息**:
- `straggler_branch_index` - straggler的branch索引
- `straggler_tokens` - token数量
- `straggler_reward` - reward得分
- `is_selected` - 是否被选为最终答案

**完整步骤信息**:
- `step_info.branch_count` - 该步骤的branch总数
- `step_info.branch_tokens` - 所有branch的token数量
- `step_info.branch_rewards` - 所有branch的reward
- `step_info.selected_branch_index` - 被选中的branch索引

**统计信息**:
- `stats.max_other_tokens` - 其他branch的最大token数
- `stats.ratio_to_max_other` - 与其他branch最大值的比例
- `stats.total_branches` - 该步骤的总branch数

## 进一步分析

### 使用Python分析

```python
import json

with open('straggler_analysis.json', 'r') as f:
    data = json.load(f)

# 找出所有被选中的straggler
selected_stragglers = [s for s in data['stragglers'] if s['is_selected']]
print(f"Selected stragglers: {len(selected_stragglers)}")

# 找出最长的straggler
longest = max(data['stragglers'], key=lambda x: x['straggler_tokens'])
print(f"Longest: {longest['straggler_tokens']} tokens in {longest['question_id']}")

# 按比例排序
by_ratio = sorted(data['stragglers'], key=lambda x: x['stats']['ratio_to_max_other'], reverse=True)
print(f"Highest ratio: {by_ratio[0]['stats']['ratio_to_max_other']:.2f}x")
```

### 按数据集统计

```python
from collections import defaultdict

dataset_stats = defaultdict(int)
for s in data['stragglers']:
    dataset = s['source_file'].split('/')[1]  # 提取数据集名
    dataset_stats[dataset] += 1

for dataset, count in sorted(dataset_stats.items()):
    print(f"{dataset}: {count} stragglers")
```

## 集成到其他工具

### 1. 与动态剪枝器集成

可以使用straggler数据来训练或调整剪枝策略：

```python
from threshold.dynamic_pruner import DynamicPruner

# 加载straggler数据
with open('straggler_analysis.json', 'r') as f:
    stragglers = json.load(f)['stragglers']

# 分析straggler的特征
straggler_rewards = [s['straggler_reward'] for s in stragglers]
print(f"Straggler reward mean: {np.mean(straggler_rewards):.4f}")
```

### 2. 生成训练数据

为straggler预测模型生成训练数据：

```python
training_data = []
for s in stragglers:
    training_data.append({
        'features': {
            'tokens': s['straggler_tokens'],
            'reward': s['straggler_reward'],
            'ratio': s['stats']['ratio_to_max_other'],
            'num_branches': s['stats']['total_branches']
        },
        'label': 'straggler',
        'is_correct': s['is_selected']
    })
```

## 相关文件

- `identify_stragglers.py` - 主程序
- `straggler_analysis.json` - JSON输出
- `straggler_report.txt` - 文本报告
- `README.md` - 本文档

---

**创建时间**: 2026-04-01  
**版本**: v1.0  
**作者**: Assistant
>>>>>>> 207a1d0 (A6000 0401)
