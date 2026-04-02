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

```json
{
  "summary": {
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
    }
  ]
}
```

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
