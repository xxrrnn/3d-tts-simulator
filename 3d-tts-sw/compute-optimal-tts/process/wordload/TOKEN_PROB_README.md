# Workload Generator - Token Probability Support

## 新功能: 可选包含 Token 概率历史

现在支持在生成workload时包含 `token_prob_history`，用于后续计算注意力熵等指标。

## 使用方法

### 基础用法（不包含token概率）

```bash
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload

# 生成标准workload（输出较小）
python gen_workload.py --input ../../src/output/AIME24_beam_search
```

### 包含Token概率

```bash
# 生成包含token概率的workload（输出较大，用于注意力熵计算）
python gen_workload.py --input ../../src/output/AIME24_beam_search --include-token-probs
```

### 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--input` | 是 | 输入目录路径 |
| `--dataset` | 否 | 数据集名称（默认使用输入目录名） |
| `--include-token-probs` | 否 | 包含token_prob_history |
| `--verbose` / `-v` | 否 | 详细输出 |

### 批量处理

```bash
# 标准模式
for dataset in AMC23_beam_search AIME24_beam_search MATH_beam_search; do
    python gen_workload.py --input ../../src/output/$dataset
done

# 包含token概率
for dataset in AMC23_beam_search AIME24_beam_search MATH_beam_search; do
    python gen_workload.py --input ../../src/output/$dataset --include-token-probs
done
```

## 输出格式

### 标准格式（不包含token概率）

```json
{
  "question_id": "question_0",
  "prefill": {
    "kv_cache_count": 563
  },
  "decode": {
    "steps": [
      {
        "step": 0,
        "branch_count": 2,
        "branch_tokens": [53, 32],
        "branch_rewards": [0.882, 0.854],
        "selected_branch_index": 0
      }
    ]
  }
}
```

### 包含Token概率的格式

```json
{
  "question_id": "question_0",
  "prefill": {
    "kv_cache_count": 563
  },
  "decode": {
    "steps": [
      {
        "step": 0,
        "branch_count": 4,
        "branch_tokens": [43, 45, 48, 46],
        "branch_rewards": [0.882, 0.854, 0.870, 0.865],
        "branch_token_probs": [
          [0.95, 0.98, 0.92, 0.89, ..., 0.94]
        ],
        "selected_branch_index": 1
      },
      {
        "step": 1,
        "branch_count": 4,
        "branch_tokens": [336, 77, 57, 61],
        "branch_rewards": [0.901, 0.843, 0.856, 0.862],
        "branch_token_probs": [
          [0.93, 0.97, 0.88, ..., 0.96]
        ],
        "selected_branch_index": 0
      }
    ]
  }
}
```

**注意**：
- `branch_count=4` 但 `branch_token_probs` 只有1个元素（实际生成的分支）
- `branch_token_probs[0]` 的长度 == `branch_tokens[selected_branch_index]`
- Step 0: `len([0.95, ...]) == 45` (匹配 `branch_tokens[1]`)
- Step 1: `len([0.93, ...]) == 336` (匹配 `branch_tokens[0]`)

## Token概率数据结构

`branch_token_probs` 是一个列表，**只包含当前step实际生成分支的token概率**：

```python
branch_token_probs[i]  # 第i个实际生成分支在当前step的token概率列表
branch_token_probs[i][j]  # 第i个分支的第j个token的概率
```

**重要说明**：
- `branch_count` 表示beam search的候选分支总数（如4个）
- `branch_token_probs` **只包含实际生成的分支**在**当前step**的概率
- 通常 `len(branch_token_probs) <= branch_count`
- **不包含空的list**，只记录有完整概率数据的分支
- 未被选中继续生成的分支不会有概率数据
- **每个step独立**，不包含历史数据

**示例**：
```json
{
  "step": 0,
  "branch_count": 4,  // beam search有4个候选
  "branch_tokens": [43, 45, 48, 46],
  "selected_branch_index": 1,
  "branch_token_probs": [  // 只有1个实际生成的分支
    [0.92, 0.65, 0.85, ...]  // 该分支在step 0的45个token的概率
  ]
}
```

**长度验证**：
```python
len(branch_token_probs[i]) == branch_tokens[selected_branch_index]
```

## 使用场景

### 1. 标准workload（推荐）

适用于：
- 基础调度模拟
- Token计数分析
- Reward分析
- Straggler识别

优点：
- 文件小
- 处理快

### 2. 包含Token概率

适用于：
- 注意力熵计算
- Token概率分布分析
- 不确定性分析
- 高级模型训练

缺点：
- 文件较大（约7倍，4KB → 28KB）
- 处理稍慢

**文件大小对比**（以question_0为例）：
```
标准workload:     4KB
包含token_probs: 28KB (约7倍)
```

## 文件大小对比

基于实际测试：

| 数据集 | 标准模式 | 包含Token概率 | 比例 |
|--------|----------|---------------|------|
| 单个question | ~2 KB | ~20-200 KB | 10-100x |
| 30题 | ~60 KB | ~0.6-6 MB | 10-100x |

**建议**：
- 如果只需要基础调度信息，使用标准模式
- 如果需要计算注意力熵，使用 `--include-token-probs`

## 注意力熵计算示例

```python
import json
import numpy as np

# 加载包含token概率的workload
with open('question_0_workload.json', 'r') as f:
    workload = json.load(f)

# 检查是否包含token概率
if workload.get('metadata', {}).get('includes_token_probs'):
    for step in workload['decode']['steps']:
        branch_token_probs = step.get('branch_token_probs', [])
        
        for i, token_probs in enumerate(branch_token_probs):
            # 计算每步的熵
            entropies = []
            for step_probs in token_probs:
                # H = -Σ p*log(p)
                probs = np.array(step_probs)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
            
            avg_entropy = np.mean(entropies)
            print(f"Step {step['step']}, Branch {i}: avg entropy = {avg_entropy:.4f}")
else:
    print("This workload does not include token probabilities")
```

## 数据提取

从原始 `record_0.jsonl` 中提取token概率：

```python
# 数据来源
data['output'][i]['token_prob_history']  # 第i个branch的token概率历史
```

提取逻辑在 `extract_decode_steps()` 函数中：

```python
if include_token_probs:
    if i < len(data['output']):
        output_item = data['output'][i]
        if 'token_prob_history' in output_item:
            token_probs = output_item['token_prob_history']
    branch_token_probs.append(token_probs)
```

## 故障排除

### 问题：输出文件太大

**解决方案**：
1. 不使用 `--include-token-probs` 标志
2. 或者只对需要分析的部分数据集使用

### 问题：Token概率为空

**原因**：原始 `record_0.jsonl` 中没有 `token_prob_history` 字段

**解决方案**：
1. 检查数据生成时是否保存了token概率
2. 确认vLLM配置是否启用了token概率记录

---

**更新时间**: 2026-04-01  
**版本**: v1.1  
**新增功能**: Token概率支持
