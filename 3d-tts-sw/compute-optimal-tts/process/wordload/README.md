# Workload 生成器说明

本目录包含两个工作负载生成器，用于从beam search结果中提取workload信息。

## 文件说明

### 1. `gen_workload.py` - 标准工作负载生成器

**目标**: 从beam search结果生成标准workload格式

**输出内容**:
- Prefill的KV cache数量
- 每个decode step的分支数量、token数量、reward得分
- 被选中的分支索引

**输出格式**:
```json
{
  "question_id": "question_X",
  "prefill": {
    "kv_cache_count": int
  },
  "decode": {
    "steps": [
      {
        "step": int,
        "branch_count": int,
        "branch_tokens": [int, ...],
        "branch_rewards": [float, ...],
        "selected_branch_index": int
      }
    ]
  }
}
```

**输出大小**: 小（每个问题 1-2 KB）

**输出目录**: `/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sim/model_workloads/`

### 2. `gen_workload_beam.py` - Beam Search配置增强版

**目标**: 在标准workload基础上增加beam search特有的配置信息

**输出内容**:
- 所有`gen_workload.py`的内容
- **额外增加**:
  - Beam search配置（beam_size, max_step）
  - 每步的terminated_count（终止分支数）
  - **selected_branch_indices（复数）**：记录所有被选中的分支索引列表

**重要说明**:
- 虽然`beam_size=2`，但每一步**可能选中2、4或更多个分支**
- 这是因为beam search从多个父节点同时扩展
- 例如：Step 0选中2个分支 → Step 1这2个分支各自扩展 → 可能产生4个被选中的子分支

**输出格式**:
```json
{
  "question_id": "question_X",
  "prefill": {
    "kv_cache_count": int
  },
  "beam_search_config": {
    "beam_size": int,
    "max_step": int
  },
  "decode_steps": [
    {
      "step": int,
      "branch_count": int,
      "branch_tokens": [int, ...],
      "branch_rewards": [float, ...],
      "branch_parent_indices": [int, ...],  // 新增：父分支索引，-1表示根节点
      "selected_branch_indices": [int, ...],  // 注意：复数，可能有多个
      "terminated_count": int
    }
  ]
}
```

**字段说明**:
- `branch_parent_indices`: 每个分支的父分支索引（指向上一步的分支数组索引）
  - `-1`: 表示根节点（仅在Step 0）
  - `>= 0`: 表示来自上一步的第几个分支（基于上一步的branch数组索引）
  - 用于追踪beam search的树形结构
  - **示例**：如果Step 1的分支3的parent=5，表示该分支来自Step 0的分支5

**树结构追踪示例**：
```
Step 0: selected=[2, 5]
  ├─ 分支2 (reward=0.062)
  └─ 分支5 (reward=0.053)

Step 1: parent_indices=[2, 2, 5, 5, 5]
  ├─ 分支0 (parent=2) ← 来自Step0分支2
  ├─ 分支1 (parent=2) ← 来自Step0分支2
  ├─ 分支2 (parent=5) ← 来自Step0分支5
  ├─ 分支3 (parent=5) ← 来自Step0分支5
  └─ 分支4 (parent=5) ← 来自Step0分支5
```

**输出大小**: 小（每个问题 5-10 KB，取决于步数）

**输出目录**: `/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sim/model_workloads_beam/`

## 使用方法

### 标准版本（gen_workload.py）

```bash
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload

# 单个数据集
python gen_workload.py --input ../../src/output/MATH_beam_search --verbose

# 批量处理
for dataset in AMC23_beam_search AIME24_beam_search MATH_beam_search; do
    python gen_workload.py --input ../../src/output/$dataset --verbose
done
```

### Beam增强版本（gen_workload_beam.py）

```bash
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload

# 单个数据集
python gen_workload_beam.py --input ../../src/output/AIME24_beam_search --verbose

# 批量处理
for dataset in AMC23_beam_search AIME24_beam_search MATH_beam_search; do
    python gen_workload_beam.py --input ../../src/output/$dataset --verbose
done
```

## 注意事项

1. **详细日志依赖**: 两个工具都需要record文件中包含`detailed_beam_search_log`字段。如果没有这个字段，文件会被跳过。

2. **格式兼容**: `gen_workload_beam.py`的输出完全兼容`gen_workload.py`，只是额外增加了beam search配置信息。

3. **多output支持**: 两个工具都能正确处理`detailed_beam_search_log`在`output[0]`或`output[1]`中的情况。

## 选择哪个工具？

- **只需要基本workload信息**: 使用`gen_workload.py`
- **需要beam search配置信息（beam_size等）**: 使用`gen_workload_beam.py`
- **需要分析beam search的探索策略**: 使用`gen_workload_beam.py`（包含terminated_count等信息）

## 关键区别

| 特性 | gen_workload.py | gen_workload_beam.py |
|------|----------------|----------------------|
| prefill信息 | ✅ | ✅ |
| decode步骤统计 | ✅ | ✅ |
| branch tokens | ✅ | ✅ |
| branch rewards | ✅ | ✅ |
| selected_branch_index | ✅ | ❌ |
| **selected_branch_indices** | ❌ | ✅ (支持多个) |
| **branch_parent_indices** | ❌ | ✅ (追踪树结构) |
| **beam_search_config** | ❌ | ✅ |
| **terminated_count** | ❌ | ✅ |
| 文件大小 | 1-2 KB | 5-10 KB |

**注意**: 
- 两个工具都**不包含**分支的文本内容，只包含统计信息。这就是workload的定义！
- `gen_workload_beam.py`使用`selected_branch_indices`（复数）来记录每步可能有**多个**被选中的分支
