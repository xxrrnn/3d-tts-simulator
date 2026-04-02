# AIME24 Beam Search Workload 生成完成报告

## 生成时间
2026-03-26

## 数据集
AIME24 - Qwen2.5-Math-1.5B-Instruct + Skywork-o1-Open-PRM-Qwen-2.5-1.5B

## 生成统计

### 配置覆盖

| 配置 | 格式 | 问题数 | 状态 | 文件大小 |
|------|------|--------|------|----------|
| 16384_2_1 | width=2, beam=1 | 0/30 | ✗ 失败 | 0 KB |
| 16384_2_2 | width=2, beam=2 | 30/30 | ✓ 成功 | 151 KB |
| 16384_4_1 | width=4, beam=1 | 30/30 | ✓ 成功 | 137 KB |
| 16384_4_2 | width=4, beam=2 | 30/30 | ✓ 成功 | 127 KB |
| 16384_8_1 | width=8, beam=1 | 30/30 | ✓ 成功 | 125 KB |
| 16384_8_2 | width=8, beam=2 | 30/30 | ✓ 成功 | 166 KB |
| 16384_8_4 | width=8, beam=4 | 30/30 | ✓ 成功 | 177 KB |

**总计**: 180 个workload文件 (6/7配置成功), 0.86 MB

### 失败原因

- `16384_2_1`: 原始record文件为空或JSON格式错误，无法解析

## Workload格式

每个workload文件包含以下关键字段：

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
      "branch_parent_indices": [int, ...],  // 关键：父分支索引
      "selected_branch_indices": [int, ...],
      "terminated_count": int
    }
  ]
}
```

### `branch_parent_indices` 说明

- **功能**: 记录每个分支的父分支索引，用于追踪beam search树结构
- **值含义**:
  - `-1`: 根节点（仅Step 0）
  - `>= 0`: 来自上一步的分支索引
- **示例**: `[2, 2, 5, 5, 5]` 表示前2个分支来自上一步的分支2，后3个来自分支5

## 验证结果

### 随机抽样验证（5个文件）

所有抽样验证通过：
- ✓ Step 0 所有parent都是-1
- ✓ Step 1+ 所有parent都在有效范围内
- ✓ 可以正确回溯完整路径
- ✓ 树结构正确

### 功能验证

使用 `verify_parent_indices.py` 脚本验证：
```bash
cd compute-optimal-tts/process/wordload
python3 verify_parent_indices.py output_beam/AIME24/16384_8_2/question_17/workload_0.json
```

验证项目：
- ✅ parent_indices范围检查
- ✅ 路径回溯功能
- ✅ 树结构可视化
- ✅ 扩展率统计

## 输出位置

```
compute-optimal-tts/process/wordload/output_beam/AIME24/
├── 16384_2_2/
│   ├── question_0/workload_0.json
│   ├── question_1/workload_0.json
│   └── ... (30个问题)
├── 16384_4_1/
│   └── ... (30个问题)
├── 16384_4_2/
│   └── ... (30个问题)
├── 16384_8_1/
│   └── ... (30个问题)
├── 16384_8_2/
│   └── ... (30个问题)
└── 16384_8_4/
    └── ... (30个问题)
```

## 使用指南

### 1. 加载workload

```python
import json

with open('workload_0.json', 'r') as f:
    workload = json.load(f)

beam_size = workload['beam_search_config']['beam_size']
steps = workload['decode_steps']
```

### 2. 追踪路径

```python
def trace_path(workload, target_step, target_branch):
    """回溯完整路径"""
    path = [(target_step, target_branch)]
    current_step, current_branch = target_step, target_branch
    
    while current_step > 0:
        parent = workload['decode_steps'][current_step]['branch_parent_indices'][current_branch]
        if parent == -1:
            break
        current_step -= 1
        current_branch = parent
        path.insert(0, (current_step, current_branch))
    
    return path
```

### 3. 分析实际扩展的分支

```python
# 查看下一步的parent_indices确定哪些分支被扩展
step1 = workload['decode_steps'][1]
step2 = workload['decode_steps'][2]

# Step 1中实际被扩展的分支
expanded_branches = set(step2['branch_parent_indices'])
print(f"Step 1 实际扩展的分支: {sorted(expanded_branches)}")

# 对比selected标记的分支
selected_branches = step1['selected_branch_indices']
print(f"Step 1 标记selected的分支: {selected_branches}")
```

## 关键发现

### Selected vs 实际扩展

**重要理解**：`selected_branch_indices` ≠ 实际扩展的分支数

```
Step 1: selected=[0, 1, 2, 3]  (标记4个)
Step 2: parent_indices=[0, 0, 0, 0, 1, 1, 1, 1]
        → 实际扩展: {0, 1}  (只有2个)
```

**原因**：
1. **局部选择**: 每个父节点独立选择top-k → 标记多个selected
2. **全局筛选**: 从所有候选中排序 → 只保留beam_size个
3. **workload记录局部选择结果**，通过下一步的parent_indices可见实际扩展数

### 资源估算建议

对于3D-TTS模拟：

```python
# ❌ 错误：使用selected数量
active_beams = len(step['selected_branch_indices'])  # 可能>beam_size

# ✅ 正确：使用beam_size
active_beams = workload['beam_search_config']['beam_size']

# ✅ 或者：查看下一步的parent_indices
next_step = workload['decode_steps'][current_step + 1]
actual_expanded = len(set(next_step['branch_parent_indices']))
```

## 相关文档

1. **`README.md`** - 工具使用说明
2. **`BEAM_SEARCH_MECHANISM.md`** - Beam search选择机制详解
3. **`PARENT_TRACKING_GUIDE.md`** - 父分支追踪使用指南
4. **`verify_parent_indices.py`** - 验证脚本

## 命令快速参考

### 生成workload

```bash
cd compute-optimal-tts/process/wordload

# 单个配置
python3 gen_workload_beam.py \
  --input ../../src/output/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/16384_8_2 \
  --dataset AIME24

# 批量所有配置
BASE_DIR="../../src/output/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
for config_dir in ${BASE_DIR}/*/; do
    python3 gen_workload_beam.py --input "$config_dir" --dataset AIME24
done
```

### 验证workload

```bash
# 验证单个文件
python3 verify_parent_indices.py output_beam/AIME24/16384_8_2/question_17/workload_0.json

# 查看统计
find output_beam/AIME24 -name "workload_0.json" | wc -l
du -sh output_beam/AIME24/
```

## 总结

✅ **成功生成**: 180个workload文件 (6个配置 × 30个问题)
✅ **包含完整的树结构追踪信息** (`branch_parent_indices`)
✅ **所有验证通过**
⚠️ **1个配置失败**: 16384_2_1 (原始数据问题)

所有生成的workload文件已就绪，可用于3D-TTS模拟研究！
