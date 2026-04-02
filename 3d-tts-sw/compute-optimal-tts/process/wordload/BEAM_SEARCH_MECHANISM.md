# Beam Search 分支选择机制详解

## 核心问题解答

### 1. `num_sequence` 是什么？

**`num_sequence` = `beam_size`**

在代码中（`evaluate.py` line 285, 301）：
```python
beam_size = args.num_sequence
```

在配置名称中（例如 `16384_8_2`）：
- `16384` = `max_step`（最大搜索步数）
- `8` = `tree_max_width`（每次生成的候选分支数）
- `2` = `num_sequence` = **`beam_size`**（目标保留的beam数量）

### 2. 为什么 beam_size=2 时，每步会选择多于2个分支？

**核心原因**：Beam search是**从多个父节点同时扩展**的算法！

#### 实际例子（beam_size=2）

```
Step 0: 1个父节点（根节点）
  ├─ 生成8个候选分支
  ├─ 选择top-2: Branch[2], Branch[5]
  └─ selected标记: 2个 ✓

Step 1: 2个父节点（来自Step 0）
  ├─ 父节点1（Branch[2]）→ 生成若干子分支 → 选择top-2
  ├─ 父节点2（Branch[5]）→ 生成若干子分支 → 选择top-2
  └─ selected标记: 2+2 = 4个 ⚠️

Step 2: 2个父节点
  ├─ 父节点1 → 选择top-2
  ├─ 父节点2 → 选择top-2
  └─ selected标记: 4个 ⚠️
```

### 3. 谁决定每次step选择几个branch？

**三重决策机制**：

#### 第一层：计算k值（`tree.py` line 370）
```python
k = beam_size - len(end_nodes)
```
- `beam_size`: 目标保留的beam数量（例如2）
- `end_nodes`: 已经终止的节点数量
- `k`: 每个父节点需要选择的子节点数量

#### 第二层：每个父节点独立选择（`tree.py` line 348-413）
```python
for cur_node in cur_nodes_to_search:  # 遍历所有活跃的父节点
    if not cur_node.terminated:
        # 当前父节点选择top-k个子节点
        top_k_children = sorted(cur_node.children)[:k]
        
        # 标记这k个为selected (line 389, 405)
        selected_indices = set(c_idx for c_idx, _, _, _, _ in sorted_candidates[:k])
        for child_idx in cur_node.children:
            branch_info["selected"] = child_idx in selected_indices
        
        # 加入候选队列
        for child in top_k_children:
            top_k_nodes.append(child)
```

**关键点**：
- 每个父节点**独立**选择top-k个子节点
- 所有被选中的子节点都被标记为 `selected=True`
- 如果有N个活跃父节点，**最多标记 N × k 个分支**

#### 第三层：全局重排序（`tree.py` line 472）
```python
top_k_nodes = heapq.nsmallest(k, top_k_nodes)
```
- 从所有候选中全局选择top-k个进入下一步扩展
- 但**selected标记已经在第二层打上了**，无法撤销

## 为什么会出现"标记数 > 实际进入下一步数"？

### 代码执行顺序

1. **Line 389**: 为每个父节点的top-k子节点计算 `selected_indices`
2. **Line 405**: 标记 `"selected": True`（记录到detailed_beam_search_log）
3. **Line 413**: 将选中的分支加入 `top_k_nodes` 队列
4. **Line 472**: 全局排序，只保留k个进入下一步

### 时序图

```
时间 →

T1: 父节点1选择 → 标记2个selected ✓
T2: 父节点2选择 → 标记2个selected ✓
T3: 全局排序   → 只保留2个进入下一步
                 但已经标记了4个selected ⚠️
```

### 关键发现

**标记selected的时机早于全局筛选**：
- Selected标记发生在**每个父节点的局部选择阶段**
- 全局筛选发生在**所有父节点都选完之后**
- 因此，workload中记录的 `selected_branch_indices` 包含**所有局部top-k**，而不是全局top-k

## Workload中 selected_branch_indices 的含义

在生成的workload文件中：
```json
{
  "step": 1,
  "decode_steps": {
    "branches": [
      {"branch_index": 0, "tokens": 45, "reward": 0.85},
      {"branch_index": 1, "tokens": 38, "reward": 0.82},
      {"branch_index": 2, "tokens": 52, "reward": 0.79},
      {"branch_index": 3, "tokens": 41, "reward": 0.76}
    ],
    "selected_branch_indices": [0, 1, 2, 3],
    "beam_search_config": {"beam_size": 2}
  }
}
```

**解读**：
- `selected_branch_indices = [0, 1, 2, 3]`：这4个分支在局部选择阶段被标记
- `beam_size = 2`：最终只有2个分支会进入下一步扩展
- **不矛盾**！因为：
  - 4个是**所有父节点的局部top-k总和**
  - 2个是**全局top-k（实际扩展数）**

## 实际影响分析

### 对资源需求的影响

假设 beam_size=2, 有2个活跃父节点：

| 阶段 | 标记selected | 实际扩展 | KV Cache需求 |
|------|-------------|---------|-------------|
| 局部选择 | 4个 | - | - |
| 全局筛选 | - | 2个 | 2×cache |
| 下一步扩展 | - | 2个 | 2×cache |

**关键结论**：
- ❌ 不应使用 `len(selected_branch_indices)` 来估算KV Cache需求
- ✅ 应使用 `beam_size` 来估算实际资源需求
- `selected_branch_indices` 记录的是**候选生成过程**，不是最终保留数

### 对3D-TTS模拟的建议

在模拟器中使用workload时：

1. **理解语义**：
   - `len(branches)`: 当前step所有候选分支数量
   - `len(selected_branch_indices)`: 局部选择阶段标记的分支数
   - `beam_size`: 实际进入下一step的beam数量

2. **资源分配**：
   ```python
   # ✅ 正确的资源估算
   active_beams = beam_size
   kv_cache_needed = active_beams * per_beam_cache
   
   # ❌ 错误的资源估算
   kv_cache_needed = len(selected_branch_indices) * per_beam_cache
   ```

3. **并行度**：
   - 候选生成阶段：可能需要并行生成 `len(branches)` 个分支
   - 保持阶段：只需维持 `beam_size` 个beam的资源
   - Reward计算：需要对 `len(branches)` 个分支进行打分

## 配置参数总结

配置格式：`{max_step}_{tree_max_width}_{num_sequence}`

例如：`16384_8_2`

| 参数 | 值 | 含义 | 影响 |
|------|---|------|------|
| max_step | 16384 | 最大搜索步数 | 搜索深度上限 |
| tree_max_width | 8 | 每次生成的候选数 | 每个节点扩展时生成8个分支 |
| num_sequence | 2 | beam_size | 目标保留2条最优路径 |

**实际行为**：
- 每个节点扩展时，LLM生成 `tree_max_width=8` 个候选分支
- 每个父节点选择 `k = beam_size - end_nodes ≈ 2` 个子节点
- 如果有2个父节点，会标记 `2×2=4` 个selected
- 全局排序后，实际保留 `beam_size=2` 个beam进入下一步

## 代码追踪参考

关键代码位置（`src/reason/guided_search/tree.py`）：

- **Line 370**: `k = beam_size - len(end_nodes)` - 计算每个父节点选k个
- **Line 348**: `for cur_node in cur_nodes_to_search` - 遍历所有父节点
- **Line 389**: `selected_indices = set(... sorted_candidates[:k])` - 标记局部top-k
- **Line 405**: `"selected": child_idx in selected_indices` - 记录selected状态
- **Line 472**: `top_k_nodes = heapq.nsmallest(k, top_k_nodes)` - 全局筛选top-k

## 总结

### `num_sequence` 是什么？
- `num_sequence` = `beam_size` = 配置中最后一个数字
- 表示**目标保留的beam数量**，不是每步选择的分支数

### 谁决定每次选几个branch？
1. **每个父节点**独立选择 `k = beam_size - end_nodes` 个子节点
2. **所有父节点**的选择汇总后，标记的selected数量 = `N_parents × k`
3. **全局排序**后，实际进入下一步的数量 = `k ≈ beam_size`

### 关键insight
- **Selected标记 ≠ 最终保留**
- Selected标记发生在**局部选择阶段**
- 最终保留发生在**全局筛选阶段**
- Workload记录的是**局部选择的中间状态**

### 对模拟的影响
- 使用 `beam_size` 而非 `len(selected_branch_indices)` 来估算资源
- 理解selected的语义：局部候选，不是最终保留
- KV Cache需求 = `beam_size` × per_beam_cache
