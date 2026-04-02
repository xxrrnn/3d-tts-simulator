# Beam Search 父分支追踪使用指南

## `branch_parent_indices` 字段说明

每个workload文件的 `branch_parent_indices` 字段记录了每个分支的父分支索引，用于追踪完整的beam search树结构。

### 字段含义

```json
{
  "step": 1,
  "branch_parent_indices": [2, 2, 5, 5, 5]
}
```

- **索引对应关系**：`branch_parent_indices[i]` 表示当前步骤的第i个分支来自上一步的哪个分支
- **值为 -1**：表示根节点（仅在Step 0出现）
- **值 >= 0**：表示来自上一步的分支索引

### 示例解读

假设 Step 1 的 `branch_parent_indices = [2, 2, 5, 5, 5]`：

- 分支0 的 parent=2 → 来自 Step 0 的分支2
- 分支1 的 parent=2 → 来自 Step 0 的分支2
- 分支2 的 parent=5 → 来自 Step 0 的分支5
- 分支3 的 parent=5 → 来自 Step 0 的分支5
- 分支4 的 parent=5 → 来自 Step 0 的分支5

## 使用场景

### 1. 回溯完整路径

从某个分支回溯到根节点，重建完整的生成路径：

```python
def trace_path(workload, target_step, target_branch):
    """回溯从根节点到指定分支的完整路径"""
    path = [(target_step, target_branch)]
    
    current_step = target_step
    current_branch = target_branch
    
    while current_step > 0:
        parent_idx = workload['decode_steps'][current_step]['branch_parent_indices'][current_branch]
        
        if parent_idx == -1:  # 到达根节点
            break
        
        current_step -= 1
        current_branch = parent_idx
        path.insert(0, (current_step, current_branch))
    
    return path

# 使用示例
path = trace_path(workload, final_step=2, final_branch=4)
print(f"完整路径: {path}")
# 输出: [(0, 2), (1, 1), (2, 4)]
```

### 2. 可视化树结构

按父节点分组显示树形结构：

```python
def visualize_tree(workload, max_steps=5):
    """可视化beam search树结构"""
    for step_idx in range(min(max_steps, len(workload['decode_steps']))):
        step = workload['decode_steps'][step_idx]
        print(f"\n{'='*80}")
        print(f"Step {step['step']}")
        print(f"{'='*80}")
        
        # 按父节点分组
        parent_groups = {}
        for i in range(step['branch_count']):
            parent_idx = step['branch_parent_indices'][i]
            if parent_idx not in parent_groups:
                parent_groups[parent_idx] = []
            parent_groups[parent_idx].append(i)
        
        # 显示每个父节点的子分支
        for parent_idx in sorted(parent_groups.keys()):
            if parent_idx == -1:
                print(f"\n从根节点扩展:")
            else:
                print(f"\n从 Step{step_idx-1}/分支{parent_idx} 扩展:")
            
            children = parent_groups[parent_idx]
            for child_idx in children:
                reward = step['branch_rewards'][child_idx]
                tokens = step['branch_tokens'][child_idx]
                selected = "[S]" if child_idx in step['selected_branch_indices'] else "[X]"
                print(f"  {selected} 分支{child_idx}: tokens={tokens:3d}, reward={reward:.4f}")
```

### 3. 分析分支效率

统计每个父节点产生了多少子分支：

```python
def analyze_branch_efficiency(workload):
    """分析每个父节点的扩展效率"""
    for step_idx in range(1, len(workload['decode_steps'])):
        step = workload['decode_steps'][step_idx]
        parent_indices = step['branch_parent_indices']
        
        # 统计每个父节点的子分支数
        from collections import Counter
        parent_counts = Counter(parent_indices)
        
        print(f"\nStep {step['step']} 扩展统计:")
        for parent_idx, count in sorted(parent_counts.items()):
            prev_step = workload['decode_steps'][step_idx - 1]
            parent_reward = prev_step['branch_rewards'][parent_idx]
            print(f"  Step{step_idx-1}/分支{parent_idx} (reward={parent_reward:.4f}) → {count}个子分支")
```

### 4. 计算KV Cache共享

识别可以共享KV Cache的分支（相同父节点）：

```python
def find_cache_sharing_opportunities(workload, step_idx):
    """找出可以共享KV Cache的分支组"""
    step = workload['decode_steps'][step_idx]
    parent_groups = {}
    
    for i, parent_idx in enumerate(step['branch_parent_indices']):
        if parent_idx not in parent_groups:
            parent_groups[parent_idx] = []
        parent_groups[parent_idx].append(i)
    
    print(f"\nStep {step_idx} KV Cache共享机会:")
    for parent_idx, children in parent_groups.items():
        if len(children) > 1:
            print(f"  从Step{step_idx-1}/分支{parent_idx}扩展的 {len(children)} 个分支可以共享prefill cache:")
            print(f"    分支: {children}")
            
            # 计算可节省的token数
            if step_idx > 0:
                prev_step = workload['decode_steps'][step_idx - 1]
                # 累计从root到parent的所有token
                total_prefix_tokens = sum(prev_step['branch_tokens'][:parent_idx+1])
                saved_tokens = total_prefix_tokens * (len(children) - 1)
                print(f"    可节省: ~{saved_tokens} tokens的重复计算")
```

## 实际案例

### 案例：追踪最优路径

```python
import json

# 加载workload
with open('workload_0.json', 'r') as f:
    workload = json.load(f)

# 找到最后一步reward最高的分支
final_step = workload['decode_steps'][-1]
best_branch_idx = final_step['branch_rewards'].index(max(final_step['branch_rewards']))
final_step_num = final_step['step']

print(f"最优分支: Step {final_step_num} / 分支 {best_branch_idx}")
print(f"Reward: {final_step['branch_rewards'][best_branch_idx]:.4f}")

# 回溯完整路径
path = trace_path(workload, final_step_num, best_branch_idx)

print("\n完整路径:")
for step_idx, branch_idx in path:
    step = workload['decode_steps'][step_idx]
    reward = step['branch_rewards'][branch_idx]
    tokens = step['branch_tokens'][branch_idx]
    print(f"  Step {step_idx}, 分支 {branch_idx}: tokens={tokens}, reward={reward:.4f}")
```

### 输出示例

```
最优分支: Step 2 / 分支 4
Reward: 0.2974

完整路径:
  Step 0, 分支 2: tokens=412, reward=0.0621
  Step 1, 分支 1: tokens=13, reward=0.0555
  Step 2, 分支 4: tokens=299, reward=0.2974
```

## 关键理解

### `selected_branch_indices` vs `branch_parent_indices`

- **`selected_branch_indices`**：当前步骤中**局部**被标记为selected的分支
  - 数量可能 > beam_size（因为多个父节点独立选择）
  - 表示"候选"，不是最终保留
  
- **`branch_parent_indices`**：每个分支的父分支索引
  - 通过查看**下一步的parent_indices**可以知道哪些分支真正被扩展
  - 实际扩展的分支数 ≈ beam_size

### 示例对比

```json
// Step 1
{
  "selected_branch_indices": [0, 1, 2, 3],  // 标记了4个
  "branch_rewards": [0.0687, 0.0555, 0.0501, 0.0501]
}

// Step 2
{
  "branch_parent_indices": [0, 0, 0, 0, 1, 1, 1, 1]
  //                        ↑ 只有0和1被扩展
}
```

**结论**：Step 1虽然标记了4个selected，但实际只有分支0和1被扩展到Step 2。

## 完整工作流示例

```python
def analyze_beam_search(workload_file):
    """完整分析beam search的工作流"""
    with open(workload_file, 'r') as f:
        workload = json.load(f)
    
    beam_size = workload['beam_search_config']['beam_size']
    print(f"Beam Size: {beam_size}")
    print(f"总步数: {len(workload['decode_steps'])}\n")
    
    # 1. 可视化树结构
    print("="*80)
    print("树结构可视化")
    print("="*80)
    visualize_tree(workload, max_steps=3)
    
    # 2. 分析扩展效率
    print("\n" + "="*80)
    print("扩展效率分析")
    print("="*80)
    analyze_branch_efficiency(workload)
    
    # 3. 找出最优路径
    print("\n" + "="*80)
    print("最优路径")
    print("="*80)
    final_step = workload['decode_steps'][-1]
    best_idx = final_step['branch_rewards'].index(max(final_step['branch_rewards']))
    path = trace_path(workload, len(workload['decode_steps'])-1, best_idx)
    
    print("完整路径:")
    for step_idx, branch_idx in path:
        step = workload['decode_steps'][step_idx]
        print(f"  Step {step_idx}/分支{branch_idx}: "
              f"reward={step['branch_rewards'][branch_idx]:.4f}, "
              f"tokens={step['branch_tokens'][branch_idx]}")
    
    # 4. KV Cache优化机会
    print("\n" + "="*80)
    print("KV Cache优化机会")
    print("="*80)
    for step_idx in range(1, min(4, len(workload['decode_steps']))):
        find_cache_sharing_opportunities(workload, step_idx)

# 运行分析
analyze_beam_search('workload_0.json')
```

## 总结

`branch_parent_indices` 提供了完整的树结构信息，使你能够：

1. ✅ 回溯任意分支的完整路径
2. ✅ 可视化beam search的探索过程
3. ✅ 分析不同父节点的扩展效率
4. ✅ 识别KV Cache共享机会
5. ✅ 理解beam_size与实际扩展数的关系

通过这个字段，可以完整重建beam search的决策过程，为3D-TTS模拟提供准确的资源需求估算。
