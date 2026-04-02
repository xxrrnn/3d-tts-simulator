# 动态Runtime剪枝器使用指南

## 概述

`DynamicStraggler剪枝器` 是一个**实时**剪枝决策系统，在beam search的每个step都可以动态决定是否剪枝某个branch，**无需知道branch的最终长度或reward**。

## 核心特点

✓ **完全基于runtime可用信息**:
  - 当前branch已生成的token数量
  - 其他branch的当前状态
  - 当前step索引
  - 已生成token的概率分布

✓ **动态阈值**:
  - 根据其他branch的长度分布自动调整
  - 早期宽松，后期严格
  - 考虑token置信度

✓ **保护性机制**:
  - 高置信度branch获得更高容忍度
  - 早期阶段更宽容
  - 考虑增长趋势

## 快速开始

### 基本用法

```python
from dynamic_pruner_runtime import DynamicStraggler剪枝器, BranchState

# 1. 创建剪枝器
pruner = DynamicStraggler剪枝器({
    'base_ratio_threshold': 2.0,     # 基础长度比例阈值
    'min_tokens_to_check': 50,       # 开始检查的最小token数
    'high_conf_boost': 1.5,          # 高置信度的容忍度加成
})

# 2. 在每个step检查branch
other_branches = [
    BranchState(0, 45, [0.95] * 45, False),
    BranchState(1, 52, [0.92] * 52, False),
]

current_branch = BranchState(
    branch_id=2,
    current_tokens=120,
    token_probs=[0.98] * 120,
    is_finished=False
)

# 3. 决定是否剪枝
should_prune, reason, details = pruner.should_prune_branch(
    current_branch,
    other_branches,
    current_step=5,
    total_steps=10,  # 可选
    verbose=True
)

if should_prune:
    print(f"剪枝branch {current_branch.branch_id}: {reason}")
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_ratio_threshold` | 2.0 | 基础长度比例阈值。branch长度超过其他branch最大值的这个倍数时考虑剪枝 |
| `min_tokens_to_check` | 50 | 开始检查的最小token数。太短的branch不检查 |
| `step_factor` | 1.0 | Step因子（未使用，保留） |
| `high_conf_boost` | 1.5 | 高置信度加成。高置信度branch的阈值乘以这个值 |
| `early_warning_ratio` | 1.5 | 早期警告比例（未使用，保留） |

### 推荐配置

**保守配置**（误剪率低）:
```python
{
    'base_ratio_threshold': 2.5,  # 更高的阈值
    'min_tokens_to_check': 80,    # 等更长再检查
    'high_conf_boost': 2.0,       # 更大的高置信度容忍
}
```

**激进配置**（剪枝更多）:
```python
{
    'base_ratio_threshold': 1.8,  # 更低的阈值
    'min_tokens_to_check': 40,    # 更早开始检查
    'high_conf_boost': 1.3,       # 较小的高置信度容忍
}
```

**平衡配置**（推荐）:
```python
{
    'base_ratio_threshold': 2.0,
    'min_tokens_to_check': 50,
    'high_conf_boost': 1.5,
}
```

## 动态阈值机制

### 1. 基础阈值计算

```
动态长度阈值 = max_other_length × 动态比例阈值
```

其中：
- `max_other_length`: 其他branch的最大长度
- `动态比例阈值`: 根据step进度调整

### 2. Step进度调整

| Step阶段 | 进度 | Step乘数 | 说明 |
|---------|------|----------|------|
| **早期** | 0-30% | 1.5x | 更宽松，给branch机会 |
| **中期** | 30-70% | 1.0x | 标准阈值 |
| **后期** | 70-100% | 0.8x | 更严格，快速收敛 |

如果不知道`total_steps`，则使用固定衰减：
```
step_multiplier = max(1.0 - current_step × 0.02, 0.7)
```

### 3. 统计调整

如果其他branch长度差异大（std > mean × 0.3），阈值提高1.2倍（更宽容）

### 4. 高置信度保护

如果branch的高置信度token比例 > 80%，阈值乘以`high_conf_boost`（默认1.5）

## 剪枝决策流程

```
1. 检查基本条件
   ├─ Branch已完成? → 不剪枝
   ├─ Token数 < min_tokens_to_check? → 不剪枝
   └─ 没有其他branch? → 不剪枝

2. 计算动态阈值
   ├─ 基于其他branch的长度分布
   ├─ 考虑当前step进度
   └─ 考虑长度差异

3. 检查绝对长度
   ├─ current_tokens ≤ 动态长度阈值? → 不剪枝
   └─ 超过阈值 → 继续检查

4. 检查长度比例
   ├─ ratio ≤ 动态比例阈值? → 不剪枝（虽然长但比例还行）
   └─ 超过比例阈值 → 检查置信度

5. 高置信度保护
   ├─ 高置信度 > 80%? → 使用adjusted_threshold
   │   ├─ ratio ≤ adjusted_threshold? → 不剪枝（保护）
   │   └─ ratio > adjusted_threshold? → 剪枝
   └─ 低置信度? → 剪枝

6. 增长趋势检查
   └─ 增长率 > 3.0x? → 剪枝
```

## 实际集成示例

### 示例1: 在Beam Search中集成

```python
def beam_search_with_dynamic_pruning(prompt, model, beam_width=4, max_steps=20):
    """带动态剪枝的beam search"""
    
    # 创建剪枝器
    pruner = DynamicStraggler剪枝器({
        'base_ratio_threshold': 2.0,
        'min_tokens_to_check': 50,
    })
    
    # 初始化branches
    branches = [
        {
            'id': i,
            'tokens': list(prompt),
            'token_probs': [],
            'finished': False
        }
        for i in range(beam_width)
    ]
    
    for step in range(max_steps):
        new_branches = []
        
        for branch in branches:
            if branch['finished']:
                new_branches.append(branch)
                continue
            
            # 生成下一个token
            next_token, prob = model.generate_next(branch['tokens'])
            
            # 更新branch状态
            new_branch = branch.copy()
            new_branch['tokens'].append(next_token)
            new_branch['token_probs'].append(prob)
            
            # 检查是否完成
            if next_token == EOS_TOKEN:
                new_branch['finished'] = True
                new_branches.append(new_branch)
                continue
            
            # 检查是否需要剪枝
            current_state = BranchState(
                branch_id=branch['id'],
                current_tokens=len(new_branch['tokens']),
                token_probs=new_branch['token_probs'],
                is_finished=False
            )
            
            other_states = [
                BranchState(
                    branch_id=b['id'],
                    current_tokens=len(b['tokens']),
                    token_probs=b['token_probs'],
                    is_finished=b['finished']
                )
                for b in branches if b['id'] != branch['id']
            ]
            
            should_prune, reason, _ = pruner.should_prune_branch(
                current_state,
                other_states,
                current_step=step,
                total_steps=max_steps
            )
            
            if should_prune:
                print(f"[Step {step}] Pruned branch {branch['id']}: {reason}")
                # 不添加到new_branches，即剪枝
            else:
                new_branches.append(new_branch)
        
        branches = new_branches
        
        # 如果所有branch都完成或被剪枝，提前结束
        if not branches or all(b['finished'] for b in branches):
            break
    
    # 统计
    stats = pruner.get_stats()
    print(f"\nPruning stats: {stats}")
    
    return branches
```

### 示例2: 增量检查模式

```python
class IncrementalPruningChecker:
    """增量剪枝检查器 - 每生成N个token检查一次"""
    
    def __init__(self, pruner, check_interval=10):
        self.pruner = pruner
        self.check_interval = check_interval
        self.last_check = {}  # branch_id -> last_check_length
    
    def should_check_now(self, branch_id, current_length):
        """是否应该现在检查"""
        last = self.last_check.get(branch_id, 0)
        
        if current_length - last >= self.check_interval:
            self.last_check[branch_id] = current_length
            return True
        return False
    
    def check_branch(self, branch, other_branches, current_step):
        """检查branch"""
        if not self.should_check_now(branch.branch_id, branch.current_tokens):
            return False, "not_time_to_check", {}
        
        return self.pruner.should_prune_branch(
            branch,
            other_branches,
            current_step
        )
```

## 测试场景

运行 `python dynamic_pruner_runtime.py` 查看5个测试场景：

1. **场景1**: 超长但高置信度 → 保护，不剪枝
2. **场景2**: 极度超长且低置信度 → 剪枝
3. **场景3**: 长度在合理范围 → 不剪枝
4. **场景4**: 早期阶段 → 宽容，不剪枝
5. **场景5**: 后期阶段 → 严格检查

## 性能考虑

### 计算开销

- **每次检查**: O(n) 其中n是其他branch数量
- **建议**: 不要每个token都检查，使用增量检查（如每10个token检查一次）

### 内存开销

- `BranchState`: 约O(m) 其中m是已生成的token数
- 如果不需要熵特征，可以不存储`token_probs`以节省内存

## 调优建议

### 根据数据集特点调整

**短文本任务**（平均 < 200 tokens）:
```python
{
    'base_ratio_threshold': 2.5,  # 更严格
    'min_tokens_to_check': 30,
}
```

**长文本任务**（平均 > 500 tokens）:
```python
{
    'base_ratio_threshold': 1.8,  # 更宽松
    'min_tokens_to_check': 100,
}
```

**数学推理**（可能需要长推导）:
```python
{
    'base_ratio_threshold': 3.0,  # 非常宽松
    'min_tokens_to_check': 80,
    'high_conf_boost': 2.0,
}
```

### 监控指标

```python
# 定期检查统计
stats = pruner.get_stats()
print(f"剪枝率: {stats['prune_rate']*100:.1f}%")

# 如果剪枝率太高（>50%），考虑:
# - 提高 base_ratio_threshold
# - 增大 high_conf_boost
# - 增大 min_tokens_to_check

# 如果剪枝率太低（<5%），考虑:
# - 降低 base_ratio_threshold
# - 减小 high_conf_boost
```

## 限制与注意事项

⚠️ **不知道最终长度**: 只能基于"当前"状态判断，可能出现误剪

⚠️ **早期误判**: 某些branch可能后期加速收敛，早期看似过长

⚠️ **需要调优**: 不同任务需要不同的阈值配置

✓ **建议**: 先在验证集上测试，找到最佳参数后再用于生产

## 下一步

1. 在实际workload上测试这个剪枝器
2. 统计实际的误剪率和资源节省
3. 根据结果调整参数
