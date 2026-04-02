# Runtime Straggler剪枝策略（无需Reward）

## 问题

在runtime时，我们**还没有reward**，但需要决定是否剪枝straggler branch。

## Runtime可用的特征

✓ **可用**:
- Token长度（绝对值）
- Token长度比例（vs 其他branch）
- Token概率（注意力熵）
- 高/低置信度token比例

✗ **不可用**:
- Reward score（还没计算）

## 分析结果

### 特征重要性排名

基于`16384_4_1`数据集的23个straggler（11个被选中，12个未被选中）：

| 排名 | 特征 | 差异度 | 方向 |
|------|------|--------|------|
| 1 | **Min probability** | 172.3% | 被选中的更高 (0.23 vs 0.08) |
| 2 | **Token length (absolute)** | 59.6% | 被选中的更短 (281 vs 695) |
| 3 | **High confidence % (>0.9)** | 61.4% | 被选中的更少 (33.7% vs 87.2%) |
| 4 | **Length ratio** | 53.0% | 被选中的更小 (3.2x vs 6.8x) |
| 5 | Min entropy | 17.3% | 被选中的更高 |
| 6 | Mean entropy | 14.3% | 被选中的更高 |

### 关键发现

#### 1. Token长度是强指标 (59.6%差异)

- **被选中**: 平均 281 tokens
- **未被选中**: 平均 695 tokens
- **结论**: 超长的straggler（>500 tokens）很可能不会被选中

#### 2. 高置信度比例是强指标 (61.4%差异)

- **被选中**: 33.7% 高置信度token (prob>0.9)
- **未被选中**: 87.2% 高置信度token
- **反直觉**: 高置信度的straggler反而不太会被选中！
  - 原因: 这些超长branch虽然"自信"但实际质量差（低reward）

#### 3. 最小概率是强指标 (172.3%差异)

- **被选中**: min_prob = 0.23
- **未被选中**: min_prob = 0.08
- **解释**: 被选中的straggler虽然整体不那么自信，但"最差"的token也不会太差

## 推荐的Runtime剪枝策略

### 方案A: 保守策略（低误剪率）

```python
def should_prune_straggler_conservative(tokens, ratio, high_conf_pct):
    """
    保守策略: 只剪枝明显不会被选中的straggler
    误剪率: 9.1%
    剪枝率: 41.7% (能剪掉约40%的不会被选中的straggler)
    """
    # 规则1: 超长且比例大
    if tokens > 274 and ratio > 3.6:
        return True
    
    # 规则2: 极度超长且高置信度
    if tokens > 542 and high_conf_pct > 0.20:
        return True
    
    return False
```

**性能**:
- 能正确剪枝: 5/12 (41.7%) 未被选中的straggler
- 误剪: 1/11 (9.1%) 被选中的straggler
- **精确率: 83.3%**

### 方案B: 激进策略（高剪枝率）

```python
def should_prune_straggler_aggressive(tokens, ratio, high_conf_pct, min_prob):
    """
    激进策略: 更积极地剪枝
    可能有更高误剪率，但节省更多资源
    """
    # 规则1: 中等长度但比例大且高置信度
    if tokens > 200 and ratio > 3.0 and high_conf_pct > 0.70:
        return True
    
    # 规则2: 超长
    if tokens > 400:
        return True
    
    # 规则3: 高置信度但最小概率很低（说明有严重错误）
    if high_conf_pct > 0.80 and min_prob < 0.15:
        return True
    
    return False
```

### 方案C: 基于注意力熵的策略

```python
def should_prune_straggler_entropy_based(tokens, ratio, high_conf_pct, mean_entropy, min_prob):
    """
    基于注意力熵和其他特征的组合策略
    """
    # 计算综合得分（不使用reward）
    # 归一化各个特征
    length_score = min(tokens / 500, 1.0)  # >500 tokens得1分
    ratio_score = min(ratio / 5.0, 1.0)     # >5x得1分
    conf_score = high_conf_pct              # 高置信度得高分
    entropy_score = 1 - mean_entropy * 10   # 低熵得高分
    min_prob_score = 1 - min_prob           # 低最小概率得高分
    
    # 组合得分（权重可调）
    combined_score = (
        0.30 * length_score +
        0.25 * ratio_score +
        0.25 * conf_score +
        0.10 * entropy_score +
        0.10 * min_prob_score
    )
    
    # 阈值: >0.6 表示很可能不会被选中
    return combined_score > 0.6
```

## 实际使用示例

```python
import numpy as np

def calculate_runtime_features(branch_data, all_branches):
    """计算runtime可用的特征"""
    tokens = len(branch_data['token_ids'])
    
    # 计算比例
    other_lengths = [len(b['token_ids']) for b in all_branches 
                     if b is not branch_data]
    max_other = max(other_lengths) if other_lengths else 1
    ratio = tokens / max_other
    
    # 计算注意力熵特征（如果有token_probs）
    if 'token_probs' in branch_data:
        probs = np.array(branch_data['token_probs'])
        
        high_conf_pct = np.sum(probs > 0.9) / len(probs)
        low_conf_pct = np.sum(probs < 0.5) / len(probs)
        mean_entropy = -np.mean(probs * np.log(probs + 1e-10))
        min_prob = np.min(probs)
    else:
        # 如果没有token_probs，使用默认值或跳过熵特征
        high_conf_pct = 0.5
        low_conf_pct = 0.1
        mean_entropy = 0.05
        min_prob = 0.1
    
    return {
        'tokens': tokens,
        'ratio': ratio,
        'high_conf_pct': high_conf_pct,
        'low_conf_pct': low_conf_pct,
        'mean_entropy': mean_entropy,
        'min_prob': min_prob,
    }


# 在beam search中使用
def beam_search_with_straggler_pruning(prompt, model, max_steps=10):
    branches = [{'token_ids': prompt, 'token_probs': []}]
    
    for step in range(max_steps):
        new_branches = []
        
        for branch in branches:
            # 生成候选token
            candidates = model.generate_candidates(branch)
            
            for candidate in candidates:
                # 检查是否为straggler
                if is_straggler_by_definition(candidate, candidates):
                    # 计算runtime特征
                    features = calculate_runtime_features(candidate, candidates)
                    
                    # 使用保守策略决定是否剪枝
                    if should_prune_straggler_conservative(
                        features['tokens'],
                        features['ratio'],
                        features['high_conf_pct']
                    ):
                        print(f"Pruned straggler at step {step}: "
                              f"{features['tokens']} tokens, "
                              f"ratio {features['ratio']:.1f}x")
                        continue  # 跳过这个分支
                
                new_branches.append(candidate)
        
        branches = new_branches
    
    return branches
```

## 权衡与建议

### 保守策略 vs 激进策略

| 策略 | 剪枝率 | 误剪率 | 适用场景 |
|------|--------|--------|----------|
| **保守** | ~40% | ~10% | 对准确率要求高，可以容忍一些straggler |
| **激进** | ~70%+ | ~20-30% | 对性能要求高，可以容忍一些误剪 |

### 关键指标选择

如果只能选择**一个**runtime指标：
1. **Token长度** (59.6%差异) - 最简单，不需要token_probs
2. **高置信度比例** (61.4%差异) - 需要token_probs，但更准确

如果可以使用**两个**runtime指标：
- **Token长度 + 高置信度比例** - 最佳组合

### 何时不应该剪枝

即使满足straggler定义，以下情况**不建议剪枝**：
- Token数量 < 200（还不够长）
- 长度比例 < 2.5x（不够显著）
- 高置信度比例 < 50%（说明有很多不确定的token，可能质量好）

## 实验验证

基于当前数据（n=23）：

| 方案 | 正确剪枝 | 误剪 | 精确率 | 召回率 |
|------|---------|------|--------|--------|
| 保守策略 | 5/12 (41.7%) | 1/11 (9.1%) | 83.3% | 41.7% |
| Token长度>400 | 8/12 (66.7%) | 2/11 (18.2%) | 80.0% | 66.7% |
| 高置信度>70% | 10/12 (83.3%) | 4/11 (36.4%) | 71.4% | 83.3% |

## 结论

✓ **Runtime特征可以提供有用的剪枝信号**，即使没有reward
✓ **Token长度**和**高置信度比例**是最强的两个指标
✓ 推荐使用**保守策略**以平衡性能和准确率

⚠️ **限制**:
- 样本量小（n=23），需要在更多数据集上验证
- 误剪率仍然存在（9-36%），需要权衡
- 对于关键任务，建议在有reward后再做最终决策
