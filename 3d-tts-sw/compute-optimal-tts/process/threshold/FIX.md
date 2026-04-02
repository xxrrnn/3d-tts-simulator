# 🔴 重要修正: 只使用历史Reward

**日期**: 2026-04-01  
**严重程度**: Critical  
**状态**: ✅ 已修复

## 问题描述

原始实现错误地假设在做剪枝决策时可以获得"当前step的reward"。但在实际运行时:

1. **Token还未生成** - 当前step的token还没有生成
2. **Reward未计算** - PRM模型需要在生成后才能评估reward
3. **因果关系错误** - 不能用"未来"的reward来决定"现在"是否剪枝

## 修正内容

### 核心变更

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| 特征名称 | `reward_current` | `reward_last` |
| 含义 | 当前step的reward | 历史最后一步的reward |
| 决策依据 | 使用当前reward | 只使用历史reward |

### 代码修改

#### 1. 特征提取

```python
# ❌ 修改前 - 错误
features['reward_current'] = rewards[-1]  # 假设当前step已有reward

# ✅ 修改后 - 正确
features['reward_last'] = rewards[-1]  # 使用历史最后一步的reward
# rewards[-1] 是上一步已完成的reward,不是当前步
```

#### 2. 决策规则

```python
# ❌ 修改前
if features['reward_current'] < threshold:
    prune()

# ✅ 修改后
if features['reward_last'] < threshold:
    prune()
```

#### 3. 数据格式

```python
# ❌ 错误: 包含当前step的reward
branch_data = {
    'reward_history': [0.5, 0.6, 0.7, 0.8],  # step 0-3
}
# 在step 3决策时使用 - 错误!step 3的reward(0.8)还不存在

# ✅ 正确: 只包含历史reward
branch_data = {
    'reward_history': [0.5, 0.6, 0.7],  # 只有step 0-2
}
# 在step 3决策时使用 - 正确!只用已知的历史
```

## 影响范围

### 受影响的文件

1. ✅ `dynamic_pruner.py` - 核心剪枝器
2. ✅ `USAGE_SUMMARY.md` - 使用文档
3. ✅ `IMPORTANT_FIX.md` - 详细修正说明
4. ✅ `example_correct_usage.py` - 正确使用示例

### 需要更新的代码

所有使用该剪枝器的代码都需要确保:

```python
# ✅ 正确的调用方式
for step in range(max_steps):
    # 1. 先决策 (基于历史数据)
    for branch in branches:
        should_prune = pruner.should_prune_branch(
            branch_data={
                'reward_history': branch['rewards'][:step],  # 只到step-1
                ...
            },
            current_step=step
        )
    
    # 2. 再生成 (对保留的分支)
    for branch in kept_branches:
        new_reward = generate_and_compute_reward(branch)
        branch['rewards'].append(new_reward)  # 现在才有step的reward
```

## 验证测试

### 测试1: 基本功能
```bash
python3 dynamic_pruner.py
```
✅ 通过 - 使用`reward_last`正常工作

### 测试2: 正确使用示例
```bash
python3 example_correct_usage.py
```
✅ 通过 - 展示正确的调用顺序

### 测试3: 完整测试
```bash
python3 test_pruner.py
```
⚠️ 需要重新运行 - 确保使用历史数据

## 阈值调整建议

由于现在使用历史reward(会"滞后一步"),建议放宽阈值:

```json
{
  "adaptive_strategy": {
    "base_thresholds": {
      "reward_relative": {
        "early_stage": 0.3,   // 放宽 (原0.5)
        "mid_stage": 0.6,     // 放宽 (原0.7)
        "late_stage": 0.8     // 放宽 (原0.85)
      },
      "reward_absolute": {
        "min_threshold": 0.30,  // 放宽 (原0.35)
        "safe_threshold": 0.35  // 放宽 (原0.39)
      }
    }
  }
}
```

## 优势

修正后的实现:

1. ✅ **可用性**: 现在可以在真实运行时使用
2. ✅ **正确性**: 不依赖"未来数据"
3. ✅ **一致性**: 符合实际beam search流程
4. ✅ **效率**: 先剪枝再计算,节省资源

## 注意事项

### 早期阶段

在step 0或step 1时:
- `reward_history`为空或很短
- 特征`reward_last`可能为0
- 建议早期阶段使用更宽松的策略

### 调用时机

```python
# ✅ 正确时机
for step in range(max_steps):
    prune_decision()  # 基于历史
    generate_token()  # 对保留的
    compute_reward()  # 现在才计算

# ❌ 错误时机
for step in range(max_steps):
    generate_token()
    compute_reward()
    prune_decision()  # 太晚了!资源已浪费
```

## 文档更新

已更新的文档:
- ✅ `IMPORTANT_FIX.md` - 详细修正说明
- ✅ `USAGE_SUMMARY.md` - 使用指南
- ✅ `example_correct_usage.py` - 代码示例
- ✅ `FIX_SUMMARY.md` - 本文件

## 检查清单

使用该剪枝器前,请确认:

- [ ] 理解"历史reward"vs"当前reward"的区别
- [ ] 在生成token**之前**调用剪枝决策
- [ ] `reward_history`只包含已计算的reward
- [ ] 不包含当前step的reward
- [ ] 对保留的分支才计算新reward
- [ ] 阅读了`example_correct_usage.py`

## 快速参考

### 正确的数据流

```
Step N开始
  ↓
判断剪枝 (基于step 0 到 N-1 的历史)
  ↓
保留/剪枝分支
  ↓
对保留的分支:
  生成token
  ↓
  计算reward (现在才有step N的reward)
  ↓
  添加到历史
  ↓
Step N+1开始 (使用step 0 到 N 的历史)
```

### 关键代码

```python
# 数据准备
branch_data = {
    'reward_history': branch['rewards'][:current_step],  # 不含current_step
}

# 调用剪枝
should_prune, reason = pruner.should_prune_branch(
    branch_data, all_branches, current_step
)

# 生成和计算 (只对保留的)
if not should_prune:
    new_reward = compute_reward(branch)
    branch['rewards'].append(new_reward)
```

---

## 相关文件

- 📄 `IMPORTANT_FIX.md` - 详细技术说明
- 📄 `example_correct_usage.py` - 完整代码示例
- 📄 `USAGE_SUMMARY.md` - 使用指南
- 📄 `dynamic_pruner.py` - 修正后的实现

---

**更新**: 2026-04-01  
**作者**: Assistant  
**状态**: ✅ 修复完成
# 重要修正: 只使用历史Reward

## 问题

原始实现错误地使用了"当前step的reward"(`reward_current`),但在运行时做剪枝决策时,**当前step的reward还没有计算出来**,因为:

1. Token还未生成
2. PRM模型需要在生成后才能评估reward

## 修正

**修改后的实现只使用历史reward**:

```python
# ❌ 错误: 使用当前step的reward (还不存在)
features['reward_current'] = rewards[-1]

# ✅ 正确: 使用历史最后一步的reward (已计算)
features['reward_last'] = rewards[-1]  # 这是上一步已完成的reward
```

## 关键变更

### 1. 特征提取 (`_extract_runtime_features`)

**之前**:
- 使用 `reward_current` - 假设当前step的reward已知

**现在**:
- 使用 `reward_last` - 只使用历史最后一步的reward
- 添加 `reward_volatility` - reward的波动性
- 添加 `history_length` - 历史长度

### 2. 决策规则

所有决策都基于 `reward_last` 而非 `reward_current`:

```python
# 规则1: 绝对reward阈值
if features['reward_last'] < min_threshold:
    prune()

# 规则2: 安全阈值
if features['reward_last'] > safe_threshold:
    keep()

# 规则3: 相对reward
if features['reward_last'] < max_reward * threshold:
    prune()
```

### 3. 综合得分

特征权重映射:
```python
feature_mapping = {
    'reward_current': 'reward_last',  # 映射到历史
    'reward_mean': 'reward_mean',
    'token_prob_mean': 'token_prob_mean',
    'cum_prob': 'cum_prob',
    'reward_trend': 'reward_trend'
}
```

## 使用场景

### 正确的调用时机

```python
# 在生成当前step之前调用剪枝决策
for step in range(max_steps):
    # 1. 检查是否应该剪枝 (基于历史数据)
    kept_branches = []
    for branch in current_branches:
        # branch['reward_history'] 只包含已完成步骤的reward
        # 不包括即将生成的step
        should_prune = pruner.should_prune_branch(
            branch_data={
                'reward_history': branch['rewards'][:step],  # 只到step-1
                'token_prob_history': branch['token_probs'][:step],
                'prob_history': branch['cum_probs'][:step]
            },
            all_branches=...,
            current_step=step
        )
        
        if not should_prune:
            kept_branches.append(branch)
    
    # 2. 对保留的分支生成下一个token
    for branch in kept_branches:
        generate_next_token(branch)
        compute_reward(branch)  # 现在才计算step的reward
        branch['rewards'].append(new_reward)
```

### 数据格式

```python
branch_data = {
    # 历史reward: [step0的reward, step1的reward, ..., step(n-1)的reward]
    # 不包括step n的reward (还未计算)
    'reward_history': [0.5, 0.6, 0.7],  # 3步历史
    
    # 历史token概率
    'token_prob_history': [
        [0.9, 0.95, ...],  # step 0的token概率
        [0.92, 0.96, ...], # step 1的token概率
        [0.94, 0.97, ...]  # step 2的token概率
    ],
    
    # 历史累积概率
    'prob_history': [0.9, 0.92, 0.94]
}

# 当前正在决策是否继续step 3
# reward_history[-1] = 0.7 是step 2的reward (已知)
# step 3的reward还未计算,所以不在历史中
```

## 影响

### 1. 特征有效性

使用历史reward意味着:
- ✅ 可以基于已知信息做决策
- ✅ 更符合实际运行时场景
- ⚠️ 决策会"滞后一步"(基于上一步而非当前步)

### 2. 阈值调整

由于使用历史reward,可能需要调整阈值:
- 历史reward可能略低于"当前最终"reward
- 建议稍微放宽阈值以补偿滞后

### 3. 早期阶段

在第0步或第1步时:
- `reward_history`可能为空或很短
- 需要更保守的策略
- 建议早期阶段不剪枝或使用更宽松阈值

## 测试验证

修改后需要重新测试:

```bash
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold

# 测试基本功能
python3 dynamic_pruner.py

# 重新运行完整测试
python3 test_pruner.py
```

## 建议配置

考虑到使用历史reward,建议调整配置:

```json
{
  "adaptive_strategy": {
    "base_thresholds": {
      "reward_relative": {
        "early_stage": 0.3,   // 放宽(从0.5)
        "mid_stage": 0.6,     // 放宽(从0.7)
        "late_stage": 0.8     // 放宽(从0.85)
      },
      "reward_absolute": {
        "min_threshold": 0.30,  // 放宽(从0.35)
        "safe_threshold": 0.35  // 放宽(从0.39)
      }
    }
  }
}
```

## 总结

这个修正是**关键性的**:
- ✅ 修复了使用"未来数据"的逻辑错误
- ✅ 使系统可以在真实运行时使用
- ✅ 更符合实际beam search流程

**所有使用该系统的代码都需要确保**:
1. `reward_history`只包含已计算的reward
2. 在生成token之前调用剪枝决策
3. 在计算reward之后更新历史

---

**更新时间**: 2026-04-01  
**严重程度**: 🔴 Critical  
**影响**: 所有使用该剪枝器的代码
