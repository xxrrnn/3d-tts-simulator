# 动态Threshold剪枝系统 - 使用总结

## 概述

本系统实现了一个基于运行时特征的动态分支剪枝策略,能够:
- ✅ 在运行时根据可观测特征(reward、probability等)动态决定是否剪枝
- ✅ 隐式地保护正确分支,不需要显式的题目标签
- ✅ 有效剪枝错误分支,减少计算开销

## 测试结果

在AIME24_beam_search数据集(16384_8_2)上的测试结果:

### 正确题目(5题)
- **最优分支保留率**: 60% (3/5)
- **总体分支保留率**: 60% (6/10)
- 说明:能够保留大部分正确题目的正确分支

### 错误题目(25题)
- **分支剪枝率**: 88% (44/50)
- 说明:能够有效剪枝错误题目的错误分支

### 总体
- **总剪枝率**: 80% (48/60)
- **计算节省**: 通过剪枝80%的分支,可以显著减少计算开销

## 核心策略

### 1. 多层次决策规则

#### 规则1: 绝对Reward阈值
- **低于最小阈值(0.3499)**: 直接剪枝
- **高于安全阈值(0.3923)**: 一定保留
- 基于训练数据统计得出

#### 规则2: 相对Reward + 排名
- 考虑分支相对于其他分支的表现
- 不同阶段使用不同阈值:
  - 早期(0-30%): 相对阈值 0.5 (宽松)
  - 中期(30-70%): 相对阈值 0.7
  - 晚期(70-100%): 相对阈值 0.85 (严格)
- 排名后30%的分支更容易被剪枝

#### 规则3: 综合得分
- 加权多个特征:
  - reward_current (40%)
  - reward_mean (20%)
  - token_prob_mean (15%)
  - cum_prob (15%)
  - reward_trend (10%)
- 综合得分低于阈值(0.464)时剪枝

### 2. 关键特征

根据特征分析,最重要的区分特征(按效应量排序):

1. **reward_min** (Cohen's d=3.30)
   - 正确题: 0.364, 错误题: 0.089
   
2. **reward_mean** (Cohen's d=3.25)
   - 正确题: 0.473, 错误题: 0.157

3. **reward_max** (Cohen's d=1.97)
   - 正确题: 0.608, 错误题: 0.277

4. **max_steps** (Cohen's d=1.85)
   - 正确题: 46.6步, 错误题: 16.2步

5. **reward_current** (Cohen's d=1.55)
   - 正确题: 0.527, 错误题: 0.235

## 使用方法

### Step 1: 导入剪枝器

```python
import sys
sys.path.append('/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process')

from threshold.dynamic_pruner import DynamicPruner
```

### Step 2: 初始化

```python
pruner = DynamicPruner(
    "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold/threshold_model.json"
)
```

### Step 3: 在Beam Search中使用

```python
# 在beam search的每一步
for step in range(max_steps):
    # ... 生成候选分支 ...
    
    # 对每个分支进行剪枝判断
    kept_branches = []
    for branch in current_branches:
        branch_data = {
            'reward_history': branch['rewards'],
            'token_prob_history': branch['token_probs'],
            'prob_history': branch['cum_probs']
        }
        
        should_prune, reason = pruner.should_prune_branch(
            branch_data=branch_data,
            all_branches=[{
                'reward_history': b['rewards'],
                'token_prob_history': b['token_probs'],
                'prob_history': b['cum_probs']
            } for b in current_branches],
            current_step=step,
            max_steps=max_steps
        )
        
        if not should_prune:
            kept_branches.append(branch)
    
    current_branches = kept_branches
    # ... 继续beam search ...
```

## 数据格式要求

### ⚠️ 重要: 只使用历史Reward

**关键点**: 在运行时做剪枝决策时,当前step的reward还**未计算**,因此只能使用历史reward。

```python
# ❌ 错误: 包含当前step的reward (还不存在)
branch_data = {
    'reward_history': [0.5, 0.6, 0.7, 0.8],  # step 0-3
}
# 在step 3决策时,step 3的reward(0.8)还未计算!

# ✅ 正确: 只包含历史reward
branch_data = {
    'reward_history': [0.5, 0.6, 0.7],  # 只有step 0-2
}
# 在step 3决策时,只使用step 0-2的reward
```

### 输入格式

```python
branch_data = {
    'reward_history': [0.5, 0.6, 0.7],  # List[float], 每步的reward值 (不含当前step)
    'token_prob_history': [              # List[List[float]], 每步的token概率
        [0.9, 0.95, ...],                # 第1步的所有token概率
        [0.92, 0.96, ...],               # 第2步的所有token概率
        [0.94, 0.97, ...]                # 第3步的所有token概率
    ],
    'prob_history': [0.9, 0.92, 0.94]   # List[float], 累积概率
}

all_branches = [branch_data_1, branch_data_2, ...]  # 所有分支的数据

# 注意: 在step N做决策时
# - reward_history 应该只包含 step 0 到 step N-1 的reward
# - step N 的reward还未计算,不应该包含在内
```

### 输出格式

```python
should_prune: bool  # True表示应该剪枝该分支
reason: str         # 剪枝或保留的原因,如:
                    # - "high_absolute_reward (last_reward=0.8 > 0.392)"
                    # - "low_absolute_reward (last_reward=0.2 < 0.350)"
                    # - "low_relative_reward (rank=1.0, last_reward=0.3 < 0.5*0.85)"
```

### 正确的使用时机

```python
# ✅ 正确: 在生成token之前做剪枝决策
for step in range(max_steps):
    # 1. 基于历史数据决定是否剪枝
    kept_branches = []
    for branch in current_branches:
        should_prune, reason = pruner.should_prune_branch(
            branch_data={
                # 只包含已完成步骤的数据 (0 到 step-1)
                'reward_history': branch['rewards'][:step],
                'token_prob_history': branch['token_probs'][:step],
                'prob_history': branch['cum_probs'][:step]
            },
            all_branches=[...],
            current_step=step
        )
        if not should_prune:
            kept_branches.append(branch)
    
    # 2. 对保留的分支生成token和计算reward
    for branch in kept_branches:
        generate_next_token(branch)
        new_reward = compute_reward(branch)  # 现在才计算step的reward
        branch['rewards'].append(new_reward)
```

## 配置和调优

### 调整剪枝严格程度

编辑 `threshold_model.json`:

```json
{
  "adaptive_strategy": {
    "base_thresholds": {
      "reward_relative": {
        "early_stage": 0.5,   // 增大 -> 更严格 (剪枝更多)
        "mid_stage": 0.7,     // 减小 -> 更宽松 (保留更多)
        "late_stage": 0.85
      },
      "reward_absolute": {
        "min_threshold": 0.35,  // 增大 -> 更严格
        "safe_threshold": 0.39  // 减小 -> 更宽松
      }
    }
  }
}
```

### 调整特征权重

```json
{
  "adaptive_strategy": {
    "feature_weights": {
      "reward_current": 0.4,      // 当前reward权重
      "reward_mean": 0.2,         // 平均reward权重
      "token_prob_mean": 0.15,    // token概率权重
      "cum_prob": 0.15,           // 累积概率权重
      "reward_trend": 0.1         // reward趋势权重
    }
  }
}
```

## 文件说明

```
threshold/
├── README.md                   # 系统说明文档
├── USAGE_SUMMARY.md            # 本文件:使用总结
├── analyze_features.py         # 特征分析脚本
├── learn_threshold.py          # 学习threshold策略
├── dynamic_pruner.py           # 运行时剪枝器 ⭐核心模块
├── test_pruner.py              # 测试剪枝器效果
├── integration_example.py      # 集成示例代码
├── feature_analysis.json       # 特征分析结果
├── threshold_model.json        # Threshold模型 ⭐核心配置
└── test_results.json           # 测试结果
```

## 优势和局限

### 优势

1. **不依赖题目标签**: 完全基于运行时特征,可以泛化到新题目
2. **动态自适应**: 根据不同阶段使用不同策略
3. **多维度决策**: 综合reward、probability、排名等多个维度
4. **易于集成**: 接口简单,只需几行代码即可集成到现有系统
5. **可调节**: 通过配置文件灵活调整剪枝策略

### 局限

1. **最优分支保留率60%**: 
   - 原因:题目3和题目18的最优分支reward较低,被误剪
   - 改进方向:放宽早期阶段的阈值,或增加保护机制

2. **依赖reward质量**: 
   - 如果PRM模型的reward不准确,剪枝效果会受影响
   - 建议:使用高质量的PRM模型

3. **需要训练数据**: 
   - 当前模型基于AIME24数据训练
   - 在其他数据集上可能需要重新训练阈值

## 改进建议

### 短期改进

1. **放宽早期阈值**: 将`early_stage`从0.5改为0.3,减少误剪
2. **添加最小保留数**: 确保每道题至少保留1-2个分支
3. **记录被剪分支**: 用于事后分析和模型改进

### 长期改进

1. **在线学习**: 根据实际结果动态调整阈值
2. **题目难度感知**: 为不同难度的题目使用不同策略
3. **集成更多特征**: 如文本长度、步骤复杂度等
4. **多模型ensemble**: 结合多个剪枝策略的投票

## 快速开始

```bash
# 1. 运行特征分析
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold
python3 analyze_features.py

# 2. 学习threshold模型
python3 learn_threshold.py

# 3. 测试剪枝器效果
python3 test_pruner.py

# 4. 查看集成示例
python3 integration_example.py
```

## 联系和反馈

如有问题或建议,请查看:
- 详细文档: `README.md`
- 集成示例: `integration_example.py`
- 测试代码: `test_pruner.py`
