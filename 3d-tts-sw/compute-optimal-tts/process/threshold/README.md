# Dynamic Pruning Threshold System

动态剪枝threshold系统 - 基于运行时特征自动剪枝beam search分支

## ⚠️ 重要提示

**只使用历史reward** - 在做剪枝决策时，当前step的reward还未计算，只能使用历史数据。详见 [FIX.md](FIX.md)

## 快速开始

```python
from dynamic_pruner import DynamicPruner

# 初始化
pruner = DynamicPruner("threshold_model.json")

# 使用 (注意: reward_history不含当前step)
should_prune, reason = pruner.should_prune_branch(
    branch_data={'reward_history': [...], ...},
    all_branches=[...],
    current_step=step
)
```

## 文件说明

### 核心文件 ⭐
- `dynamic_pruner.py` - 动态剪枝器（主程序）
- `threshold_model.json` - Threshold配置
- `examples.py` - 使用示例代码

### 工具脚本
- `analyze_features.py` - 特征分析
- `learn_threshold.py` - 学习threshold策略
- `test_pruner.py` - 测试评估

### 文档
- `README.md` - 本文件
- `USAGE.md` - 详细使用指南
- `FIX.md` - 重要修正说明

### 数据文件
- `feature_analysis.json` - 特征分析结果
- `test_results.json` - 测试结果
- `correct_questions.json` - 正确题目索引

## 核心功能

1. **动态Threshold** - 根据运行时特征自动调整剪枝阈值
2. **隐式保护** - 通过特征模式区分正确/错误分支，不需要题目标签
3. **多层决策** - 绝对阈值、相对阈值、综合得分多维判断
4. **阶段自适应** - 早期宽松(0.5)、中期中等(0.7)、晚期严格(0.85)

## 测试结果

在AIME24数据集上:
- **总剪枝率**: 80% (节省80%计算)
- **正确分支保留率**: 60%
- **错误分支剪枝率**: 88%

## 使用流程

```bash
# 1. 分析特征 (可选)
python analyze_features.py

# 2. 学习threshold (可选)
python learn_threshold.py

# 3. 查看示例
python examples.py

# 4. 测试效果
python test_pruner.py
```

## 关键要点

1. ✅ 在生成token**之前**调用剪枝决策
2. ✅ `reward_history`只包含已计算的reward（不含当前step）
3. ✅ 对保留的分支才生成token和计算reward
4. ✅ 将新reward添加到历史中

详细说明请参考 [USAGE.md](USAGE.md)

## 配置调整

编辑 `threshold_model.json`:

```json
{
  "base_thresholds": {
    "reward_relative": {
      "early_stage": 0.5,   // 增大->更严格
      "mid_stage": 0.7,
      "late_stage": 0.85
    },
    "reward_absolute": {
      "min_threshold": 0.35,  // 增大->更严格
      "safe_threshold": 0.39
    }
  }
}
```

---

**版本**: v1.0  
**状态**: ✅ Production Ready
