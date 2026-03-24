# 极端情况分析功能说明

## 功能概述

新增的**极端情况分析**功能用于识别和统计在beam search过程中，当某个branch的token数量**明显偏离**其他branches平均值时，模型选择该branch的概率。

## 极端值定义

### 检测标准

使用**阈值倍数**（threshold）来判定是否为极端情况：

- **极端大 (Extreme LARGE)**：`branch_tokens >= threshold × avg(other_branches)`
- **极端小 (Extreme SMALL)**：`branch_tokens <= avg(other_branches) / threshold`

默认阈值为 **1.5**，可通过命令行参数调整。

### 示例

假设有4个branches的token数量为：`[10, 20, 25, 80]`，阈值为1.5：

1. 检查branch 3 (80 tokens)是否为极端大：
   - 其他branches平均值：(10 + 20 + 25) / 3 = 18.33
   - 80 >= 1.5 × 18.33 = 27.5？ **是** ✅
   - 结论：branch 3 是**极端大**

2. 检查branch 0 (10 tokens)是否为极端小：
   - 其他branches平均值：(20 + 25 + 80) / 3 = 41.67
   - 10 <= 41.67 / 1.5 = 27.78？ **是** ✅
   - 结论：branch 0 是**极端小**

## 统计指标

对于每个配置（如256_2_1, 256_4_1, 256_8_1），分别统计：

### 1. 极端大的branch

- `total_cases`: 存在极端大branch的决策总数
- `selected_count`: 极端大branch被选中的次数
- `selection_rate`: 选择率 = selected_count / total_cases × 100%
- `sample_cases`: 前5个样例（用于验证）

### 2. 极端小的branch

- `total_cases`: 存在极端小branch的决策总数
- `selected_count`: 极端小branch被选中的次数
- `selection_rate`: 选择率 = selected_count / total_cases × 100%
- `sample_cases`: 前5个样例（用于验证）

### 3. 同时存在两个极端的情况

- `total_cases`: 同时存在极端大和极端小branch的决策数
- `selected_extreme_count`: 选择了任一极端branch的次数
- `selection_rate`: 选择率

## 关键发现

### 发现1: 不同Reward Model对极端值的偏好完全不同

**Skywork-PRM** 倾向选择**极端大**的branch：
```
配置256_8_1，8个branches时：
- 极端大选择率: 22.71% (远高于随机概率12.5%)
- 极端小选择率: 16.21%
- 倾向：明显偏好token多的branch
```

**Qwen2.5-Math-PRM** 倾向选择**极端小**的branch：
```
配置256_8_1，8个branches时：
- 极端大选择率: 7.14% (远低于随机概率12.5%)
- 极端小选择率: 17.05%
- 倾向：明显偏好token少的branch
```

### 发现2: Branch数量影响极端值选择的绝对概率

以Skywork-PRM为例：

| Branches | 极端大选择率 | 极端小选择率 | Large/Small比率 |
|----------|------------|------------|----------------|
| 2        | 66.10%     | 33.90%     | 1.95           |
| 4        | 40.20%     | 21.21%     | 1.90           |
| 8        | 22.71%     | 16.21%     | 1.40           |

**观察**：
- 绝对选择率随branch数量增加而降低
- 但相对偏好（比率）保持稳定
- 说明模型的"偏好"是内在的，不受branch数量影响

### 发现3: 极端情况非常普遍

- 2 branches: ~70% 的决策存在极端情况
- 4 branches: ~70-80% 的决策存在极端情况
- 8 branches: ~90% 的决策存在极端情况

这说明在实际beam search中，不同branches生成的token数量差异是**常态**而非**例外**。

## 使用方法

### 基本用法

```bash
# 使用默认阈值1.5
python analyze_branch_selection.py

# 使用自定义阈值2.0（更严格的极端标准）
python analyze_branch_selection.py --extreme-threshold 2.0

# 详细输出
python analyze_branch_selection.py --verbose
```

### 查看结果

```bash
# 查看JSON结果
cat branch_selection_analysis.json

# 使用测试脚本进行对比分析
python test_extreme_analysis.py
```

### JSON结构示例

```json
{
  "by_configuration": {
    "DATASET/POLICY/REWARD": {
      "256_8_1": {
        "extreme_branch_analysis": {
          "threshold": 1.5,
          "extreme_large": {
            "total_cases": 252,
            "selected_count": 18,
            "selection_rate": 7.14,
            "sample_cases": [
              {
                "branch_tokens": [10, 15, 12, 50, 14, 11, 13, 16],
                "selected_index": 3,
                "selected_tokens": 50
              }
            ]
          },
          "extreme_small": {
            "total_cases": 264,
            "selected_count": 45,
            "selection_rate": 17.05,
            "sample_cases": [...]
          },
          "both_extremes": {
            "total_cases": 248,
            "selected_extreme_count": 63,
            "selection_rate": 25.40
          }
        }
      }
    }
  }
}
```

## 实践应用

### 1. 预测模型行为

根据极端情况选择率，可以预测：
- 当生成的branches长度差异很大时，模型会选择哪一个
- 不同reward model在极端情况下的行为差异

### 2. 优化beam search参数

- 如果希望避免极端长的生成，使用Qwen2.5-Math-PRM
- 如果希望保留详细推理，使用Skywork-PRM
- 根据极端值选择倾向调整beam width

### 3. 解释异常结果

- 为什么某些生成特别长？→ 可能选择了极端大的branch
- 为什么某些生成特别短？→ 可能选择了极端小的branch
- 通过极端分析找到根本原因

## 技术细节

### 算法实现

```python
def detect_extreme_branches(branch_tokens, threshold=1.5):
    for i, tokens in enumerate(branch_tokens):
        # 计算除当前branch外的平均值
        other_avg = mean([t for j, t in enumerate(branch_tokens) if j != i])
        
        # 检查是否极端大
        if tokens >= threshold * other_avg:
            return i, 'large'
        
        # 检查是否极端小
        if tokens <= other_avg / threshold:
            return i, 'small'
    
    return None, None
```

### 特殊处理：2个branches的情况

对于只有2个branches的情况，直接比较两个值的比率：
- 如果 `max/min >= threshold`，则认为存在极端情况
- 大的那个是极端大，小的那个是极端小

## 参数调优建议

### 阈值选择

- **1.5** (默认): 适合大多数场景，能识别明显的差异
- **2.0**: 更严格，只关注最极端的情况
- **1.2**: 更宽松，捕获更多轻微差异

### 选择标准

根据你的需求选择合适的阈值：
- **研究模型偏好** → 使用1.5（平衡）
- **识别异常情况** → 使用2.0（严格）
- **全面分析差异** → 使用1.2（宽松）

## 相关文件

- `analyze_branch_selection.py`: 主分析脚本
- `branch_selection_analysis.json`: 分析结果
- `test_extreme_analysis.py`: 测试和演示脚本
- `README.md`: 完整分析报告

## 更新日志

- **2026-03-15**: 新增极端情况分析功能
  - 支持自定义阈值
  - 统计极端大/小branch的选择率
  - 提供样例数据用于验证
  - 生成详细的JSON报告
