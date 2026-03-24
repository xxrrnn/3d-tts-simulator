# Branch Selection Analysis - 按配置分组分析

## 📊 更新说明

**最新功能 (2026-03-15)**：新增**极端情况分析** (Extreme Branch Analysis)

- 分析当branch中存在token数量**明显偏离**其他branches平均值的情况
- 统计这些极端branch被选择的概率
- 默认阈值：1.5x（可通过`--extreme-threshold`参数调整）

**重要**：本次分析已按**配置参数**（如 256_2_1, 256_4_1, 256_8_1）分组统计。

配置参数含义：
- `256_2_1` → max_step=256, beam_size=2, num_beams=1
- `256_4_1` → max_step=256, beam_size=4, num_beams=1  
- `256_8_1` → max_step=256, beam_size=8, num_beams=1
- `60_8_1` → max_step=60, beam_size=8, num_beams=1

---

## 🔥 极端情况分析 (NEW!)

### 什么是"极端branch"？

**极端大 (Extreme LARGE)**：某个branch的token数量 ≥ 1.5x 其他branches的平均值  
**极端小 (Extreme SMALL)**：某个branch的token数量 ≤ (1/1.5)x 其他branches的平均值

### 核心发现

#### 1. **Qwen2.5-Math-PRM 更倾向选择极端小的branch**

**示例：AIME24/Llama-3.1-8B/Qwen2.5-Math-1.5B**

| Config | Avg Branches | Extreme Large Rate | Extreme Small Rate | 倾向 |
|--------|-------------|-------------------|-------------------|------|
| 256_2_1 | 2.00 | **45.18%** | **54.82%** | ⬇️ 偏向Small |
| 256_4_1 | 3.60 | **17.95%** | **23.65%** | ⬇️ 偏向Small |
| 256_8_1 | 6.43 | **7.14%** | **17.05%** | ⬇️ **强烈偏向Small** |

**关键观察**：
- ✅ Branch数量越多，极端branch被选择的概率越低
- ✅ 但**极端小的branch相对更容易被选择**（17.05% vs 7.14%）
- ✅ 当只有2个branches时，如果存在极端情况，选择率接近50%（因为只有两个选择）

#### 2. **Skywork-PRM 更倾向选择极端大的branch**

**示例：AIME24/Llama-3.1-8B/Skywork-PRM-1.5B**

| Config | Avg Branches | Extreme Large Rate | Extreme Small Rate | 倾向 |
|--------|-------------|-------------------|-------------------|------|
| 256_2_1 | 2.00 | **66.10%** | **33.90%** | ⬆️ **强烈偏向Large** |
| 256_4_1 | 3.35 | **40.20%** | **21.21%** | ⬆️ 偏向Large |
| 256_8_1 | 5.73 | **22.71%** | **16.21%** | ⬆️ 偏向Large |

**关键观察**：
- ✅ Skywork-PRM **明显倾向选择token数量多的极端branch**
- ✅ 即使在8个branches的情况下，极端大的branch选择率仍然是22.71%，远高于随机概率（~12.5%）
- ✅ 2个branches时，极端大branch的选择率高达66%！

#### 3. **极端情况在不同配置下的普遍性**

从数据来看：
- **2 branches配置**：约70%的决策存在极端情况（因为只有两个值，差异明显）
- **4 branches配置**：约70-80%的决策存在极端情况
- **8 branches配置**：约90%的决策存在极端情况

这说明在实际的beam search过程中，**不同branches生成的token数量差异是普遍存在的**。

---

## 🔍 关键发现

### 1. **Beam Size 对选择倾向的影响最显著**

通过对比不同beam size配置，发现了一个**非常重要的规律**：

#### 示例：AIME24/Llama-3.1-8B/Qwen2.5-Math-1.5B

| 配置 | Avg Branches | Select MAX | Select MIN | 倾向 |
|------|-------------|------------|------------|------|
| 256_2_1 (2 branches) | 2.00 | 48.19% | **51.81%** | ⚖️ 平衡 |
| 256_4_1 (4 branches) | 3.60 | 23.49% | **31.67%** | ⬇️ MIN |
| 256_8_1 (8 branches) | 6.43 | 16.42% | **31.39%** | ⬇️ **强MIN** |

**观察**：
- ✅ **Branch数量增加时，选择MAX的比例急剧下降** (48% → 23% → 16%)
- ✅ **选择MIN的比例保持相对稳定** (52% → 32% → 31%)
- ✅ **选择OTHER（中间值）的比例大幅上升** (0% → 20% → 35%)

#### 示例：AIME24/Llama-3.1-8B/Skywork-PRM-1.5B

| 配置 | Avg Branches | Select MAX | Select MIN | 倾向 |
|------|-------------|------------|------------|------|
| 256_2_1 (2 branches) | 2.00 | **57.14%** | 42.86% | ⬆️ **强MAX** |
| 256_4_1 (4 branches) | 3.35 | **41.15%** | 20.80% | ⬆️ MAX |
| 256_8_1 (8 branches) | 5.73 | **27.11%** | 21.25% | ⬆️ MAX (减弱) |

**观察**：
- ✅ **Branch数量增加时，MAX倾向减弱但仍保持** (57% → 41% → 27%)
- ✅ **选择分布变得更加均匀**

---

### 2. **Reward Model + Branch Size 的交互效应**

#### A. **Skywork-PRM系列** → 在所有Branch配置下都倾向MAX

**规律**：
- 2个branches时：MAX比例 **50-57%** (非常强)
- 4个branches时：MAX比例 **36-42%** (强)
- 8个branches时：MAX比例 **21-34%** (中等)

**即使在8个branches的情况下，Skywork-PRM仍然倾向选择较多token的分支！**

#### B. **Qwen2.5-Math系列** → Branch数量越多，MIN倾向越强

**规律**：
- 2个branches时：MAX vs MIN 基本平衡 (48% vs 52%)
- 4个branches时：MIN开始占优 (23% vs 32%)
- 8个branches时：MIN明显占优 (16% vs 31%)

#### C. **math-shepherd-mistral-7b-prm** → 高度依赖Branch数量

**规律**：
- 2个branches时：基本平衡 (50% vs 50%)
- 4个branches时：略倾向MIN (28-37%)
- 8个branches时：选择变得更分散

---

### 3. **平均Branch数量 vs 实际Branch数量**

配置的beam_size并不等于实际的average branches：

| 配置 | 设定beam_size | 实际avg_branches | 说明 |
|------|--------------|------------------|------|
| 256_2_1 | 2 | 2.00 | ✅ 完全一致 |
| 256_4_1 | 4 | 3.35-3.75 | ⚠️ 实际偏少 |
| 256_8_1 | 8 | 5.73-6.65 | ⚠️ 实际明显偏少 |

**原因**：某些step可能生成的branch数量少于设定值。

---

## 📈 核心结论

### 1. **Branch Size是最重要的影响因素**

**Branch数量越多**：
- ✅ 选择MAX的比例大幅下降
- ✅ 选择分布更加分散（OTHER比例上升）
- ✅ 模型的选择策略变得更加"保守"

### 2. **Reward Model决定基本倾向性**

- **Skywork-PRM** → 基线倾向MAX，但受branch数量影响
- **Qwen2.5-Math-PRM** → 基线倾向MIN，branch越多倾向越强
- **math-shepherd** → 基线平衡，受branch数量影响大

### 3. **实际应用建议**

#### 如果你想要**更长、更详细的推理**：
```
✅ 使用 Skywork-PRM 
✅ 使用较少的beam_size (2-4)
✅ 结果：更倾向选择token数量多的分支
```

#### 如果你想要**简洁、高效的推理**：
```
✅ 使用 Qwen2.5-Math-PRM
✅ 使用较多的beam_size (4-8)
✅ 结果：更倾向选择token数量少的分支
```

#### 如果你想要**平衡的选择**：
```
✅ 使用 math-shepherd-PRM
✅ 使用中等的beam_size (4)
✅ 结果：在MAX和MIN之间平衡
```

---

## 📊 具体数据示例

### MATH数据集 - Llama-3.1-8B + Qwen2.5-Math-1.5B-Instruct (PRM)

| Config | Decisions | Avg Branches | MAX% | MIN% | Avg Tokens |
|--------|-----------|--------------|------|------|------------|
| 256_2_1 | 3,746 | 2.00 | 28.96% | **45.73%** | 32.42 |
| 256_4_1 | 3,745 | 3.71 | 27.86% | **44.17%** | 31.65 |
| 256_8_1 | 3,709 | 6.30 | **31.35%** | 45.65% | 34.04 |

**观察**：MIN倾向在所有配置下都很强（44-46%），非常稳定！

### AIME24数据集 - Qwen2.5-3B + math-shepherd-mistral-7b-prm

| Config | Decisions | Avg Branches | MAX% | MIN% | Avg Tokens |
|--------|-----------|--------------|------|------|------------|
| 256_2_1 | 319 | 2.00 | 50.16% | 49.84% | 91.24 |
| 256_4_1 | 375 | 3.61 | 27.47% | **36.27%** | 73.25 |
| 256_8_1 | 459 | 6.27 | 22.00% | 24.84% | 69.34 |

**观察**：从平衡(2 branches) → MIN倾向(4 branches) → 更分散(8 branches)

---

## 🔧 使用方法

```bash
# 运行分析（按配置分组，包含极端情况分析）
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/analyze
python analyze_branch_selection.py --verbose

# 使用自定义极端值阈值（默认1.5）
python analyze_branch_selection.py --extreme-threshold 2.0 --verbose

# 查看结果
cat branch_selection_analysis.json
```

### 命令行参数

- `--workload-dir`: Workload目录路径（默认：`/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sim/model_workloads`）
- `--output`: 输出JSON文件路径（默认：`branch_selection_analysis.json`）
- `--extreme-threshold`: 极端值阈值倍数（默认：1.5）
  - 值越大，判定为"极端"的标准越严格
  - 例如：2.0 表示某个branch的token数量需要是其他平均值的2倍以上才算极端
- `--verbose` / `-v`: 详细输出

### JSON文件结构

```json
{
  "by_configuration": {
    "DATASET/POLICY_MODEL/REWARD_MODEL": {
      "256_2_1": {
        "total_decisions": 276,
        "avg_branches_per_decision": 2.0,
        "selection_distribution": { ... },
        "extreme_branch_analysis": {
          "threshold": 1.5,
          "extreme_large": {
            "total_cases": 197,
            "selected_count": 89,
            "selection_rate": 45.18,
            "sample_cases": [...]
          },
          "extreme_small": {
            "total_cases": 197,
            "selected_count": 108,
            "selection_rate": 54.82,
            "sample_cases": [...]
          },
          "both_extremes": {
            "total_cases": 197,
            "selected_extreme_count": 197,
            "selection_rate": 100.0
          }
        }
      }
    }
  },
  "overall": {
    "DATASET/POLICY_MODEL/REWARD_MODEL": { 
      // 所有配置合并的统计（不包含极端情况分析）
    }
  }
}
```

---

## 🎯 总结

### 基本规律

1. **Branch Size > Reward Model > Policy Model** (影响力排序)
2. 不同配置下的选择倾向可能**完全不同**
3. 需要根据**实际需求**（详细 vs 简洁）选择合适的配置组合
4. Skywork-PRM对branch数量不敏感，始终倾向MAX
5. Qwen2.5-Math-PRM对branch数量敏感，branch越多MIN倾向越强

### 极端情况规律

6. **极端branch在实际beam search中非常普遍**（70-90%的决策存在极端情况）
7. **Reward Model决定对极端值的偏好**：
   - Skywork-PRM → 显著偏好极端大的branch（token多）
   - Qwen2.5-Math-PRM → 显著偏好极端小的branch（token少）
8. **Branch数量越多，极端branch被选择的绝对概率越低**（但相对倾向保持）
   - 2 branches: 极端选择率 45-66%
   - 4 branches: 极端选择率 18-40%
   - 8 branches: 极端选择率 7-23%
9. **极端情况分析验证了之前的MAX/MIN倾向结论**：
   - 倾向MAX的模型，在极端情况下更倾向选择极端大的branch
   - 倾向MIN的模型，在极端情况下更倾向选择极端小的branch

### 实践意义

- **预测模型行为**：根据极端情况选择率，可以更准确预测模型在token差异明显时的选择
- **优化beam search策略**：了解模型对极端值的偏好，可以调整beam width等参数
- **解释生成结果**：当生成结果偏长或偏短时，可以从极端值选择倾向找原因

---

**生成时间**：2026-03-15  
**分析文件**：`branch_selection_analysis.json`  
**脚本文件**：`analyze_branch_selection.py`  
**极端值阈值**：1.5x（可配置）
