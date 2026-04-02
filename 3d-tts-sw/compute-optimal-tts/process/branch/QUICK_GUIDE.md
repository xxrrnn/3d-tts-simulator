# Branch Token详细汇总 - 快速指南

本文档快速说明所有生成的文件，帮助你快速找到需要的信息。

## 🎯 核心文件推荐

### 最重要的3个文件：

1. **`branch_tokens_summary_16384_8_1.txt`** (73KB) ⭐⭐⭐
   - **用途**：查看所有8/7/6/5 branch步骤的Token详细信息
   - **内容**：每个步骤的branch tokens数组、reward值、选中信息
   - **格式**：易读的文本格式，按branch count和问题分组
   
2. **`branch_tokens_16384_8_1.csv`** (13KB) ⭐⭐⭐
   - **用途**：在Excel或Python中进行数据分析
   - **内容**：223行数据，每行代表一个步骤
   - **格式**：CSV表格，包含所有branch的token数和reward值
   
3. **`branch_distribution_summary_16384_8_1.txt`** (59KB) ⭐⭐
   - **用途**：了解整体branch分布情况和统计
   - **内容**：branch count分布、出现频率、汇总表格
   - **格式**：统计报告格式

## 📊 所有文件列表

| 文件名 | 大小 | 类型 | 主要内容 |
|--------|------|------|----------|
| `branch_tokens_summary_16384_8_1.txt` | 73KB | 文本报告 | **所有8/7/6/5 branch的token详细信息** |
| `branch_tokens_16384_8_1.csv` | 13KB | CSV表格 | **223个步骤的token数据（适合Excel分析）** |
| `branch_tokens_data_16384_8_1.json` | 161KB | JSON数据 | 所有步骤的结构化数据 |
| `branch_distribution_summary_16384_8_1.txt` | 59KB | 文本报告 | Branch分布统计和分析 |
| `branch_distribution_data_16384_8_1.json` | 167KB | JSON数据 | Branch分布的结构化数据 |
| `extract_branch_tokens.py` | 11KB | Python脚本 | 提取token信息的脚本 |
| `analyze_branch_distribution.py` | 11KB | Python脚本 | 分析branch分布的脚本 |

## 📈 关键统计数据

### 总体统计（16384_8_1配置）
- **数据源**：30个问题的workload文件
- **总步骤数**：382步
- **8/7/6/5 branch步骤**：223步（58.4%）

### 按Branch Count分类

| Branch Count | 步数 | 占比 | 总Tokens | 平均Token/branch | Token范围 |
|--------------|------|------|----------|------------------|-----------|
| **8 branches** | 139 | 36.4% | 133,041 | 119.64 | [6, 1354] |
| **7 branches** | 31 | 8.1% | 16,482 | 75.95 | [7, 397] |
| **6 branches** | 30 | 7.9% | 17,005 | 94.47 | [6, 2134] |
| **5 branches** | 23 | 6.0% | 5,314 | 46.21 | [5, 294] |

### 关键发现
1. **8 branches占主导**：139步，几乎每个问题都有
2. **Token数量递减**：branch count越少，平均token数也越少
3. **变异范围大**：最小6 tokens，最大2134 tokens

## 🔍 如何使用这些文件

### 场景1：查看某个问题的所有token信息
👉 打开 `branch_tokens_summary_16384_8_1.txt`，搜索问题ID（如"question_0"）

### 场景2：在Excel中分析token分布
👉 打开 `branch_tokens_16384_8_1.csv`，使用Excel的筛选、排序、统计功能

### 场景3：编程分析（Python/JavaScript等）
👉 读取 `branch_tokens_data_16384_8_1.json`，使用JSON解析器处理

### 场景4：了解整体分布情况
👉 查看 `branch_distribution_summary_16384_8_1.txt` 的统计表格部分

### 场景5：对比不同branch count的特征
👉 查看 `branch_tokens_summary_16384_8_1.txt` 末尾的"对比分析"部分

## 📝 数据示例

### branch_tokens_summary_16384_8_1.txt 示例：
```
【question_0】共 4 个8-branch步骤

  Step 0:
    Branch Tokens: [52, 70, 44, 42, 83, 46, 30, 50]
      - 总计: 417 tokens
      - 平均: 52.12
      - 范围: [30, 83]
      - 选中: Branch 3 (42 tokens)
    Branch Rewards: [0.6929, 0.6689, 0.7461, 0.7773, ...]
      - 选中reward: 0.7773
```

### branch_tokens_16384_8_1.csv 示例：
```csv
question_id,step_index,branch_count,selected_branch_index,branch_0_tokens,branch_1_tokens,...
question_0,0,8,3,52,70,44,42,83,46,30,50,0.7773,0.7037
question_0,1,8,7,168,37,115,45,100,140,1354,285,0.8872,0.6034
```

## 🎓 术语说明

- **Branch Count**：该步骤生成的候选分支数量（8/7/6/5）
- **Branch Tokens**：每个分支包含的token数量（数组）
- **Selected Branch Index**：被选中继续搜索的分支索引（0-7）
- **Branch Rewards**：每个分支的奖励分数，用于选择最佳分支
- **Step Index**：在解题过程中的步骤编号

## 🔄 重新生成数据

如果需要重新分析或更新数据：

```bash
# 进入目录
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/branch

# 生成Token详细信息
python extract_branch_tokens.py

# 生成分布统计
python analyze_branch_distribution.py
```

## 📅 数据版本

- **生成时间**：2026-04-01 21:17:08
- **配置**：16384_8_1 (context=16384, beam=8, other=1)
- **模型**：Qwen2.5-Math-1.5B-Instruct + math-shepherd-mistral-7b-prm
- **数据集**：AIME24_beam_search (30个问题)

---

**需要帮助？** 查看 `README.md` 获取更多详细信息。
