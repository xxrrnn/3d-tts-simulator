# Branch分布分析

本目录包含对workload文件中branch分布情况的分析结果。

## 文件说明

### 1. `analyze_branch_distribution.py`
分析脚本，用于处理workload JSON文件并生成汇总报告。

**功能：**
- 读取指定目录下所有的`question_*_workload.json`文件
- 统计每个问题中8、7、6、5 branch的出现次数
- 生成详细的分析报告（文本格式和JSON格式）

**使用方法：**
```bash
python analyze_branch_distribution.py
```

### 2. `extract_branch_tokens.py`
提取并汇总所有branch_count=8/7/6/5的step详细信息，重点关注token数量。

**功能：**
- 提取所有8、7、6、5 branch的步骤
- 详细记录每个branch的token数量
- 提供多种格式输出（TXT、JSON、CSV）

**使用方法：**
```bash
python extract_branch_tokens.py
```

### 3. `branch_distribution_summary_16384_8_1.txt`
分析报告（文本格式），包含以下内容：

#### 第1部分：总体统计
- 各个branch count（8/7/6/5）的总出现次数
- 涉及的问题数量
- 平均每题出现次数

#### 第2部分：按Branch Count分类详细信息
针对每种branch count（8/7/6/5），列出：
- 所有涉及的问题
- 每个问题的出现次数和占比

#### 第3部分：每个问题的详细分析
针对每个问题，列出：
- 总步数
- Branch count分布
- 目标branch（8/7/6/5）的详细信息，包括：
  - 选中的branch索引
  - 各branch的token数
  - 各branch的reward值

#### 第4部分：汇总统计表格
以表格形式展示所有问题的branch分布情况

**关键统计（16384_8_1配置）：**
- 分析问题数：30个
- 8 Branches：139次（36.4%），涉及30个问题
- 7 Branches：31次（8.1%），涉及18个问题
- 6 Branches：30次（7.9%），涉及17个问题
- 5 Branches：23次（6.0%），涉及14个问题

### 4. `branch_distribution_data_16384_8_1.json`
分析结果的JSON格式数据，便于程序化处理。

**数据结构：**
```json
{
  "metadata": {
    "source_dir": "源目录路径",
    "total_questions": "问题总数",
    "generated_at": "生成时间"
  },
  "global_statistics": {
    "8": {
      "total_occurrences": "总出现次数",
      "question_count": "问题数",
      "questions": ["问题列表"]
    },
    ...
  },
  "question_analyses": [
    {
      "question_id": "问题ID",
      "branch_count_stats": {"branch_count分布"},
      "step_details": ["步骤详细信息"],
      "total_steps": "总步数"
    },
    ...
  ]
}
```

### 5. `branch_tokens_summary_16384_8_1.txt` ⭐
**重点文件** - 所有8/7/6/5 branch步骤的Token详细汇总报告。

**内容结构：**
- **元数据**：数据源、总步骤数、各branch count的步数
- **按Branch Count分组**：每种branch count（8/7/6/5）独立展示
  - Token统计（总数、平均、最小、最大、中位数）
  - 按问题分组的详细信息
  - 每个step的branch tokens数组
  - 每个branch的reward值
  - 选中的branch信息
- **对比分析**：不同branch count的token统计对比表
- **按问题汇总**：每个问题在各branch count下的token统计

**关键数据（16384_8_1配置）：**
- 总步骤数：223步（全部是8/7/6/5 branch）
- 8 Branches：139步，总计133,041 tokens，平均119.64 tokens/branch
- 7 Branches：31步，总计16,482 tokens，平均75.95 tokens/branch
- 6 Branches：30步，总计17,005 tokens，平均94.47 tokens/branch
- 5 Branches：23步，总计5,314 tokens，平均46.21 tokens/branch

### 6. `branch_tokens_data_16384_8_1.json`
结构化的JSON数据，包含所有8/7/6/5 branch步骤的完整信息。

**数据结构：**
```json
{
  "metadata": {...},
  "steps_by_branch_count": {
    "8": [
      {
        "question_id": "...",
        "step_index": 0,
        "branch_count": 8,
        "selected_branch_index": 3,
        "branch_tokens": [52, 70, 44, 42, 83, 46, 30, 50],
        "branch_rewards": [0.6929, 0.6689, ...],
        "source_file": "..."
      },
      ...
    ],
    "7": [...],
    "6": [...],
    "5": [...]
  }
}
```

### 7. `branch_tokens_16384_8_1.csv`
CSV格式的数据表，便于在Excel或数据分析工具中使用。

**列结构：**
- `question_id`: 问题ID
- `step_index`: 步骤索引
- `branch_count`: branch数量（8/7/6/5）
- `selected_branch_index`: 选中的branch索引
- `branch_0_tokens` ~ `branch_7_tokens`: 各个branch的token数量
- `selected_reward`: 选中branch的reward值
- `avg_reward`: 所有branch的平均reward

**数据量：** 223行（不含表头），对应223个步骤

## 数据来源

**源目录：**
```
/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload/model_workloads/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/math-shepherd-mistral-7b-prm/16384_8_1
```

**配置参数：**
- 最大上下文长度：16384
- 初始beam宽度：8
- 其他配置：1

## 关键发现

### 1. Branch分布特征
- **8 branches占主导**：在所有步数中占36.4%，几乎每个问题都有8 branches的情况
- **7/6/5 branches较少**：分别只占8.1%、7.9%、6.0%
- **其他branch count**：占41.6%（主要是4 branches和更少）

### 2. 问题特征
- **高频8-branch问题**：
  - Q7：100%的步数都是8 branches
  - Q2：70%的步数是8 branches
  - Q12, Q21, Q29：75%的步数是8 branches
  
- **复杂问题**（步数多）：
  - Q18：48步（最多）
  - Q15：28步
  - Q25：25步
  - Q26：23步

### 3. Branch减少的原因
从数据中可以看出，branch count从8减少到7、6、5通常发生在：
- 搜索过程的后期阶段
- 某些特定的推理步骤
- 可能与beam pruning策略相关

## 后续分析建议

1. **性能分析**：对比8/7/6/5 branch情况下的求解性能差异
2. **时机分析**：分析branch减少发生在搜索的哪个阶段
3. **原因分析**：结合reward值分析branch减少的触发条件
4. **优化建议**：基于分析结果提出beam search策略的优化方案

## 生成时间
2026-04-01 21:10:42
