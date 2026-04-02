# Branch 分析汇总 - 文件索引

本目录包含16384_8_1配置下所有branch_count=8/7/6/5步骤的详细分析。

## 📁 文件导航

### 🌟 推荐首先查看

1. **QUICK_GUIDE.md** - 快速入门指南
   - 5分钟了解所有文件
   - 查看关键统计数据
   - 学习如何使用这些文件

2. **branch_tokens_summary_16384_8_1.txt** - Token详细汇总（73KB）
   - ⭐ 核心文件：所有223个步骤的完整token信息
   - 按branch count和问题分组
   - 包含每个branch的token数组和reward值

3. **branch_tokens_16384_8_1.csv** - 数据表格（13KB）
   - ⭐ Excel友好格式
   - 223行 × 15列
   - 适合数据分析和可视化

### 📊 数据文件

| 文件名 | 格式 | 内容 | 用途 |
|--------|------|------|------|
| `branch_tokens_summary_16384_8_1.txt` | TXT | Token详细信息 | 人类阅读 |
| `branch_tokens_16384_8_1.csv` | CSV | Token数据表 | Excel分析 |
| `branch_tokens_data_16384_8_1.json` | JSON | 结构化数据 | 程序处理 |
| `branch_distribution_summary_16384_8_1.txt` | TXT | 分布统计 | 整体了解 |
| `branch_distribution_data_16384_8_1.json` | JSON | 分布数据 | 程序处理 |

### 🛠️ 工具脚本

| 脚本名 | 功能 | 使用方法 |
|--------|------|----------|
| `view_branch_tokens.py` | 快速查看工具 | `python view_branch_tokens.py <命令>` |
| `extract_branch_tokens.py` | 提取token信息 | `python extract_branch_tokens.py` |
| `analyze_branch_distribution.py` | 分析分布 | `python analyze_branch_distribution.py` |

### 📖 文档

- `README.md` - 完整文档说明
- `QUICK_GUIDE.md` - 快速入门指南
- `INDEX.md` - 本文件（文件索引）

## 🚀 快速开始

### 1. 查看整体统计
```bash
python view_branch_tokens.py stats
```

### 2. 列出所有问题
```bash
python view_branch_tokens.py list
```

### 3. 查看特定问题的token信息
```bash
python view_branch_tokens.py q question_0
```

### 4. 查看特定branch count的统计
```bash
python view_branch_tokens.py bc 8
```

## 📈 数据概览

### 基本信息
- **配置**: 16384_8_1
- **模型**: Qwen2.5-Math-1.5B-Instruct + math-shepherd-mistral-7b-prm
- **数据集**: AIME24_beam_search
- **问题数**: 30个
- **总步骤**: 382步
- **8/7/6/5步骤**: 223步（58.4%）

### Branch Count分布

```
Branch Count    步数    占比    总Tokens    平均Token/branch
─────────────────────────────────────────────────────────
8 branches      139    36.4%    133,041         119.64
7 branches       31     8.1%     16,482          75.95
6 branches       30     7.9%     17,005          94.47
5 branches       23     6.0%      5,314          46.21
其他            159    41.6%         -              -
```

## 🎯 常见使用场景

### 场景1: 想知道某个问题的所有token分布
→ 打开 `branch_tokens_summary_16384_8_1.txt`，搜索问题ID

### 场景2: 需要在Excel中分析数据
→ 用Excel打开 `branch_tokens_16384_8_1.csv`

### 场景3: 用Python进行数据分析
```python
import json
with open('branch_tokens_data_16384_8_1.json') as f:
    data = json.load(f)
steps_8br = data['steps_by_branch_count']['8']
```

### 场景4: 快速查看某个问题
```bash
python view_branch_tokens.py q question_14
```

### 场景5: 对比不同branch count的特征
→ 查看 `branch_tokens_summary_16384_8_1.txt` 末尾的"对比分析"

## 📊 文件关系图

```
源数据 (workload JSON文件)
    ↓
[extract_branch_tokens.py]
    ↓
    ├─→ branch_tokens_summary_16384_8_1.txt  (详细报告)
    ├─→ branch_tokens_data_16384_8_1.json    (JSON数据)
    └─→ branch_tokens_16384_8_1.csv          (CSV表格)
         ↑
         └─ [view_branch_tokens.py] (查看工具)

源数据 (workload JSON文件)
    ↓
[analyze_branch_distribution.py]
    ↓
    ├─→ branch_distribution_summary_16384_8_1.txt
    └─→ branch_distribution_data_16384_8_1.json
```

## 🔍 数据字段说明

### CSV文件字段
- `question_id`: 问题标识
- `step_index`: 步骤编号
- `branch_count`: 分支数量（8/7/6/5）
- `selected_branch_index`: 选中的分支索引
- `branch_0_tokens` ~ `branch_7_tokens`: 各分支token数
- `selected_reward`: 选中分支的reward值
- `avg_reward`: 所有分支的平均reward

### JSON数据结构
```json
{
  "metadata": {
    "source_dir": "源目录",
    "total_steps": 223,
    "generated_at": "时间戳"
  },
  "steps_by_branch_count": {
    "8": [步骤数组],
    "7": [步骤数组],
    "6": [步骤数组],
    "5": [步骤数组]
  }
}
```

## 💡 提示和技巧

1. **快速定位**: TXT文件支持文本搜索，可以快速定位到问题ID
2. **Excel透视表**: CSV文件可以用Excel透视表进行多维度分析
3. **命令行工具**: `view_branch_tokens.py` 提供便捷的命令行查询
4. **JSON灵活性**: JSON格式便于各种编程语言处理

## 📅 版本信息

- **生成时间**: 2026-04-01 21:17:08
- **数据版本**: v1.0
- **脚本版本**: v1.0

## 📧 更多信息

查看 `README.md` 获取完整文档，或查看 `QUICK_GUIDE.md` 快速入门。

---

**祝分析愉快！** 🎉
