# 工作负载生成器 (Workload Generator)

从 beam search JSON 数据构建推理工作负载的工具，用于分析和理解整个推理过程。

## 功能特性

- **Prefill 阶段分析**: 提取 KV cache 数量、初始 tokens、beam size 等信息
- **Decode 阶段分析**: 详细记录每个 step 的 branch 数量、token 数、选择的 branch index 等
- **结构化输出**: 按照 `dataset/policy_model/reward_model` 的层次结构组织输出
- **独立文件**: 每个问题生成单独的工作负载文件

## 安装和使用

### 基本用法

```bash
# 生成工作负载
python gen_workload.py --input src/output/AMC23_beam_search --verbose

# 分析工作负载
python analyze_workload.py --workload-dir workload/AMC23_beam_search/Qwen2.5-1.5B/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/60_8_1/ --max-files 5
```

### 命令行参数

#### gen_workload.py
- `--input`: 输入目录路径 (必需)
- `--dataset`: 数据集名称 (可选，默认从路径自动提取)
- `--verbose`: 详细输出

#### analyze_workload.py
- `--workload-dir`: 工作负载目录路径 (必需)
- `--max-files`: 最多分析的文件数 (默认: 3)

## 输出文件结构

```
workload/
├── AMC23_beam_search/              # 数据集名称
│   └── Qwen2.5-1.5B/               # 策略模型
│       └── Skywork-o1-Open-PRM-Qwen-2.5-1.5B/  # 奖励模型
│           └── 60_8_1/             # 配置参数
│               ├── question_0_workload.json
│               ├── question_1_workload.json
│               └── ...
```

## 工作负载文件格式

每个工作负载文件包含以下信息：

### 基本信息
```json
{
  "question_id": "question_0",
  "original_question": "问题原文...",
  "groundtruth": "27.0",
  "final_answer": "27",
  "generation_metadata": {
    "dataset": "AMC23_beam_search",
    "source_file": "原始文件路径",
    "generated_by": "gen_workload.py"
  }
}
```

### Prefill 阶段
```json
{
  "prefill": {
    "phase": "prefill",
    "question": "问题原文",
    "question_length": 258,
    "kv_cache_count": 220,      // KV cache 数量
    "initial_tokens": 168,      // 初始 tokens 数
    "beam_size": 1,            // beam search 大小
    "max_step": 60,            // 最大步数
    "initial_branches": 4       // 初始分支数
  }
}
```

### Decode 阶段
```json
{
  "decode": {
    "phase": "decode",
    "total_steps": 9,
    "steps": [
      {
        "step": 0,
        "current_nodes": 1,
        "selection_process": {
          "available_branches": 4,
          "terminated_count": 0,
          "branches": [
            {
              "branch_index": 0,
              "selected": true,           // 是否被选择
              "reward_score": 0.9968,     // 奖励分数
              "prior_prob": 0.9877,       // 先验概率
              "num_tokens": 75,           // token 数量
              "branch_content_length": 335,
              "full_path_length": 335
            }
          ]
        },
        "expansion_results": {
          "pre_expansion_value": 0.9968,
          "final_status": "expanded",
          "num_new_children": 2,
          "api_completion_tokens": 168,
          "terminated": false
        },
        "summary": {
          "nodes_to_expand": 1,
          "terminated_nodes": 0,
          "beam_width_used": 1,
          "nodes_expanded": 1,
          "final_terminated_nodes": 0
        }
      }
    ],
    "final_path_info": {
      "final_node_value": 0.9899,
      "is_terminated": true
    },
    "statistics": {
      "total_completion_tokens": 2187,
      "tree_completion_tokens": 761,
      "reward_history": [0.9968, 0.9931, ...],
      "token_history": [75, 21, 40, ...],
      "prob_history": [0.9877, 0.9861, ...]
    }
  }
}
```

## 关键指标说明

### Prefill 阶段指标
- **kv_cache_count**: 预填充后的 KV cache 条目数量
- **initial_tokens**: 第一次推理产生的 token 数量
- **initial_branches**: 第一次推理产生的分支数量

### Decode 阶段指标
- **available_branches**: 每步可选择的分支数量
- **selected**: 分支是否被选择 (true/false)
- **num_tokens**: 每个分支产生的 token 数量
- **reward_score**: 分支的奖励分数
- **api_completion_tokens**: API 调用产生的 completion token 数
- **final_status**: 节点最终状态 (expanded/terminated)

## 使用场景

1. **性能分析**: 了解推理过程的计算开销和资源需求
2. **算法优化**: 分析 beam search 的分支选择策略
3. **Token 追踪**: 监控 prefill 和 decode 阶段的 token 消耗
4. **质量评估**: 通过奖励分数和选择路径评估推理质量

## 输出示例

运行分析工具后的输出示例：

```
🔧 Prefill 阶段:
  - 问题长度: 258 字符
  - KV Cache 数量: 220
  - 初始 tokens: 168
  - Beam size: 1
  - 初始分支数: 4

🚀 Decode 阶段:
  - 总步数: 9
  Step 0:
    - 可用分支: 4
    - 选择分支: 1
    - 新子节点: 2
    - API tokens: 168
    - 节点扩展: expanded

📊 整体统计:
  - 总完成tokens: 2187
  - 树搜索tokens: 761
  - 总生成分支数: 29
  - 总生成tokens: 284
  - 奖励分数范围: 0.9899 - 0.9974
  - 最终奖励分数: 0.9899

🎯 选择的推理路径:
  Step 0: Branch 0 (奖励: 0.9968, tokens: 75)
  Step 1: Branch 0 (奖励: 0.9931, tokens: 21)
  ...
```

## 注意事项

1. 确保输入目录包含有效的 beam search 日志文件
2. 工作负载文件以 JSON 格式保存，便于后续处理和分析
3. 支持多种数据集和模型组合的批量处理
4. KV cache 数量是基于问题长度和初始 tokens 的估算值