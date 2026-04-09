# LLM Beam Search 推理确定性修复文档

## 概述

本文档记录了在 beam search 推理过程中实现严格确定性（逐 token 级别可复现）的完整解决方案。

### 问题背景

在进行 straggler 剪枝实验时，需要对比不同配置下的精度变化。为保证实验的可控性，要求：
- **相同 seed + 相同输入 → 逐 token 完全一致的输出**
- **不同分支之间仍然保持多样性**（通过不同的派生 seed）
- **采样仍具有随机性**（`temperature > 0`，非贪心解码）

### 观察到的问题

设置 `branch_width=8`, `temperature=0.7`, `seed=42` 后，多次运行同一题目：
- run2 和 run3 的 8 个分支中有 7 个完全一致
- **第 4 个分支 token 数量不同**：run2 = 55 tokens，run3 = 36 tokens
- 这说明存在随机的非确定性，导致实验结果无法精确复现

---

## 根本原因分析

### 1. 初步排除的因素

| 因素 | 是否是原因 | 说明 |
|---|---|---|
| Attention backend | ❌ | Flash-Attn 和 XFormers 的 forward pass 都是确定性的 |
| cuBLAS GEMM | ❌ | 在单独测试中 `torch.matmul` 是确定性的 |
| LLM sampling seed | ❌ | vLLM 的 seed 已正确设置和传递 |
| Temperature=0.0 | ✅ 部分原因 | 会导致 greedy，所有分支相同（已修复为 0.7）|
| top_k=branch_width | ✅ 部分原因 | 限制采样空间，降低多样性（已移除）|

### 2. 真正的根本原因：Batch 组合导致的浮点非确定性

**实验验证**：

```python
# 测试：batch processing vs single processing
x_batch = torch.randn(8, 128, 4096, device='cuda', dtype=torch.float16)
weight = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)

# Case 1: 8个样本一起做 matmul
result_batch = torch.matmul(x_batch, weight.T)

# Case 2: 逐个做 matmul
results_single = [torch.matmul(x_batch[i:i+1], weight.T) for i in range(8)]
result_concat = torch.cat(results_single, dim=0)

# 结果
same = torch.equal(result_batch, result_concat)  # False!
max_diff = (result_batch - result_concat).abs().max().item()  # 0.25
```

**结论**：同样的计算，在不同 batch size 下执行时，由于 fp16 的精度限制和 CUDA 并行执行顺序的差异，会产生不同的结果。

### 3. vLLM Continuous Batching 的调度不确定性

当前实现中：
1. `base_env.py` 第 324 行使用 `ThreadPoolExecutor` 并行发送 8 个 branch 请求
2. 线程调度的时序是不确定的，请求到达 vLLM 的顺序不固定
3. vLLM 的 continuous batching 会将同时到达的请求合并处理
4. **不同运行中，同一个请求可能与不同数量的其他请求被 batch 在一起**
5. batch 组合不同 → GEMM 计算结果不同 → logits 不同 → 采样到不同 token

**示例**：
```
Run 1:
  Request 0,1,2 到达 → batch [0,1,2] 处理
  Request 3,4   到达 → batch [3,4]   处理
  ...

Run 2:
  Request 0,1,2,3 到达 → batch [0,1,2,3] 处理  ← batch size 变了！
  Request 4       到达 → batch [4]       处理
  ...
```

虽然每个请求的 seed 相同，但因为 batch size 不同，Request 3 在两次运行中得到的 logits 有微小差异（fp16 精度下可达 0.25），这个差异在采样边界上就会导致选择不同的 token。

---

## 解决方案

### 核心思路

**确保每个请求在完全相同的计算环境中执行**，即：
- 每个请求独占 vLLM，batch size 始终为 1
- 请求按固定顺序串行发送和处理
- 结合 seed + Flash-Attn + cuBLAS 确定性模式

### 实现方式：新增 `--deterministic` 模式

当 `deterministic=1` 时：
1. **串行发送**：不使用 `ThreadPoolExecutor`，逐个发送 branch 请求
2. **Flash-Attn backend**：forward pass 确定性（文档明确保证）
3. **cuBLAS 确定性**：设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`

---

## 代码修改清单

### 1. `vllm_worker.py` - 切换到 Flash-Attn 后端

**文件**：`reason/llm_service/workers/vllm_worker.py`

**修改**：
```python
import os
# 使用 FLASH_ATTN 后端（forward pass 始终确定性）；若需回退到 xformers 可设环境变量
# VLLM_ATTENTION_BACKEND=XFORMERS
os.environ.setdefault('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')
# 强制 cuBLAS 使用确定性算法（避免 GEMM 运算的浮点非确定性）
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
```

**原因**：
- Flash-Attn 2.8.3 的文档明确说明 forward pass 始终确定性
- cuBLAS workspace config 强制使用确定性的矩阵乘法算法

---

### 2. `base_env.py` - 实现串行请求发送

**文件**：`envs/base_env.py`

**修改位置**：第 324-329 行

**修改前**：
```python
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    parts: List[ConcatedLMGenResult] = list(ex.map(_one_branch, range(n)))
result: ConcatedLMGenResult = _merge_concated_lm_results(parts)
```

**修改后**：
```python
deterministic_mode = bool(self.config.get("deterministic", False))
if deterministic_mode:
    parts: List[ConcatedLMGenResult] = [_one_branch(i) for i in range(n)]
else:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        parts: List[ConcatedLMGenResult] = list(ex.map(_one_branch, range(n)))
result: ConcatedLMGenResult = _merge_concated_lm_results(parts)
```

**原因**：
- `deterministic=True` 时使用列表推导式串行调用，确保请求按顺序逐个发送
- `deterministic=False` 时保持并行发送，性能不受影响

---

### 3. `methods.py` - 配置传递

**文件**：`reason/evaluation/methods.py`

#### 3.1 在 `TreeSearchConfig` 添加字段

**位置**：第 48 行后

**添加**：
```python
# 严格确定性模式：串行发送 LLM 请求避免 batch 组合差异导致的浮点非确定性
deterministic: bool = False
```

#### 3.2 传递给 env config

**位置**：第 93-95 行

**修改前**：
```python
"eval_seed": config.eval_seed,
"split_lm_n_for_seeds": config.split_lm_n_for_seeds,
"split_lm_parallel_workers": config.split_lm_parallel_workers,
```

**修改后**：
```python
"eval_seed": config.eval_seed,
"split_lm_n_for_seeds": config.split_lm_n_for_seeds,
"split_lm_parallel_workers": config.split_lm_parallel_workers,
"deterministic": config.deterministic,
```

---

### 4. `evaluate.py` - 命令行参数

**文件**：`reason/evaluation/evaluate.py`

#### 4.1 添加命令行参数

**位置**：第 127 行后

**添加**：
```python
parser.add_argument(
    "--deterministic",
    type=int,
    default=0,
    choices=[0, 1],
    help="1=严格确定性模式：串行发送 LLM 请求，消除 batch 组合导致的浮点非确定性；会降低吞吐",
)
```

#### 4.2 传递给 BeamSearchConfig

**位置**：第 345 行和第 370 行（两处）

**添加**：
```python
eval_seed=int(args.seed),
deterministic=bool(args.deterministic),  # ← 新增这行
```

---

### 5. `run.sh` - 参数传递

**文件**：`scripts/run.sh`

#### 5.1 定义默认值

**位置**：第 47 行后

**添加**：
```bash
deterministic=0 #1=严格确定性模式（串行 LLM 请求，消除 batch 浮点差异，会降低吞吐）
```

#### 5.2 参数解析

**位置**：第 145 行后

**添加**：
```bash
--deterministic)
    deterministic="$2"
    shift 2
    ;;
```

#### 5.3 传递给 evaluate.py

**位置**：第 269 行后

**添加**：
```bash
--straggler_deferred_prune $straggler_deferred_prune \
--deterministic $deterministic  # ← 新增这行
```

---

### 6. `eval_all_combinations_straggler.sh` - 启用确定性

**文件**：`scripts/eval_all_combinations_straggler.sh`

**位置**：第 401 行

**修改前**：
```bash
--straggler_prune_other_reward_threshold "$straggler_other_thr" \
--straggler_deferred_prune "$straggler_deferred"
```

**修改后**：
```bash
--straggler_prune_other_reward_threshold "$straggler_other_thr" \
--straggler_deferred_prune "$straggler_deferred" \
--deterministic 1  # ← 新增这行，启用确定性模式
```

---

## 使用方法

### 启用确定性模式

在 `eval_all_combinations_straggler.sh` 中已默认启用 `--deterministic 1`。

如果需要在其他场景使用，直接在 `run.sh` 调用时添加参数：

```bash
bash scripts/run.sh \
    --method beam_search \
    --LM <policy_model> \
    --RM <reward_model> \
    --width 8 \
    --num_seq 1 \
    --temperature 0.7 \
    --seed 42 \
    --deterministic 1  # 启用确定性
```

### 关闭确定性模式（恢复并行性能）

如果不需要严格确定性，可以设置 `--deterministic 0`：

```bash
--deterministic 0  # 使用并行发送，提升吞吐
```

或者在 `eval_all_combinations_straggler.sh` 第 401 行删除或注释掉该参数。

---

## 性能影响

### 吞吐量对比

| 模式 | branch 请求发送方式 | 单 step 耗时 | 吞吐比例 |
|---|---|---|---|
| `deterministic=0`（并行） | ThreadPoolExecutor 并行 | ~T | 100% |
| `deterministic=1`（串行） | 列表推导式串行 | ~n×T | ~12.5%（n=8时）|

**说明**：
- `n = branch_width`，即每个 step 生成的分支数
- 串行模式下，每个 branch 必须等待前一个完成，因此耗时约为 `n` 倍
- 对于 `branch_width=8` 的配置，串行模式吞吐约为并行模式的 12.5%

### 适用场景建议

| 场景 | 推荐模式 | 原因 |
|---|---|---|
| **对比实验**（需要精确复现） | `deterministic=1` | 确保结果严格一致，便于分析精度变化 |
| **大规模评估**（数百题） | `deterministic=0` | 性能优先，可接受微小的随机差异 |
| **最终测试集评估** | `deterministic=0` | 统计上多样性更真实 |
| **Debug / 单题分析** | `deterministic=1` | 便于逐 token 对比分析 |

---

## 验证方法

### 1. 运行两次相同配置

```bash
# Run 1
bash scripts/eval_all_combinations_straggler.sh

# Run 2 (使用相同的 seed)
bash scripts/eval_all_combinations_straggler.sh
```

### 2. 比较 workload 文件

```bash
cd model_workloads/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/math-shepherd-mistral-7b-prm/

# 比较两次运行的 branch_tokens
diff \
  40_8_1_straggler_0_0_0_0_0_def0_run1/question_0_workload.json \
  40_8_1_straggler_0_0_0_0_0_def0_run2/question_0_workload.json
```

### 3. 预期结果

启用 `deterministic=1` 后，两次运行应该：
- ✅ 所有 step 的 `branch_count` 完全一致
- ✅ 所有 step 的 `branch_tokens` 数组完全一致（逐元素相等）
- ✅ 所有 `branch_token_probs` 完全一致（浮点精度内）
- ✅ `selected_branch_index` 完全一致

如果仍有差异，检查：
1. vLLM worker 是否已重启（确保环境变量生效）
2. 是否有其他请求同时访问 vLLM（应独占使用）
3. seed 是否确实相同

---

## 技术原理总结

### 确定性的三层保障

```
┌─────────────────────────────────────────────┐
│ 1. Seed 机制：派生唯一确定性的随机数种子     │
│    _derive_lm_branch_seed(eval_seed, q, step, idx) │
│    → SHA256(42 + question + 0 + 3) = 固定seed  │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ 2. 串行发送：消除 batch 组合的不确定性       │
│    for i in range(n):                       │
│        send_request(seed=derived_seed[i])   │
│    → 每个请求独占 vLLM，batch_size=1 固定   │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ 3. 确定性算子：消除底层计算的浮点误差       │
│    - FLASH_ATTN: forward pass 确定性        │
│    - CUBLAS_WORKSPACE_CONFIG: GEMM 确定性   │
└─────────────────────────────────────────────┘
              ↓
        逐 token 完全一致
```

### 与多样性的平衡

- **分支间多样性**：由 `branch_idx` 不同 → seed 不同 → 输出不同 保证
- **采样随机性**：`temperature=0.7` → 非贪心，有随机性
- **可复现性**：同 seed + 串行 + 确定性算子 → 结果固定

这三者并不冲突，完美实现了需求。

---

## 常见问题（FAQ）

### Q1: 为什么不能用 `temperature=0.0` 来保证确定性？

**A**: `temperature=0.0` 是贪心解码，所有分支都会产生完全相同的输出（因为每个位置都选概率最高的 token），这违背了 beam search 需要多样性的初衷。正确的做法是用 seed + 串行发送。

### Q2: 为什么要换到 FLASH_ATTN？XFORMERS 不也是确定性的吗？

**A**: 实验验证 XFORMERS 的 forward pass 在隔离测试中确实是确定性的。但 FLASH_ATTN 的文档明确保证了这一点，且性能更好。为了万无一失，推荐使用 FLASH_ATTN。如果环境不兼容，可以设置环境变量回退：
```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
```

### Q3: 为什么 `branch_count` 还是会小于 `branch_width`？

**A**: 这是正常现象，原因包括：
1. **LLM 生成了重复的回复**：去重后只保留唯一的（合理）
2. **finish_reason != "stop"**：因长度限制等原因异常终止的被过滤（合理）
3. **Straggler 剪枝**：异常长的分支被剪枝（如果启用）

这些都是设计预期的行为，不是 bug。确定性保证的是：**相同配置下，这些过滤后的结果完全一致**。

### Q4: 串行模式太慢了，有没有折中方案？

**A**: 如果只需要"大致确定性"（统计上可接受），可以：
1. 保持 `deterministic=0`（并行）
2. 设置较小的 `split_lm_parallel_workers`（如 2-4），减少并发度
3. 接受 10-20% 的差异概率

但对于精确的对比实验，仍建议使用 `deterministic=1`。

---

## 相关文件清单

| 文件 | 作用 |
|---|---|
| `reason/llm_service/workers/vllm_worker.py` | 设置 vLLM attention backend 和 cuBLAS 确定性 |
| `envs/base_env.py` | 实现串行/并行请求发送的切换逻辑 |
| `reason/evaluation/methods.py` | 定义 `deterministic` 配置字段 |
| `reason/evaluation/evaluate.py` | 命令行参数解析 |
| `scripts/run.sh` | 参数传递中间层 |
| `scripts/eval_all_combinations_straggler.sh` | 启用确定性模式的主脚本 |

---

## 版本历史

| 日期 | 版本 | 修改内容 | 修改人 |
|---|---|---|---|
| 2026-04-09 | v1.0 | 初始版本，实现 deterministic 模式 | - |

---

## 参考资料

1. [vLLM Attention Backends](https://github.com/vllm-project/vllm)
2. [Flash-Attention 2 文档](https://github.com/Dao-AILab/flash-attention)
3. [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
4. [CUDA cuBLAS Determinism](https://docs.nvidia.com/cuda/cublas/index.html)
