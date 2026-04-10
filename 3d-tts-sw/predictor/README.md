# Adaptive Threshold Predictor

这个目录保留了运行自适应 straggler threshold 所需的最小文件集合：

- 训练：从 `output/**/record_0.jsonl` 构建数据、随机切分 record、训练 MLP、导出权重
- 评估：在固定随机划分上评估 train / val / test 的 precision / recall / threshold gap
- 预测：
  - 对单个 branch 直接预测 threshold
  - 对单个 `record_0.jsonl` 的某个 `step_idx`，判断哪些 branch 触发 threshold

## 保留文件

- [__main__.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/__main__.py)
- [config.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/config.py)
- [record_utils.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/record_utils.py)
- [adaptive_threshold_features.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_features.py)
- [adaptive_threshold_dataset.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_dataset.py)
- [adaptive_threshold_model.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_model.py)
- [adaptive_threshold_evaluate.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_evaluate.py)
- [adaptive_threshold_trainer.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_trainer.py)
- [record_step_predict.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/record_step_predict.py)
- [adaptive_threshold_weights.json](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_weights.json)
- [adaptive_threshold_weights_low_fp.json](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_weights_low_fp.json)
- [adaptive_threshold_priors.json](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_priors.json)

默认推荐：

- `adaptive_threshold_weights.json`：balanced 版本
- `adaptive_threshold_weights_low_fp.json`：更低误报版本，适合 RM 调用代价更敏感的场景
- 默认 `active_branch_gate = 2`

## Straggler 定义

定义在 [adaptive_threshold_dataset.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_dataset.py#L28)：

- 总 branch 数 `> 1`
- 当前 branch 长度 `my_tokens > 100`
- 当前 branch 长度 `my_tokens > 1.5 * max_other`

## 触发逻辑

运行时逻辑在 [adaptive_threshold_model.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_model.py#L29) 和 [adaptive_threshold_model.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_model.py#L36)：

1. 先等 branch 走到 `obs_point = max_other`
2. 只有当当前动态活跃 branch 数 `n_active_branches <= active_branch_gate` 时，才允许调用 MLP
3. 默认 `active_branch_gate = 2`，可配成 `1` 或 `2`
4. 用 `obs_point` 之前的特征预测一个 `budget`
5. 得到 `threshold = max_other + budget`
6. 若 `current_tokens > threshold`，则触发 `predicted_straggler=true`

所以这不是“提前很早就触发”的逻辑，而是：

- 先等大部分 branch 已停止
- 只在尾部还剩 1-2 个活跃 branch 时才调用 predictor
- 然后再判断当前 branch 是否超过预测出来的 threshold

## 使用的特征

特征定义在 [adaptive_threshold_features.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_features.py#L22)。

只使用 `obs_point` 之前、运行时已经可见的信息，不使用 reward。

### 1. 当前 branch 的 token 概率前缀

- `prob_mean`：到观察点为止，当前 branch 的 token 概率平均值
- `prob_min`：到观察点为止，最差一次 token 概率
- `prob_recent_mean`：最近一小段 token 的平均概率
- `low_prob_frac_05`：低置信 token 比例，阈值是 `0.5`
- `low_prob_frac_02`：更低置信 token 比例，阈值是 `0.2`

输入来源：

- `branch["token_probs"][:obs_point]`

### 2. top-k logprob 特征

- `topk_entropy_recent`：最近一小段 top-k 分布的不确定性，越大表示越不稳定
- `topk_margin_recent`：最近一小段 top-1 与 top-2 的差距，越大表示越确定
- `tracked_token_recent_rate`：最近一小段里，特殊 token 是否频繁出现在 top-k 中

输入来源：

- `branch["token_topk_logprobs"][:obs_point]`
- 当前默认跟踪 token：`272`

### 3. branch / sibling 长度关系

- `max_other_tokens`：其他 branch 中最长的那个长度

### 4. 运行时上下文

- `n_active_branches`：到观察点时仍然活着的 branch 数
- `prior_step_total_tokens`：之前所有 step 累计产生了多少 token
- `config_prior`：同一 `policy_model + prm + beam` 在训练集上的 straggler 先验比例

这版特征相对之前已经做了收缩：

- 当前最终保留的是 12 维特征，都是比较便宜的统计量
- 去掉了 `prob_std`、`prob_p10`、`prob_recent_min`
- 去掉了 sibling 平均/方差、finished ratio、step idx、step total、beam width 这类次要上下文
- 去掉了全局 mean 型 top-k 特征，只保留 recent 特征
- 去掉了更多派生 length 特征，只保留最核心的 `max_other`

`prior_step_total_tokens` 只来自前面已经完成的 step，不会偷看 threshold 之后的信息。

## 数据划分

数据集构造在 [adaptive_threshold_dataset.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_dataset.py#L198)。

- 先从 `output/` 下递归找到所有 `record_0.jsonl`
- 按 record 随机切分，不按 question id 固定切
- 只保留满足 `n_active_branches <= active_branch_gate` 的样本
- 默认 `15%` record 做测试集
- 剩余 record 里再切 `15%` 做验证集
- 默认随机种子是 `42`

## 训练

### 命令

```bash
python -m predictor train \
  --output-dir output \
  --save predictor/adaptive_threshold_weights.json \
  --hidden 32,16 \
  --epochs 60 \
  --lr 1e-3 \
  --batch-size 256 \
  --patience 10 \
  --dropout 0.05 \
  --quantile 0.93 \
  --coverage 0.97 \
  --min-group-negatives 8 \
  --active-branch-gate 2 \
  --test-fraction 0.15 \
  --val-fraction 0.15 \
  --split-seed 42
```

### 训练输出

会生成：

- `predictor/adaptive_threshold_weights.json`
- `predictor/adaptive_threshold_weights.metrics.json`
- `predictor/adaptive_threshold_priors.json`

当前默认 balanced 权重在测试集上的结果：

- `active_branch_gate = 2`
- `train precision = 0.8541`
- `val precision = 0.8552`
- `test precision = 0.8851`
- `test recall = 0.8610`
- `test f1 = 0.8729`
- `test avg_threshold_gap = 57.274`
- `test avg_saved_tokens_tp = 163.456`

### 训练代码入口

- CLI: [__main__.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/__main__.py#L12)
- 主训练逻辑: [adaptive_threshold_trainer.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_trainer.py#L217)

## 评估

### 命令

```bash
python -m predictor evaluate \
  --output-dir output \
  --weights predictor/adaptive_threshold_weights.json \
  --active-branch-gate 2
```

### 输出

会打印 `train`、`val` 和 `test` 的：

- `precision`
- `recall`
- `f1`
- `tp/fp/fn/tn`
- `avg_threshold_gap`
- `avg_saved_tokens_tp`

评估代码入口在 [adaptive_threshold_evaluate.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/adaptive_threshold_evaluate.py#L18)。

当前 low-fp 权重在测试集上的结果：

- `active_branch_gate = 2`
- `train precision = 0.9018`
- `val precision = 0.8859`
- `test precision = 0.9273`
- `test recall = 0.8097`
- `test f1 = 0.8645`
- `test avg_threshold_gap = 64.496`
- `test avg_saved_tokens_tp = 165.825`

## 预测一条 branch

适合你已经在 runtime 里自己维护好特征字段，只想拿模型推一次。

```bash
python -m predictor predict \
  --weights predictor/adaptive_threshold_weights_low_fp.json \
  --token-probs '[0.9,0.8,0.7]' \
  --topk '[{\"13\":-0.1,\"272\":-2.3},{\"15\":-0.2,\"272\":-1.9},{\"27\":-0.3}]' \
  --sibling-counts '[56]' \
  --max-other 56 \
  --n-branches 2 \
  --n-active-branches 2 \
  --step-idx 5 \
  --total-steps 10 \
  --step-total-tokens 112 \
  --prior-step-total-tokens 801 \
  --beam-width 2 \
  --policy-model 'Qwen2.5-Math-1.5B-Instruct' \
  --config-prior 0.1097 \
  --obs-point 56 \
  --current-tokens 120 \
  --active-branch-gate 2
```

返回：

- `budget`
- `threshold`
- `fired`

## 预测一个 record 的某个 step

这是最适合直接调试 `record_0.jsonl` 的入口。

```bash
python -m predictor predict-record-step \
  --record-path output/output/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/16384_2_1/question_0/record_0.jsonl \
  --step-idx 5 \
  --weights predictor/adaptive_threshold_weights_low_fp.json \
  --priors predictor/adaptive_threshold_priors.json \
  --active-branch-gate 2
```

返回字段包括：

- `has_triggered_branch`
- `triggered_branch_indices`
- 每个 branch 的：
  - `branch_idx`
  - `num_tokens`
  - `max_other`
  - `eligible_for_prediction`
  - `gate_ok`
  - `threshold`
  - `budget`
  - `threshold_fired`
  - `predicted_straggler`
  - `true_straggler`

实现入口在 [record_step_predict.py](/mnt/g/PKU/Mine/Simulator/3d-tts-simulator/predictor/predictor/record_step_predict.py#L40)。

## Python 代码直接调用

```python
from predictor.record_step_predict import predict_record_step

report = predict_record_step(
    "output/output/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/"
    "Skywork-o1-Open-PRM-Qwen-2.5-1.5B/16384_2_1/question_0/record_0.jsonl",
    step_idx=5,
    weights_path="predictor/adaptive_threshold_weights_low_fp.json",
    priors_path="predictor/adaptive_threshold_priors.json",
)

print(report["has_triggered_branch"])
print(report["triggered_branch_indices"])
for branch in report["branches"]:
    print(
        branch["branch_idx"],
        branch["predicted_straggler"],
        branch["true_straggler"],
        branch["threshold"],
    )
```

## 运行时如何接到你的 scheduler

最简单的接法：

1. 先等多数 branch 停掉，直到 `n_active_branches <= active_branch_gate`
2. branch 到达 `max_other` 时，算一次 `threshold`
3. 后续每生成一个 token，只检查：
   `current_tokens > threshold`
4. 一旦触发，就提早发 reward model 请求
5. 这个 branch 仍然可以继续 decode，不需要立即停掉
6. 后续把这个 straggler branch 和新 step 的 branch 一起放进下一轮 batch

## 当前建议
*
- 若你更关心召回：用 `predictor/adaptive_threshold_weights.json`
- 若你更关心低误报、避免冗余 RM 调用：用 `predictor/adaptive_threshold_weights_low_fp.json`*
