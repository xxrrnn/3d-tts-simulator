# Question 文件夹完整性检查工具

这个工具用于检查 beam search 输出目录中的 `question_*` 文件夹是否完整、连续。

## 功能

1. **自动扫描**：递归扫描所有 beam_search 输出目录
2. **问题检测**：
   - 检测完全没有 question 文件夹的配置
   - 检测 question 编号不连续的情况
   - 检测 question 编号不从 0 开始的情况
3. **交互式删除**：可以选择性地删除有问题的配置目录
4. **彩色输出**：使用颜色高亮显示问题严重程度

## 使用方法

### 1. 只检查，不删除

```bash
cd /root/autodl-tmp/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/src/scripts/process
./check_questions.sh
```

或直接运行 Python 脚本：

```bash
python3 check_incomplete_questions.py
```

### 2. 检查并可选择删除

```bash
./check_questions.sh --delete
```

脚本会：
1. 显示所有问题的详细列表
2. 按严重程度分类
3. 询问你要删除哪些配置
4. 要求最终确认后才执行删除

### 3. 自定义输出目录

```bash
python3 check_incomplete_questions.py --output-dir /path/to/custom/output
```

### 4. 查看帮助

```bash
python3 check_incomplete_questions.py --help
```

## 输出说明

### 问题类型

- **NO_QUESTIONS** (高严重): 完全没有 question_* 文件夹
  - 可能是任务未运行或运行失败
  - 建议删除

- **NOT_CONTINUOUS** (低/高严重): question 编号不连续
  - 缺失少量 (≤5个): 低严重，可能还在运行
  - 缺失大量 (>5个): 高严重，可能出错
  - 根据具体情况决定是否删除

- **NOT_START_ZERO** (中严重): question 编号不从 0 开始
  - 可能是部分完成的任务
  - 建议检查日志确认原因

### 输出示例

```
📊 问题汇总统计
================================================================================

总问题数: 58
  ● 完全没有question文件夹: 23 个
  ● question编号不连续: 28 个
  ● question不从0开始: 7 个

📁 按数据集分类
================================================================================

MATH_beam_search: 26 个问题
  无question: 11 | 不连续: 14 | 不从0开始: 1

AMC23_beam_search: 16 个问题
  无question: 6 | 不连续: 7 | 不从0开始: 3

AIME24_beam_search: 16 个问题
  无question: 6 | 不连续: 7 | 不从0开始: 3
```

## 删除模式

使用 `--delete` 选项时，脚本会提供以下删除选项：

1. **只删除高严重程度的** (完全没有question文件夹)
   - 最安全的选择
   - 这些配置肯定是有问题的

2. **删除高+中严重程度的**
   - 包括完全没有question和不从0开始的

3. **删除所有有问题的配置**
   - 会删除所有检测到的问题配置
   - 包括可能还在运行的任务

4. **自定义选择**
   - 可以指定具体的编号，如: `1,3,5` 或 `1-10`

5. **取消**
   - 不删除任何内容

### 安全机制

- 删除前会显示详细列表
- 需要输入 `yes` 最终确认
- 所有删除操作都会显示结果

## 注意事项

⚠️ **警告**：删除操作不可恢复！

- 删除前请确认这些配置确实不需要
- 如果任务还在运行，可能会检测到"不连续"的问题
- 建议先只检查，确认问题后再使用删除功能
- 可以先用选项1（只删除高严重程度）进行保守清理

## 技术细节

- 问题严重程度判断：
  - `high`: 完全没有question文件夹，或缺失>5个编号
  - `medium`: question不从0开始
  - `low`: 缺失≤5个编号（可能还在运行）

- 检查逻辑：
  - 递归扫描所有包含 `*_beam_search` 的目录
  - 识别格式为 `{数字}_{数字}_{数字}` 的配置目录
  - 检查每个配置目录中的 `question_*` 文件夹

## 示例工作流

```bash
# 1. 先检查问题
./check_questions.sh

# 2. 如果确认需要清理，启用删除模式
./check_questions.sh --delete

# 3. 选择删除选项（建议先选1，只删除高严重程度）
请选择 [0-4]: 1

# 4. 确认删除
确认删除? (yes/no): yes
```

## 文件说明

### Question 完整性检查
- `check_incomplete_questions.py`: 检查 question 文件夹完整性的主脚本
- `check_questions.sh`: Bash 包装脚本，方便调用
- `quick_check.sh`: 快速统计脚本

### 无效组合清理
- `remove_invalid_combinations.py`: 删除不在配置列表中的模型组合
- `remove_invalid.sh`: Bash 包装脚本，方便调用

### 文档
- `README.md`: 本说明文档
- `QUICKSTART.md`: 快速开始指南
