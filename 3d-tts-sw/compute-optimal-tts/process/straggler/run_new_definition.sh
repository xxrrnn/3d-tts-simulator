#!/bin/bash
# 完整流程: 使用新定义重新检测并训练

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
DATA_DIR="$SCRIPT_DIR/data"

echo "================================================================"
echo "  Straggler检测与训练 - 新定义完整流程"
echo "================================================================"
echo ""
echo "新定义特点:"
echo "  1. token数量 > 100 (基础阈值)"
echo "  2. 该分支 > (除自己外的最大token数) × 3"
echo "  3. 允许top2或更多个straggler"
echo "  4. 支持branch_count >= 2 (包括2和3)"
echo ""
echo "训练策略:"
echo "  - 按配置单独训练 (256_8_1, 16384_4_1 等)"
echo "  - 按branch数单独训练 (2, 3, 4, 8 等)"
echo "  - 按配置+branch组合训练 (最精细)"
echo "================================================================"
echo ""

# 步骤1: 重新检测stragglers
echo "步骤 1/3: 使用新定义重新检测stragglers..."
echo "  这将扫描所有workload文件并应用新的检测规则"
echo ""

cd "$SRC_DIR"
python3 detect_stragglers_new.py 2>&1 | tee "$DATA_DIR/../logs/detect_new.log"

# 检查是否成功
if [ ! -f "$DATA_DIR/stragglers_new.json" ]; then
    echo "错误: 检测失败,未找到 stragglers_new.json"
    exit 1
fi

echo ""
echo "步骤 2/3: 精细化分层训练..."
echo "  训练参数:"
echo "    - 种群大小: 100"
echo "    - 进化代数: 150"  
echo "    - 使用全部数据 (不采样)"
echo "    - 侧重Precision的适应度函数"
echo "  这将需要较长时间 (约15-40分钟)"
echo ""

# 修改训练脚本使用新的数据文件
cd "$SRC_DIR"

# 创建临时训练脚本
cat > train_with_new_data.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from precision_focused_train import train_precision_system

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    straggler_dir = os.path.dirname(script_dir)
    
    model_workloads_dir = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sim/model_workloads"
    straggler_file = os.path.join(straggler_dir, "data", "stragglers_new.json")
    output_dir = os.path.join(straggler_dir, "data")
    
    print("使用新定义的straggler数据进行训练...")
    print(f"Straggler文件: {straggler_file}")
    
    train_precision_system(
        model_workloads_dir=model_workloads_dir,
        straggler_file=straggler_file,
        output_dir=output_dir,
        population_size=100,
        generations=150,
        train_config=True,
        train_branch=True,
        train_config_branch=True
    )
PYTHON_SCRIPT

python3 train_with_new_data.py 2>&1 | tee "$DATA_DIR/../logs/precision_train_new.log"

# 清理临时文件
rm -f train_with_new_data.py

echo ""
echo "步骤 3/3: 生成对比报告..."
echo ""

python3 - << 'EOF'
import json
import os

data_dir = "../data"
results_dir = "../results"

# 读取检测结果
with open(f"{data_dir}/metadata_new.json", 'r') as f:
    metadata = json.load(f)

print("="*80)
print("检测结果统计 (新定义)")
print("="*80)
print(f"\n总步骤数: {metadata['total_steps']:,}")
print(f"Straggler数: {metadata['straggler_count']:,}")
print(f"Straggler率: {100*metadata['straggler_count']/metadata['total_steps']:.2f}%")

print(f"\n按Branch数统计:")
for branch, data in sorted(metadata['by_branch_count'].items(), key=lambda x: int(x[0])):
    if data['steps'] > 0:
        rate = 100 * data['stragglers'] / data['steps']
        print(f"  Branch {branch}: {data['stragglers']}/{data['steps']} ({rate:.2f}%)")

print(f"\n按配置统计 (Top 10):")
configs = sorted(metadata['by_config'].items(), 
                key=lambda x: x[1]['stragglers'], reverse=True)[:10]
for config, data in configs:
    if data['steps'] > 0:
        rate = 100 * data['stragglers'] / data['steps']
        print(f"  {config:15s}: {data['stragglers']:5d}/{data['steps']:6d} ({rate:5.2f}%)")

# 读取训练结果
if os.path.exists(f"{data_dir}/precision_results.json"):
    with open(f"{data_dir}/precision_results.json", 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("训练结果摘要")
    print("="*80)
    
    if 'config' in results and results['config']:
        print("\n按配置训练 (Top 5 Precision):")
        configs = sorted(results['config'].items(), 
                        key=lambda x: x[1]['final_metrics']['precision'], 
                        reverse=True)[:5]
        for config, res in configs:
            m = res['final_metrics']
            print(f"  {config:15s}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")
    
    if 'branch' in results and results['branch']:
        print("\n按Branch数训练:")
        for branch, res in sorted(results['branch'].items(), key=lambda x: int(x[0])):
            m = res['final_metrics']
            print(f"  Branch {branch}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")

print("\n" + "="*80)
EOF

echo ""
echo "================================================================"
echo "  完整流程完成!"
echo "================================================================"
echo ""
echo "生成的文件:"
echo "  检测结果:"
echo "    - data/stragglers_new.json"
echo "    - data/metadata_new.json"
echo "    - data/all_steps_new.json"
echo "    - results/detection_report_new.txt"
echo ""
echo "  训练结果:"
echo "    - data/precision_system.json"
echo "    - data/precision_results.json"
echo "    - results/precision_report.txt"
echo ""
echo "  日志:"
echo "    - logs/detect_new.log"
echo "    - logs/precision_train_new.log"
echo ""
echo "使用方法:"
echo "  from src.precision_predictor import PrecisionPredictor"
echo "  predictor = PrecisionPredictor('data/precision_system.json')"
echo ""
echo "  # 预测 (支持branch_count >= 2)"
echo "  is_strag, indices, info = predictor.predict("
echo "      step=5, branch_count=2,"
echo "      branch_tokens=[120, 450],"
echo "      config='256_2_1'"
echo "  )"
echo ""
echo "  print(f'检测到 {len(indices)} 个straggler: {indices}')"
echo "  print(f'参数来源: {info[\"params_source\"]}')"
echo ""
echo "================================================================"
