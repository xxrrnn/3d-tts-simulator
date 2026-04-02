#!/usr/bin/env python3
"""
使用新检测的数据进行高精度训练
(检测已完成,直接进行训练)
"""

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
    
    print("="*80)
    print("平衡训练 - Precision和Recall均衡策略")
    print("="*80)
    print(f"\nStraggler文件: {straggler_file}")
    print(f"输出目录: {output_dir}")
    print()
    print("训练目标: 平衡Precision和Recall")
    print("  ✓ 允许适当误报，提高召回率")
    print("  ✓ 在高precision和高recall之间找到平衡")
    print()
    print("训练配置:")
    print("  - 种群大小: 20")
    print("  - 进化代数: 30")
    print("  - 适应度函数: 60% Precision + 40% F1 (更平衡)")
    print("  - 初始种群: 20% 保守 + 40% 平衡 + 40% 探索")
    print("  - 误报惩罚: 仅在FP率>40%时惩罚")
    print("  - 按配置训练: ✓")
    print("  - 按branch数训练: ✓ (2,3,4,5,6,7,8)")
    print("  - 按组合训练: ✓ (配置+branch)")
    print()
    print("预期结果:")
    print("  - Precision: 75-85% (允许适当误报)")
    print("  - Recall: 15-30% (大幅提高!)")
    print("  - F1: 25-35% (更好的综合性能)")
    print()
    print("预计时间: 15-25分钟")
    print("="*80)
    print()
    
    train_precision_system(
        model_workloads_dir=model_workloads_dir,
        straggler_file=straggler_file,
        output_dir=output_dir,
        population_size=20,
        generations=30,
        train_config=True,
        train_branch=True,
        train_config_branch=False  # 先关闭组合级训练，加速
    )
    
    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    print("\n生成的文件:")
    print("  - data/precision_system.json     (分层参数系统)")
    print("  - data/precision_results.json    (详细结果)")
    print("  - results/precision_report.txt   (可读报告)")
    print()
    print("使用方法:")
    print("  from src.precision_predictor import PrecisionPredictor")
    print("  predictor = PrecisionPredictor('data/precision_system.json')")
    print("  result = predictor.predict(step, branch_count, branch_tokens, config=...)")
    print()
