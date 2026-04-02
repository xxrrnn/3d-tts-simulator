#!/usr/bin/env python3
"""
统计每个step的branch总token数分布
"""

import json
from pathlib import Path
from collections import Counter


def main():
    source_dir = Path('/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload/model_workloads/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/math-shepherd-mistral-7b-prm/16384_8_1')
    
    workload_files = sorted(source_dir.glob('question_*_workload.json'))
    
    # 收集每个step的总token数
    step_total_tokens = []
    step_details = []  # (question_id, step_idx, branch_count, total_tokens)
    
    for filepath in workload_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        question_id = data['question_id']
        steps = data['decode']['steps']
        
        for step_idx, step in enumerate(steps):
            branch_count = step['branch_count']
            
            # 只统计8,7,6,5 branch的步骤
            if branch_count in [8, 7, 6, 5]:
                branch_tokens = step['branch_tokens']
                total_tokens = sum(branch_tokens)
                
                step_total_tokens.append(total_tokens)
                step_details.append((question_id, step_idx, branch_count, total_tokens))
    
    # 统计分析
    step_total_tokens.sort()
    n = len(step_total_tokens)
    
    print("=" * 80)
    print("每个Step的Branch总Token数分布统计")
    print("=" * 80)
    print(f"\n总步骤数: {n}")
    print(f"\n基本统计:")
    print(f"  最小值: {min(step_total_tokens)}")
    print(f"  最大值: {max(step_total_tokens)}")
    print(f"  平均值: {sum(step_total_tokens)/n:.2f}")
    print(f"  中位数: {step_total_tokens[n//2]}")
    print(f"  25分位: {step_total_tokens[n//4]}")
    print(f"  75分位: {step_total_tokens[n*3//4]}")
    
    # 按区间统计
    print(f"\n区间分布:")
    intervals = [0, 200, 400, 600, 800, 1000, 1500, 2000, 3000, float('inf')]
    interval_names = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000-1500', '1500-2000', '2000-3000', '3000+']
    
    for i in range(len(intervals)-1):
        count = sum(1 for t in step_total_tokens if intervals[i] <= t < intervals[i+1])
        pct = count / n * 100
        bar = '█' * int(pct / 2)
        print(f"  {interval_names[i]:12s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # 按branch_count分组统计
    print(f"\n按Branch Count分组:")
    for bc in [8, 7, 6, 5]:
        bc_totals = [t for (_, _, b, t) in step_details if b == bc]
        if bc_totals:
            print(f"  {bc} branches: 平均={sum(bc_totals)/len(bc_totals):.2f}, "
                  f"范围=[{min(bc_totals)}, {max(bc_totals)}], 步数={len(bc_totals)}")
    
    # Top 10最大的steps
    print(f"\nTop 10 总Token数最多的步骤:")
    step_details.sort(key=lambda x: x[3], reverse=True)
    for i, (qid, step_idx, bc, total) in enumerate(step_details[:10], 1):
        print(f"  {i:2d}. {qid}, Step {step_idx}, {bc}-br, {total} tokens")
    
    # Top 10最小的steps
    print(f"\nTop 10 总Token数最少的步骤:")
    for i, (qid, step_idx, bc, total) in enumerate(step_details[-10:][::-1], 1):
        print(f"  {i:2d}. {qid}, Step {step_idx}, {bc}-br, {total} tokens")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
