#!/usr/bin/env python3
"""
验证 branch_parent_indices 的正确性
"""
import json
import sys

def verify_parent_indices(workload_file):
    """验证parent_indices的正确性"""
    with open(workload_file, 'r') as f:
        workload = json.load(f)
    
    print(f"验证文件: {workload_file}")
    print("="*80)
    
    errors = []
    warnings = []
    
    for step_idx in range(len(workload['decode_steps'])):
        step = workload['decode_steps'][step_idx]
        
        # 验证1: Step 0 所有parent应该是-1
        if step_idx == 0:
            for i, parent in enumerate(step['branch_parent_indices']):
                if parent != -1:
                    errors.append(f"Step {step_idx} 分支{i}: parent应该是-1，实际是{parent}")
        
        # 验证2: Step > 0 的parent应该指向有效的上一步分支索引
        if step_idx > 0:
            prev_step = workload['decode_steps'][step_idx - 1]
            prev_branch_count = prev_step['branch_count']
            
            for i, parent in enumerate(step['branch_parent_indices']):
                if parent < 0 or parent >= prev_branch_count:
                    errors.append(
                        f"Step {step_idx} 分支{i}: parent={parent} 超出上一步范围[0, {prev_branch_count-1}]"
                    )
                else:
                    # 验证3: parent_value应该匹配上一步的reward
                    # (从原始record中获取parent_value进行验证，这里我们跳过)
                    pass
        
        # 验证4: selected分支应该在下一步有对应的parent
        if step_idx < len(workload['decode_steps']) - 1:
            next_step = workload['decode_steps'][step_idx + 1]
            next_parents = set(next_step['branch_parent_indices'])
            
            # 实际被扩展的分支
            actually_expanded = sorted(next_parents)
            selected = step['selected_branch_indices']
            
            # selected分支中，哪些真正被扩展了
            expanded_from_selected = [idx for idx in selected if idx in next_parents]
            not_expanded = [idx for idx in selected if idx not in next_parents]
            
            if not_expanded:
                warnings.append(
                    f"Step {step_idx}: selected={selected} 中的 {not_expanded} "
                    f"未被扩展到下一步（正常现象，因为全局排序）"
                )
            
            print(f"\nStep {step_idx}:")
            print(f"  selected: {selected}")
            print(f"  实际扩展到下一步: {actually_expanded}")
            print(f"  扩展率: {len(actually_expanded)}/{len(selected)}")
    
    print("\n" + "="*80)
    
    if errors:
        print("❌ 发现错误:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ parent_indices验证通过!")
    
    if warnings:
        print("\n⚠️  注意事项:")
        for warning in warnings[:3]:  # 只显示前3个warning
            print(f"  - {warning}")
        if len(warnings) > 3:
            print(f"  ... (还有{len(warnings)-3}个类似情况)")
    
    return True

def test_path_tracing(workload_file):
    """测试路径回溯功能"""
    with open(workload_file, 'r') as f:
        workload = json.load(f)
    
    print("\n" + "="*80)
    print("测试路径回溯功能")
    print("="*80)
    
    # 找到最后一个有分支的步骤
    final_step = None
    final_step_num = -1
    for i in range(len(workload['decode_steps']) - 1, -1, -1):
        step = workload['decode_steps'][i]
        if step['branch_count'] > 0 and len(step['branch_rewards']) > 0:
            final_step = step
            final_step_num = step['step']
            break
    
    if not final_step:
        print("⚠️  没有找到有效的分支")
        return True
    
    # 选择reward最高的分支
    best_idx = final_step['branch_rewards'].index(max(final_step['branch_rewards']))
    
    print(f"\n追踪最优分支: Step {final_step_num} / 分支{best_idx}")
    print(f"Reward: {final_step['branch_rewards'][best_idx]:.4f}\n")
    
    # 回溯路径
    path = [(final_step_num, best_idx)]
    current_step = final_step_num
    current_branch = best_idx
    
    while current_step > 0:
        parent_idx = workload['decode_steps'][current_step]['branch_parent_indices'][current_branch]
        if parent_idx == -1:
            break
        current_step -= 1
        current_branch = parent_idx
        path.insert(0, (current_step, current_branch))
    
    print("完整路径:")
    for step_idx, branch_idx in path:
        step = workload['decode_steps'][step_idx]
        reward = step['branch_rewards'][branch_idx]
        tokens = step['branch_tokens'][branch_idx]
        print(f"  Step {step_idx}/分支{branch_idx}: tokens={tokens:3d}, reward={reward:.6f}")
    
    print("\n✅ 路径回溯功能正常!")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        workload_file = sys.argv[1]
    else:
        workload_file = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload/output_beam/AIME24/16384_8_2/question_17/workload_0.json"
    
    success = verify_parent_indices(workload_file)
    if success:
        test_path_tracing(workload_file)
    
    sys.exit(0 if success else 1)
