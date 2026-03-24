#!/usr/bin/env python3
"""
删除不在配置列表中的模型组合目录

根据 eval_all_combinations.sh 中的 POLICY_MODELS 和 REWARD_MODELS 列表，
删除 output 目录中不符合配置的组合。
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# ANSI 颜色代码
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

# 配置的 Policy 模型列表（来自 eval_all_combinations.sh）
POLICY_MODELS = [
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-Math-1.5B-Instruct",
    "Qwen2.5-Math-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B"
]

# 配置的 Reward 模型列表（来自 eval_all_combinations.sh）
REWARD_MODELS = [
    "Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
    "Skywork-o1-Open-PRM-Qwen-2.5-7B",
    "math-shepherd-mistral-7b-prm",
    "Qwen2.5-Math-PRM-7B"
]


def find_invalid_combinations(output_dir):
    """查找不在配置列表中的组合"""
    invalid_combinations = []
    
    if not os.path.exists(output_dir):
        print(f"{Colors.RED}错误: 输出目录不存在: {output_dir}{Colors.END}")
        return invalid_combinations
    
    # 遍历 output 目录
    for dataset in os.listdir(output_dir):
        dataset_path = os.path.join(output_dir, dataset)
        if not os.path.isdir(dataset_path) or 'beam_search' not in dataset:
            continue
        
        for policy_model in os.listdir(dataset_path):
            policy_path = os.path.join(dataset_path, policy_model)
            if not os.path.isdir(policy_path):
                continue
            
            for reward_model in os.listdir(policy_path):
                reward_path = os.path.join(policy_path, reward_model)
                if not os.path.isdir(reward_path):
                    continue
                
                # 检查这个组合是否在配置列表中
                policy_valid = policy_model in POLICY_MODELS
                reward_valid = reward_model in REWARD_MODELS
                
                if not policy_valid or not reward_valid:
                    # 计算目录大小
                    try:
                        total_size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(reward_path)
                            for filename in filenames
                        )
                        size_mb = total_size / (1024 * 1024)
                    except:
                        size_mb = 0
                    
                    invalid_combinations.append({
                        'dataset': dataset,
                        'policy': policy_model,
                        'reward': reward_model,
                        'policy_valid': policy_valid,
                        'reward_valid': reward_valid,
                        'path': reward_path,
                        'size_mb': size_mb
                    })
    
    return invalid_combinations


def print_invalid_combinations(invalid_combinations):
    """打印无效组合的详细信息"""
    if not invalid_combinations:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ 所有组合都在配置列表中！{Colors.END}\n")
        return
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}❌ 发现 {len(invalid_combinations)} 个不在配置列表中的组合{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    # 按问题类型分组
    invalid_policy = [c for c in invalid_combinations if not c['policy_valid']]
    invalid_reward = [c for c in invalid_combinations if not c['reward_valid'] and c['policy_valid']]
    both_invalid = [c for c in invalid_combinations if not c['policy_valid'] and not c['reward_valid']]
    
    total_size = sum(c['size_mb'] for c in invalid_combinations)
    
    if invalid_policy:
        print(f"{Colors.YELLOW}=== Policy 模型不在列表中 ({len(invalid_policy)} 个) ==={Colors.END}")
        for i, c in enumerate(invalid_policy, 1):
            print(f"{Colors.RED}{i}. {c['dataset']}/{c['policy']}/{c['reward']}{Colors.END}")
            print(f"   大小: {c['size_mb']:.2f} MB")
            print(f"   路径: {c['path']}")
        print()
    
    if invalid_reward:
        print(f"{Colors.YELLOW}=== Reward 模型不在列表中 ({len(invalid_reward)} 个) ==={Colors.END}")
        for i, c in enumerate(invalid_reward, 1):
            print(f"{Colors.RED}{i}. {c['dataset']}/{c['policy']}/{c['reward']}{Colors.END}")
            print(f"   大小: {c['size_mb']:.2f} MB")
            print(f"   路径: {c['path']}")
        print()
    
    if both_invalid:
        print(f"{Colors.YELLOW}=== Policy 和 Reward 都不在列表中 ({len(both_invalid)} 个) ==={Colors.END}")
        for i, c in enumerate(both_invalid, 1):
            print(f"{Colors.RED}{i}. {c['dataset']}/{c['policy']}/{c['reward']}{Colors.END}")
            print(f"   大小: {c['size_mb']:.2f} MB")
            print(f"   路径: {c['path']}")
        print()
    
    # 汇总统计
    print(f"{Colors.BOLD}=== 汇总统计 ==={Colors.END}")
    unique_policies = set(c['policy'] for c in invalid_combinations if not c['policy_valid'])
    unique_rewards = set(c['reward'] for c in invalid_combinations if not c['reward_valid'])
    
    if unique_policies:
        print(f"不在列表中的 Policy 模型: {Colors.CYAN}{', '.join(sorted(unique_policies))}{Colors.END}")
    if unique_rewards:
        print(f"不在列表中的 Reward 模型: {Colors.CYAN}{', '.join(sorted(unique_rewards))}{Colors.END}")
    
    print(f"\n{Colors.BOLD}总计大小: {Colors.RED}{total_size:.2f} MB ({total_size/1024:.2f} GB){Colors.END}")


def prompt_delete(invalid_combinations):
    """询问用户是否删除"""
    if not invalid_combinations:
        return []
    
    print(f"\n{Colors.BOLD}{Colors.RED}⚠️  删除操作{Colors.END}\n")
    print(f"{Colors.YELLOW}这些目录不在 eval_all_combinations.sh 的配置列表中！{Colors.END}")
    print(f"{Colors.YELLOW}删除操作不可恢复！{Colors.END}\n")
    
    print(f"{Colors.BOLD}删除选项:{Colors.END}")
    print("  1. 删除所有无效组合")
    print("  2. 自定义选择要删除的组合")
    print("  0. 取消，不删除任何内容")
    
    while True:
        choice = input(f"\n{Colors.BOLD}请选择 [0-2]:{Colors.END} ").strip()
        
        if choice == '0':
            print(f"{Colors.GREEN}✓ 已取消删除操作{Colors.END}")
            return []
        
        elif choice == '1':
            to_delete = invalid_combinations
            break
        
        elif choice == '2':
            print(f"\n{Colors.CYAN}输入要删除的组合编号 (用逗号或空格分隔，如: 1,3,5 或 1-5):{Colors.END}")
            print(f"{Colors.CYAN}可用编号: 1-{len(invalid_combinations)}{Colors.END}")
            indices_input = input("编号: ").strip()
            
            try:
                indices = []
                for part in indices_input.replace(',', ' ').split():
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        indices.extend(range(start, end + 1))
                    else:
                        indices.append(int(part))
                
                to_delete = [invalid_combinations[i-1] for i in indices if 0 < i <= len(invalid_combinations)]
                break
            except:
                print(f"{Colors.RED}输入格式错误，请重新输入{Colors.END}")
                continue
        
        else:
            print(f"{Colors.RED}无效选择，请输入 0-2{Colors.END}")
    
    if not to_delete:
        print(f"{Colors.YELLOW}没有选中任何组合{Colors.END}")
        return []
    
    # 最终确认
    total_size = sum(c['size_mb'] for c in to_delete)
    print(f"\n{Colors.RED}{Colors.BOLD}将要删除 {len(to_delete)} 个组合 (共 {total_size:.2f} MB):{Colors.END}")
    for i, c in enumerate(to_delete[:10], 1):
        print(f"  {i}. {c['dataset']}/{c['policy']}/{c['reward']} ({c['size_mb']:.2f} MB)")
    
    if len(to_delete) > 10:
        print(f"  ... 还有 {len(to_delete) - 10} 个")
    
    confirm = input(f"\n{Colors.BOLD}确认删除? 输入 'DELETE' 确认:{Colors.END} ").strip()
    
    if confirm == 'DELETE':
        return to_delete
    else:
        print(f"{Colors.GREEN}✓ 已取消删除操作{Colors.END}")
        return []


def delete_directories(to_delete):
    """删除指定的目录"""
    print(f"\n{Colors.BOLD}开始删除...{Colors.END}\n")
    
    success = 0
    failed = 0
    total_freed = 0
    
    for c in to_delete:
        path = c['path']
        size_mb = c['size_mb']
        try:
            print(f"删除: {path} ({size_mb:.2f} MB)")
            shutil.rmtree(path)
            success += 1
            total_freed += size_mb
            print(f"  {Colors.GREEN}✓ 成功{Colors.END}")
        except Exception as e:
            print(f"  {Colors.RED}✗ 失败: {e}{Colors.END}")
            failed += 1
    
    print(f"\n{Colors.BOLD}删除完成:{Colors.END}")
    print(f"  {Colors.GREEN}✓{Colors.END} 成功: {success}")
    if failed > 0:
        print(f"  {Colors.RED}✗{Colors.END} 失败: {failed}")
    print(f"  {Colors.CYAN}释放空间: {total_freed:.2f} MB ({total_freed/1024:.2f} GB){Colors.END}")


def main():
    parser = argparse.ArgumentParser(
        description='删除不在配置列表中的模型组合目录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只检查，不删除
  python remove_invalid_combinations.py
  
  # 检查并删除
  python remove_invalid_combinations.py --delete
  
  # 指定输出目录
  python remove_invalid_combinations.py --output-dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--delete',
        action='store_true',
        help='启用删除模式（会询问确认）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='指定输出目录路径（默认: ../output）'
    )
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / 'output'
    
    output_dir = Path(output_dir).resolve()
    
    print(f"{Colors.BOLD}{Colors.CYAN}检查输出目录: {Colors.END}{output_dir}\n")
    print("扫描中...")
    
    # 查找无效组合
    invalid_combinations = find_invalid_combinations(str(output_dir))
    
    # 打印结果
    print_invalid_combinations(invalid_combinations)
    
    # 如果启用删除模式
    if args.delete and invalid_combinations:
        to_delete = prompt_delete(invalid_combinations)
        
        if to_delete:
            delete_directories(to_delete)
    elif not args.delete and invalid_combinations:
        print(f"\n{Colors.CYAN}💡 提示: 使用 --delete 选项可以删除这些无效组合{Colors.END}\n")


if __name__ == '__main__':
    main()
