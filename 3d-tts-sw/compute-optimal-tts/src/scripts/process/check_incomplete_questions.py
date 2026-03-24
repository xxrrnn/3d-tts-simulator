#!/usr/bin/env python3
"""
检查输出目录中不完整或有问题的 question_* 文件夹

用法:
    python check_incomplete_questions.py [--delete] [--output-dir PATH]
    
选项:
    --delete        删除有问题的配置目录（需要确认）
    --output-dir    指定输出目录路径（默认: ../output）
    --help          显示帮助信息
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# ANSI 颜色代码
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_question_folders(base_path):
    """检查目录中的question_*文件夹是否连续"""
    results = []
    
    for root, dirs, files in os.walk(base_path):
        # 过滤出question_*目录
        question_dirs = sorted([d for d in dirs if d.startswith('question_')])
        
        if not question_dirs:
            # 检查是否是应该包含question_*的目录（比如256_4_1这样的配置目录）
            parent_name = os.path.basename(root)
            if parent_name and '_' in parent_name and parent_name.replace('_', '').isdigit():
                results.append({
                    'path': root,
                    'issue': 'NO_QUESTIONS',
                    'details': '完全没有 question_* 文件夹',
                    'severity': 'high'
                })
            continue
        
        # 提取question编号
        try:
            question_nums = []
            for d in question_dirs:
                num_str = d.replace('question_', '')
                if num_str.isdigit():
                    question_nums.append(int(num_str))
            
            if not question_nums:
                continue
                
            question_nums.sort()
            
            # 检查是否从0开始
            if question_nums[0] != 0:
                results.append({
                    'path': root,
                    'issue': 'NOT_START_ZERO',
                    'details': f'从 {question_nums[0]} 开始，应该从 0 开始',
                    'numbers': question_nums,
                    'severity': 'medium'
                })
            
            # 检查是否连续
            expected = list(range(question_nums[0], question_nums[-1] + 1))
            if question_nums != expected:
                missing = set(expected) - set(question_nums)
                severity = 'high' if len(missing) > 5 else 'low'
                results.append({
                    'path': root,
                    'issue': 'NOT_CONTINUOUS',
                    'details': f'缺失 {len(missing)} 个编号',
                    'numbers': question_nums,
                    'missing': sorted(missing),
                    'total': len(question_nums),
                    'expected_total': len(expected),
                    'severity': severity
                })
                
        except Exception as e:
            results.append({
                'path': root,
                'issue': 'ERROR',
                'details': str(e),
                'severity': 'high'
            })
    
    return results


def print_summary(results):
    """打印汇总信息"""
    stats = defaultdict(list)
    for r in results:
        stats[r['issue']].append(r)
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}📊 问题汇总统计{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    print(f"{Colors.BOLD}总问题数: {Colors.RED}{len(results)}{Colors.END}")
    print(f"  {Colors.YELLOW}●{Colors.END} 完全没有question文件夹: {len(stats['NO_QUESTIONS'])} 个")
    print(f"  {Colors.YELLOW}●{Colors.END} question编号不连续: {len(stats['NOT_CONTINUOUS'])} 个")
    print(f"  {Colors.YELLOW}●{Colors.END} question不从0开始: {len(stats['NOT_START_ZERO'])} 个")
    
    # 按数据集分组
    by_dataset = defaultdict(list)
    for r in results:
        parts = r['path'].split('/')
        for i, p in enumerate(parts):
            if 'beam_search' in p:
                by_dataset[p].append(r)
                break
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}📁 按数据集分类{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    
    for dataset, items in sorted(by_dataset.items()):
        no_q = sum(1 for x in items if x['issue'] == 'NO_QUESTIONS')
        not_cont = sum(1 for x in items if x['issue'] == 'NOT_CONTINUOUS')
        not_zero = sum(1 for x in items if x['issue'] == 'NOT_START_ZERO')
        print(f"\n{Colors.BOLD}{dataset}{Colors.END}: {len(items)} 个问题")
        print(f"  无question: {no_q} | 不连续: {not_cont} | 不从0开始: {not_zero}")


def print_detailed_list(results):
    """打印详细列表"""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}📋 详细问题列表{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    for i, r in enumerate(results, 1):
        # 简化路径显示
        path_parts = r['path'].split('/')
        rel_path = '/'.join(path_parts[-4:])
        
        # 根据严重程度选择颜色
        color = Colors.RED if r.get('severity') == 'high' else Colors.YELLOW
        
        print(f"{color}{i}. {rel_path}{Colors.END}")
        print(f"   问题类型: {r['issue']}")
        print(f"   详细信息: {r['details']}")
        
        if 'missing' in r:
            missing = r['missing']
            print(f"   缺失编号: {missing[:20]}{'...' if len(missing) > 20 else ''}")
            print(f"   完成度: {r['total']}/{r['expected_total']} ({r['total']*100//r['expected_total']}%)")
        
        print(f"   {Colors.BLUE}完整路径:{Colors.END} {r['path']}")
        print()


def prompt_delete(results):
    """询问用户是否删除"""
    print(f"\n{Colors.BOLD}{Colors.RED}⚠️  删除操作{Colors.END}\n")
    print(f"{Colors.YELLOW}以下操作将删除有问题的配置目录及其所有内容！{Colors.END}")
    print(f"{Colors.YELLOW}这个操作不可恢复！{Colors.END}\n")
    
    # 按严重程度分组
    by_severity = defaultdict(list)
    for r in results:
        by_severity[r.get('severity', 'unknown')].append(r)
    
    print(f"{Colors.BOLD}按严重程度统计:{Colors.END}")
    print(f"  {Colors.RED}●{Colors.END} 高严重 (建议删除): {len(by_severity['high'])} 个")
    print(f"  {Colors.YELLOW}●{Colors.END} 中严重: {len(by_severity['medium'])} 个")
    print(f"  {Colors.GREEN}●{Colors.END} 低严重 (可能还在运行): {len(by_severity['low'])} 个")
    
    print(f"\n{Colors.BOLD}删除选项:{Colors.END}")
    print("  1. 只删除高严重程度的 (完全没有question文件夹)")
    print("  2. 删除高+中严重程度的")
    print("  3. 删除所有有问题的配置")
    print("  4. 自定义选择")
    print("  0. 取消，不删除任何内容")
    
    while True:
        choice = input(f"\n{Colors.BOLD}请选择 [0-4]:{Colors.END} ").strip()
        
        if choice == '0':
            print(f"{Colors.GREEN}✓ 已取消删除操作{Colors.END}")
            return []
        
        elif choice == '1':
            to_delete = by_severity['high']
            break
        
        elif choice == '2':
            to_delete = by_severity['high'] + by_severity['medium']
            break
        
        elif choice == '3':
            to_delete = results
            break
        
        elif choice == '4':
            print(f"\n{Colors.CYAN}输入要删除的配置编号 (用逗号或空格分隔，如: 1,3,5 或 1-5):{Colors.END}")
            indices_input = input("编号: ").strip()
            
            try:
                indices = []
                for part in indices_input.replace(',', ' ').split():
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        indices.extend(range(start, end + 1))
                    else:
                        indices.append(int(part))
                
                to_delete = [results[i-1] for i in indices if 0 < i <= len(results)]
                break
            except:
                print(f"{Colors.RED}输入格式错误，请重新输入{Colors.END}")
                continue
        
        else:
            print(f"{Colors.RED}无效选择，请输入 0-4{Colors.END}")
    
    if not to_delete:
        print(f"{Colors.YELLOW}没有选中任何配置{Colors.END}")
        return []
    
    # 最终确认
    print(f"\n{Colors.RED}{Colors.BOLD}将要删除 {len(to_delete)} 个配置:{Colors.END}")
    for i, r in enumerate(to_delete[:10], 1):
        path_parts = r['path'].split('/')
        rel_path = '/'.join(path_parts[-4:])
        print(f"  {i}. {rel_path}")
    
    if len(to_delete) > 10:
        print(f"  ... 还有 {len(to_delete) - 10} 个")
    
    confirm = input(f"\n{Colors.BOLD}确认删除? (yes/no):{Colors.END} ").strip().lower()
    
    if confirm in ['yes', 'y']:
        return to_delete
    else:
        print(f"{Colors.GREEN}✓ 已取消删除操作{Colors.END}")
        return []


def delete_directories(to_delete):
    """删除指定的目录"""
    import shutil
    
    print(f"\n{Colors.BOLD}开始删除...{Colors.END}\n")
    
    success = 0
    failed = 0
    
    for r in to_delete:
        path = r['path']
        try:
            print(f"删除: {path}")
            shutil.rmtree(path)
            success += 1
        except Exception as e:
            print(f"{Colors.RED}✗ 失败: {e}{Colors.END}")
            failed += 1
    
    print(f"\n{Colors.BOLD}删除完成:{Colors.END}")
    print(f"  {Colors.GREEN}✓{Colors.END} 成功: {success}")
    if failed > 0:
        print(f"  {Colors.RED}✗{Colors.END} 失败: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description='检查输出目录中不完整或有问题的 question_* 文件夹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只检查，不删除
  python check_incomplete_questions.py
  
  # 检查并可选择删除
  python check_incomplete_questions.py --delete
  
  # 指定输出目录
  python check_incomplete_questions.py --output-dir /path/to/output
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
        help='指定输出目录路径（默认: 相对于脚本的 ../output）'
    )
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / 'output'
    
    output_dir = Path(output_dir).resolve()
    
    if not output_dir.exists():
        print(f"{Colors.RED}错误: 输出目录不存在: {output_dir}{Colors.END}")
        sys.exit(1)
    
    print(f"{Colors.BOLD}{Colors.CYAN}检查输出目录: {Colors.END}{output_dir}\n")
    print("扫描中...")
    
    # 检查问题
    results = check_question_folders(str(output_dir))
    
    if not results:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ 所有配置都正常，没有发现问题！{Colors.END}\n")
        return
    
    # 打印汇总
    print_summary(results)
    
    # 打印详细列表
    print_detailed_list(results)
    
    # 如果启用删除模式
    if args.delete:
        to_delete = prompt_delete(results)
        
        if to_delete:
            delete_directories(to_delete)
    else:
        print(f"\n{Colors.CYAN}💡 提示: 使用 --delete 选项可以交互式删除有问题的配置{Colors.END}\n")


if __name__ == '__main__':
    main()
