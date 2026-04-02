#!/usr/bin/env python3
"""
高精度预测器 - 使用分层训练的参数进行预测
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent))
from adaptive_threshold import ThresholdParams


class PrecisionPredictor:
    """高精度straggler预测器"""
    
    def __init__(self, system_file: str):
        """
        初始化预测器
        
        Args:
            system_file: precision_system.json文件路径
        """
        self.system_file = system_file
        self.global_params = None
        self.config_params = {}
        self.branch_params = {}
        self.config_branch_params = {}
        
        self._load_system()
    
    def _load_system(self):
        """加载分层参数系统"""
        with open(self.system_file, 'r') as f:
            data = json.load(f)
        
        if data.get('global_params'):
            self.global_params = ThresholdParams.from_dict(data['global_params'])
        
        self.config_params = {
            k: ThresholdParams.from_dict(v) 
            for k, v in data.get('config_params', {}).items()
        }
        
        self.branch_params = {
            int(k): ThresholdParams.from_dict(v) 
            for k, v in data.get('branch_params', {}).items()
        }
        
        self.config_branch_params = {
            k: ThresholdParams.from_dict(v) 
            for k, v in data.get('config_branch_params', {}).items()
        }
        
        print(f"加载高精度系统:")
        print(f"  配置参数: {len(self.config_params)}")
        print(f"  Branch参数: {len(self.branch_params)}")
        print(f"  组合参数: {len(self.config_branch_params)}")
    
    def get_params(self, config: str = None, branch_count: int = None) -> ThresholdParams:
        """
        获取最适合的参数
        
        优先级: config+branch > config > branch > global
        """
        # 1. 最精细: 配置+branch组合
        if config and branch_count:
            key = f"{config}_b{branch_count}"
            if key in self.config_branch_params:
                return self.config_branch_params[key]
        
        # 2. 按配置
        if config and config in self.config_params:
            return self.config_params[config]
        
        # 3. 按branch数
        if branch_count and branch_count in self.branch_params:
            return self.branch_params[branch_count]
        
        # 4. 全局参数
        if self.global_params:
            return self.global_params
        
        # 5. 默认保守参数(高精度倾向)
        return ThresholdParams(
            base_ratio=3.0,
            base_length=100.0,
            step_factor=0.0,
            branch_factor=0.0,
            std_factor=0.5,
            avg_factor=0.2,
            history_factor=0.1
        )
    
    def predict(self,
                step: int,
                branch_count: int,
                branch_tokens: List[int],
                question_id: str = None,
                config: str = None,
                history_max: float = 0.0) -> Tuple[bool, List[int], Dict]:
        """
        预测是否有straggler (使用新定义)
        
        新定义:
        - token数量 > 100 (或动态阈值)
        - 该分支 > (除自己外的最大token数) × 3 (或动态比值)
        - 允许top2或更多个straggler
        - 支持branch_count >= 2
        
        Args:
            step: 当前步骤
            branch_count: 分支数量
            branch_tokens: 各分支的token数列表
            question_id: 问题ID (可选)
            config: 配置名称,如 "256_8_1" (可选,用于选择最佳参数)
            history_max: 历史最大token数 (可选)
        
        Returns:
            (is_straggler, straggler_indices, info)
            - is_straggler: 是否检测到straggler
            - straggler_indices: straggler分支的索引列表
            - info: 额外信息字典
        """
        if branch_count < 2 or len(branch_tokens) < 2:
            # 不满足基本条件
            return False, [], {
                'reason': 'branch_count < 2',
                'params_source': 'none'
            }
        
        # 获取适合的参数
        params = self.get_params(config, branch_count)
        
        # 确定参数来源(用于调试)
        params_source = 'default'
        if config and branch_count:
            key = f"{config}_b{branch_count}"
            if key in self.config_branch_params:
                params_source = f'config_branch:{key}'
        if params_source == 'default' and config and config in self.config_params:
            params_source = f'config:{config}'
        if params_source == 'default' and branch_count in self.branch_params:
            params_source = f'branch:{branch_count}'
        if params_source == 'default' and self.global_params:
            params_source = 'global'
        
        # 使用新的检测逻辑
        is_straggler, indices = self._detect_straggler_new(
            branch_tokens, params, step, branch_count, history_max
        )
        
        # 返回详细信息
        info = {
            'params_source': params_source,
            'config': config,
            'branch_count': branch_count,
            'step': step,
            'base_ratio': params.base_ratio,
            'base_length': params.base_length,
            'straggler_count': len(indices)
        }
        
        return is_straggler, indices, info
    
    def _detect_straggler_new(self, branch_tokens: List[int], params: ThresholdParams,
                             step: int, branch_count: int, history_max: float) -> Tuple[bool, List[int]]:
        """
        新的straggler检测逻辑
        
        定义：对于每个分支，如果：
        1. token数量 > 动态长度阈值
        2. 该分支token数 > (除自己外的最大token数) × 动态比值阈值
        则该分支是straggler
        
        允许有多个straggler (top2或更多的情况)
        """
        if len(branch_tokens) < 2:
            return False, []
        
        # 计算动态阈值
        avg_tokens = sum(branch_tokens) / len(branch_tokens)
        std_tokens = 0
        if len(branch_tokens) > 1:
            variance = sum((x - avg_tokens) ** 2 for x in branch_tokens) / len(branch_tokens)
            std_tokens = variance ** 0.5
        
        ratio_threshold = params.base_ratio + \
                         params.step_factor * step + \
                         params.branch_factor * branch_count + \
                         params.std_factor * std_tokens / (avg_tokens + 1)
        
        length_threshold = params.base_length + \
                          params.avg_factor * avg_tokens + \
                          params.history_factor * history_max
        
        # 检查每个分支
        straggler_indices = []
        for i, token in enumerate(branch_tokens):
            if token <= length_threshold:
                continue
            
            # 找到除当前分支外的最大值
            other_tokens = branch_tokens[:i] + branch_tokens[i+1:]
            if not other_tokens:
                continue
            
            max_other = max(other_tokens)
            
            # 检查是否满足straggler条件
            if max_other > 0 and token >= max_other * ratio_threshold:
                straggler_indices.append(i)
        
        return len(straggler_indices) > 0, straggler_indices
    
    def predict_workload(self, workload_file: str) -> Dict:
        """
        预测整个workload文件
        
        Args:
            workload_file: workload JSON文件路径
        
        Returns:
            预测结果字典
        """
        with open(workload_file, 'r') as f:
            workload = json.load(f)
        
        question_id = workload.get('question_id', 'unknown')
        metadata = workload.get('metadata', {})
        config = metadata.get('config', None)
        
        results = {
            'question_id': question_id,
            'config': config,
            'predictions': []
        }
        
        history_max = 0.0
        
        for step_data in workload.get('steps', []):
            step = step_data['step']
            branch_count = step_data.get('branch_count', 0)
            branch_tokens = step_data.get('branch_tokens', [])
            
            is_straggler, indices, info = self.predict(
                step, branch_count, branch_tokens,
                question_id, config, history_max
            )
            
            if is_straggler:
                results['predictions'].append({
                    'step': step,
                    'straggler_indices': indices,
                    'branch_tokens': branch_tokens,
                    'info': info
                })
            
            if branch_tokens:
                history_max = max(history_max, max(branch_tokens))
        
        return results


def demo():
    """演示高精度预测器"""
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    straggler_dir = os.path.dirname(script_dir)
    system_file = os.path.join(straggler_dir, "data", "precision_system.json")
    
    if not os.path.exists(system_file):
        print(f"错误: 找不到 {system_file}")
        print("请先运行: python3 precision_focused_train.py")
        return
    
    print("="*80)
    print("高精度Straggler预测器演示")
    print("="*80)
    
    # 初始化预测器
    predictor = PrecisionPredictor(system_file)
    
    # 示例1: 使用配置信息
    print("\n示例1: 使用配置 '256_8_1', branch=8")
    is_strag, indices, info = predictor.predict(
        step=5,
        branch_count=8,
        branch_tokens=[120, 130, 650, 125, 115, 140, 135, 128],
        config="256_8_1"
    )
    print(f"  检测到straggler: {is_strag}")
    print(f"  Straggler分支: {indices}")
    print(f"  参数来源: {info['params_source']}")
    print(f"  使用阈值: ratio={info['base_ratio']:.2f}, length={info['base_length']:.1f}")
    
    # 示例2: 使用配置信息
    print("\n示例2: 使用配置 '16384_4_1', branch=4")
    is_strag, indices, info = predictor.predict(
        step=10,
        branch_count=4,
        branch_tokens=[200, 850, 210, 195],
        config="16384_4_1"
    )
    print(f"  检测到straggler: {is_strag}")
    print(f"  Straggler分支: {indices}")
    print(f"  参数来源: {info['params_source']}")
    print(f"  使用阈值: ratio={info['base_ratio']:.2f}, length={info['base_length']:.1f}")
    
    # 示例3: 只提供branch数
    print("\n示例3: 只提供 branch=4 (无配置)")
    is_strag, indices, info = predictor.predict(
        step=3,
        branch_count=4,
        branch_tokens=[100, 450, 105, 98]
    )
    print(f"  检测到straggler: {is_strag}")
    print(f"  Straggler分支: {indices}")
    print(f"  参数来源: {info['params_source']}")
    print(f"  使用阈值: ratio={info['base_ratio']:.2f}, length={info['base_length']:.1f}")
    
    # 示例4: 正常情况(无straggler)
    print("\n示例4: 正常情况(所有分支长度相似)")
    is_strag, indices, info = predictor.predict(
        step=2,
        branch_count=4,
        branch_tokens=[100, 105, 103, 98],
        config="256_4_1"
    )
    print(f"  检测到straggler: {is_strag}")
    print(f"  Straggler分支: {indices}")
    print(f"  参数来源: {info['params_source']}")
    
    print("\n" + "="*80)
    print("演示完成!")
    print("="*80)


if __name__ == "__main__":
    demo()
