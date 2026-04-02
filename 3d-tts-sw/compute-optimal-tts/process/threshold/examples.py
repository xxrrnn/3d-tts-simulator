#!/usr/bin/env python3
"""
动态剪枝器使用示例

包含:
1. 正确使用示例 - 只使用历史reward
2. 集成示例 - 如何集成到beam search
"""

from dynamic_pruner import DynamicPruner
from typing import List, Dict
import copy


# ============================================================================
# Part 1: 正确使用示例 - 只使用历史reward
# ============================================================================

def example_correct_usage():
    """正确的使用方式 - 关键点: 只使用历史reward"""
    
    print("="*80)
    print("示例1: 正确使用 - 只使用历史reward")
    print("="*80)
    
    pruner = DynamicPruner("threshold_model.json")
    
    # 模拟beam search
    branches = [
        {'id': 0, 'rewards': [], 'token_probs': [], 'cum_probs': []},
        {'id': 1, 'rewards': [], 'token_probs': [], 'cum_probs': []}
    ]
    
    max_steps = 5
    
    for step in range(max_steps):
        print(f"\n{'='*60}")
        print(f"Step {step}")
        print(f"{'='*60}")
        
        # === 关键点1: 在生成token之前做剪枝决策 ===
        kept_branches = []
        
        for branch in branches:
            print(f"分支 {branch['id']}: 历史reward={branch['rewards']}")
            
            branch_data = {
                'reward_history': branch['rewards'],  # 不包含当前step
                'token_prob_history': branch['token_probs'],
                'prob_history': branch['cum_probs']
            }
            
            all_branches_data = [
                {'reward_history': b['rewards'],
                 'token_prob_history': b['token_probs'],
                 'prob_history': b['cum_probs']}
                for b in branches
            ]
            
            should_prune, reason = pruner.should_prune_branch(
                branch_data, all_branches_data, step
            )
            
            if not should_prune:
                kept_branches.append(branch)
                print(f"  -> 保留")
            else:
                print(f"  -> 剪枝 ({reason[:50]}...)")
        
        if len(kept_branches) == 0:
            kept_branches = branches[:1]
        
        branches = kept_branches
        
        # === 关键点2: 对保留的分支生成token和计算reward ===
        for branch in branches:
            import random
            new_reward = (0.6 if branch['id'] == 0 else 0.2) + 0.05 * step + random.uniform(-0.05, 0.05)
            branch['rewards'].append(new_reward)
            branch['token_probs'].append([0.9 + 0.01 * step] * 10)
            branch['cum_probs'].append((branch['cum_probs'][-1] if branch['cum_probs'] else 1.0) * 0.95)
    
    print(f"\n{'='*60}")
    print(f"完成! 保留 {len(branches)} 个分支")
    stats = pruner.get_stats()
    print(f"剪枝率: {stats['prune_rate']:.1%}")


# ============================================================================
# Part 2: 集成示例 - Beam Search集成
# ============================================================================

class BeamSearchWithPruning:
    """带动态剪枝的Beam Search示例"""
    
    def __init__(self, beam_size=8, max_steps=100, pruner_model_path=None):
        self.beam_size = beam_size
        self.max_steps = max_steps
        
        if pruner_model_path:
            self.pruner = DynamicPruner(pruner_model_path)
            print(f"✓ 动态剪枝器已启用")
        else:
            self.pruner = None
    
    def generate(self, question: str) -> List[Dict]:
        """生成答案候选"""
        beams = [{
            'text': question,
            'reward_history': [],
            'token_prob_history': [],
            'prob_history': [],
            'score': 0.0
        }]
        
        for step in range(self.max_steps):
            # 1. 扩展beams
            new_beams = self._expand_beams(beams)
            
            # 2. 应用动态剪枝
            if self.pruner:
                new_beams = self._apply_pruning(new_beams, step)
            
            # 3. 保留top-k
            new_beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:self.beam_size]
            beams = new_beams
            
            if all(b.get('finished', False) for b in beams):
                break
        
        return beams
    
    def _expand_beams(self, beams: List[Dict]) -> List[Dict]:
        """扩展beams (占位实现)"""
        new_beams = []
        for beam in beams:
            for i in range(2):  # 每个beam生成2个候选
                new_beam = copy.deepcopy(beam)
                # 模拟生成和计算reward
                import random
                new_reward = random.random()
                new_beam['reward_history'].append(new_reward)
                new_beam['token_prob_history'].append([random.random()] * 10)
                new_beam['prob_history'].append(random.random())
                new_beam['score'] = new_reward
                new_beams.append(new_beam)
        return new_beams
    
    def _apply_pruning(self, beams: List[Dict], step: int) -> List[Dict]:
        """应用动态剪枝"""
        kept_beams = []
        
        for beam in beams:
            branch_data = {
                'reward_history': beam['reward_history'],
                'token_prob_history': beam['token_prob_history'],
                'prob_history': beam['prob_history']
            }
            
            all_branches = [
                {'reward_history': b['reward_history'],
                 'token_prob_history': b['token_prob_history'],
                 'prob_history': b['prob_history']}
                for b in beams
            ]
            
            should_prune, _ = self.pruner.should_prune_branch(
                branch_data, all_branches, step
            )
            
            if not should_prune:
                kept_beams.append(beam)
        
        # 确保至少保留一个
        if len(kept_beams) == 0:
            kept_beams = [max(beams, key=lambda x: x['score'])]
        
        return kept_beams


def example_beam_search_integration():
    """Beam Search集成示例"""
    
    print("\n" + "="*80)
    print("示例2: Beam Search集成")
    print("="*80)
    
    pruner_path = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold/threshold_model.json"
    
    beam_search = BeamSearchWithPruning(
        beam_size=4,
        max_steps=3,
        pruner_model_path=pruner_path
    )
    
    print("\n运行beam search (3步)...")
    results = beam_search.generate("示例问题")
    
    print(f"\n完成! 得到 {len(results)} 个候选答案")
    print(f"最佳得分: {max(r['score'] for r in results):.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("动态剪枝器使用示例\n")
    
    # 示例1: 正确使用方式
    example_correct_usage()
    
    # 示例2: Beam Search集成
    example_beam_search_integration()
    
    print("\n" + "="*80)
    print("关键要点:")
    print("="*80)
    print("1. ✓ 在生成token之前做剪枝决策")
    print("2. ✓ 只使用历史reward (不含当前step)")
    print("3. ✓ 对保留的分支才生成token和计算reward")
    print("4. ✓ 将新reward添加到历史中")
    print("="*80)


if __name__ == '__main__':
    main()

    """正确的使用方式"""
    
    print("="*80)
    print("正确使用示例: 只使用历史reward")
    print("="*80)
    
    # 初始化剪枝器
    pruner = DynamicPruner("threshold_model.json")
    
    # 模拟beam search过程
    branches = [
        {
            'id': 0,
            'rewards': [],  # 历史reward
            'token_probs': [],
            'cum_probs': []
        },
        {
            'id': 1,
            'rewards': [],
            'token_probs': [],
            'cum_probs': []
        }
    ]
    
    max_steps = 5
    
    for step in range(max_steps):
        print(f"\n{'='*80}")
        print(f"Step {step}: 决策是否剪枝")
        print(f"{'='*80}")
        
        # === 关键点1: 在生成token之前做剪枝决策 ===
        kept_branches = []
        
        for branch in branches:
            print(f"\n分支 {branch['id']}:")
            print(f"  历史reward: {branch['rewards']}")
            print(f"  历史长度: {len(branch['rewards'])}")
            
            # 准备剪枝器输入 - 只包含历史数据
            branch_data = {
                'reward_history': branch['rewards'],  # 不包含当前step
                'token_prob_history': branch['token_probs'],
                'prob_history': branch['cum_probs']
            }
            
            all_branches_data = [
                {
                    'reward_history': b['rewards'],
                    'token_prob_history': b['token_probs'],
                    'prob_history': b['cum_probs']
                }
                for b in branches
            ]
            
            # 调用剪枝决策
            should_prune, reason = pruner.should_prune_branch(
                branch_data=branch_data,
                all_branches=all_branches_data,
                current_step=step,
                verbose=True
            )
            
            if not should_prune:
                kept_branches.append(branch)
                print(f"  决策: 保留")
            else:
                print(f"  决策: 剪枝 ({reason})")
        
        if len(kept_branches) == 0:
            print("\n所有分支都被剪枝,保留得分最高的")
            kept_branches = branches[:1]
        
        branches = kept_branches
        
        # === 关键点2: 对保留的分支生成token和计算reward ===
        print(f"\n生成step {step}的token和reward...")
        for branch in branches:
            # 模拟生成token
            new_token_probs = [0.9 + 0.01 * step] * 10
            branch['token_probs'].append(new_token_probs)
            
            # 模拟累积概率
            prev_cum_prob = branch['cum_probs'][-1] if branch['cum_probs'] else 1.0
            new_cum_prob = prev_cum_prob * 0.95
            branch['cum_probs'].append(new_cum_prob)
            
            # === 关键点3: 现在才计算reward并添加到历史 ===
            import random
            # 分支0模拟高reward,分支1模拟低reward
            if branch['id'] == 0:
                new_reward = 0.6 + 0.05 * step + random.uniform(-0.05, 0.05)
            else:
                new_reward = 0.2 + 0.02 * step + random.uniform(-0.05, 0.05)
            
            branch['rewards'].append(new_reward)
            print(f"  分支 {branch['id']}: new_reward={new_reward:.4f}")
        
        print(f"\n保留 {len(branches)} 个分支")
    
    print("\n"+"="*80)
    print("Beam search完成")
    print("="*80)
    
    stats = pruner.get_stats()
    print(f"\n剪枝统计:")
    print(f"  总决策: {stats['total_decisions']}")
    print(f"  剪枝: {stats['pruned_count']}")
    print(f"  保留: {stats['kept_count']}")
    print(f"  剪枝率: {stats['prune_rate']:.2%}")


def example_wrong_usage():
    """错误的使用方式 - 仅作示例,不要这样做!"""
    
    print("\n"+"="*80)
    print("❌ 错误示例: 不要这样做!")
    print("="*80)
    
    pruner = DynamicPruner("threshold_model.json")
    
    branch = {
        'rewards': [0.5, 0.6, 0.7]
    }
    
    # ❌ 错误1: 在计算reward之后才做剪枝决策
    print("\n❌ 错误1: 先计算reward,再决定剪枝")
    print("这是错误的顺序,因为如果要剪枝,就不应该浪费计算资源生成token")
    
    # 先计算了新的reward
    new_reward = 0.8
    branch['rewards'].append(new_reward)
    
    # 然后才做剪枝决策 - 太晚了!
    should_prune, _ = pruner.should_prune_branch(
        {'reward_history': branch['rewards'][:-1]},  # 不包含刚计算的
        [{'reward_history': branch['rewards'][:-1]}],
        current_step=3
    )
    print(f"  结果: {'剪枝' if should_prune else '保留'} - 但reward已经浪费计算了!")
    
    # ❌ 错误2: 包含当前step的reward
    print("\n❌ 错误2: 包含当前step的reward")
    branch2 = {'rewards': [0.5, 0.6, 0.7, 0.8]}  # 4步
    
    # 在step 3做决策时,包含了step 3的reward (0.8)
    # 但实际上step 3的reward还未计算!
    should_prune, _ = pruner.should_prune_branch(
        {'reward_history': branch2['rewards']},  # 错误地包含了当前step
        [{'reward_history': branch2['rewards']}],
        current_step=3
    )
    print(f"  这会导致使用'未来数据',在真实运行时无法获得")


def main():
    print("动态剪枝器使用示例\n")
    
    # 正确的使用方式
    example_correct_usage()
    
    # 错误的使用方式(仅作对比)
    example_wrong_usage()
    
    print("\n"+"="*80)
    print("总结:")
    print("="*80)
    print("✅ 1. 在生成token之前做剪枝决策")
    print("✅ 2. 只使用历史reward(不含当前step)")
    print("✅ 3. 对保留的分支才生成token和计算reward")
    print("✅ 4. 将新reward添加到历史中")
    print("="*80)


if __name__ == '__main__':
    main()
