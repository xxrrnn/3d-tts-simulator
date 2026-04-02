import json
import os
from pathlib import Path
from typing import List, Dict, Any

def get_all_stopper(data_dir: str) -> Dict[str, List[Dict]]:
    """
    遍历加载指定目录中的所有workload JSON文件，
    提取所有的branch_token_topk_logprobs序列
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        字典，key为question_id，value为该问题的所有branch_token_topk_logprobs数据
    """
    data_dir = Path(data_dir)
    all_stopper = set()
    
    # 获取所有workload JSON文件
    json_files = sorted(data_dir.glob('question_*_workload.json'))
    
    print(f"找到 {len(json_files)} 个文件")
    
    for json_file in json_files:
        print(f"加载 {json_file.name}...")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        question_id = data['question_id']
        question_data = []
        
        # 遍历所有steps
        for step_idx, step in enumerate(data['decode']['steps']):
            # 检查是否有branch_token_topk_logprobs字段
            if 'branch_token_topk_logprobs' not in step:
                continue
            branch_lens = step['branch_tokens']
            for idx, branch_len in enumerate(branch_lens):
                print(branch_len, len(step['branch_token_topk_logprobs'][idx]))
                assert branch_len == len(step['branch_token_topk_logprobs'][idx])
            for branch in step['branch_token_topk_logprobs']:
                stop_token = branch[-1]
                stopper = sorted(stop_token.items(), key=lambda x: x[1], reverse=True)[0][0]
            
            all_stopper.add(stopper)
                
        

    
    return all_stopper


# 使用示例
if __name__ == '__main__':
    data_dir = '/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload/model_workloads/16384_4_1'
    
    all_stopper = get_all_stopper(data_dir)
    
    print(all_stopper)
    