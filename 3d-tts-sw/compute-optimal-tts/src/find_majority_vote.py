#!/usr/bin/env python3
import json
import os
import glob

def find_majority_vote_questions(folder_path):
    """
    在指定文件夹中找出所有majority_vote = 1的题目索引
    """
    json_files = glob.glob(os.path.join(folder_path, "question_*_workload.json"))
    majority_vote_questions = []
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # 检查majority_vote是否为1
            if data.get("result", {}).get("majority_vote") == 1:
                # 从文件名提取题目编号
                filename = os.path.basename(json_file)
                question_num = filename.replace("question_", "").replace("_workload.json", "")
                majority_vote_questions.append(int(question_num))
                
        except Exception as e:
            print(f"读取文件 {json_file} 时出错: {e}")
    
    return sorted(majority_vote_questions)

def main():
    base_path = "/DISK1/data/rnxu_24/Paper/xlong_32/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/src/model_workloads/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/math-shepherd-mistral-7b-prm"
    
    folders = [
        "40_4_1_straggler_0_0_0_0_0_def0",
        "40_4_1_straggler_1_1.5_80_0_0_def0", 
        "40_4_1_straggler_1_1.5_80_0_0_def1",
        "40_4_1_straggler_1_1.5_100_0_0_def0",
        "40_8_1_straggler_0_0_0_0_0_def0",
        "40_8_1_straggler_1_1.5_80_0_0_def0",
        "40_8_1_straggler_1_1.5_80_0_0_def1", 
        "40_8_1_straggler_1_1.5_100_0_0_def0",
        "40_8_1_straggler_1_1.5_100_0_0_def1"
    ]
    
    results = {}
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            majority_vote_questions = find_majority_vote_questions(folder_path)
            results[folder] = majority_vote_questions
            print(f"\n{folder}:")
            print(f"  majority_vote = 1 的题目索引: {majority_vote_questions}")
            print(f"  总数: {len(majority_vote_questions)}")
        else:
            print(f"\n警告: 文件夹 {folder_path} 不存在")
    
    # 打印汇总
    print("\n" + "="*80)
    print("汇总结果:")
    print("="*80)
    for folder, questions in results.items():
        print(f"{folder}: {questions} (共{len(questions)}题)")

if __name__ == "__main__":
    main()