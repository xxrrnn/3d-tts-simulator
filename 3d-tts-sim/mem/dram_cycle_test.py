import json
from math import ceil
from tracemalloc import start
from dram_latency import SimpleDramLatencyModel

# 加载 workload JSON 文件
# workload_path = '../model_workloads/AMC23_beam_search/Qwen2.5-1.5B/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/60_8_1/question_0_workload.json'
workload_path = 'G:\\PKU\\Mine\\Simulator\\3d-tts-simulator\\3d-tts-sim\\model_workloads\\AMC23_beam_search\\Qwen2.5-1.5B\\Skywork-o1-Open-PRM-Qwen-2.5-1.5B\\60_8_1\\question_0_workload.json'
HEAD_DIM = 128
ROW_SIZE = 2048
ROW_COUNT = ROW_SIZE // HEAD_DIM

with open(workload_path, 'r', encoding='utf-8') as f:
    workload = json.load(f)

print(f"加载的 workload: {workload['question_id']}")
print(f"Prefill 阶段 KV cache 数量: {workload['prefill']['kv_cache_count']}")
print(f"Decode 阶段步数: {len(workload['decode']['steps'])}")

# 初始化 DRAM 模型
dram = SimpleDramLatencyModel.lpddr4_x16_lockstep_defaults(4)
start_addr = workload['prefill']['kv_cache_count']# * HEAD_DIM

### naive save - 每个branch按顺序交错存储
navie_latency = {}
for step_idx, step in enumerate(workload['decode']['steps']):
    # 读取selected_branch_index对应的地址
    selected_branch_idx = step['selected_branch_index']
    if selected_branch_idx == -1:
        break
    cur_addr = start_addr
    branch_tokens = step['branch_tokens']
    branch_count = step['branch_count']
    addr_dict = {i: [] for i in range(branch_count)}
    branch_tokens = step['branch_tokens'].copy()
    while any(branch_tokens):  # 只要还有非0的token就继续
        for branch_idx in range(branch_count):
            if branch_tokens[branch_idx] > 0:
                addr_dict[branch_idx].append(cur_addr)
                branch_tokens[branch_idx] -= 1
                cur_addr += HEAD_DIM
    
    selected_addrs = addr_dict[selected_branch_idx]
    if selected_addrs:
        print(f"\n=== Step {step_idx} - Selected Branch {selected_branch_idx} ===")
        print(f"Token数量: {len(selected_addrs)}")
        # 读取该branch的所有地址
        report = dram.read_addresses(selected_addrs, access_size=HEAD_DIM)
        navie_latency[step_idx] = report.total_cycles
    # 更新下一个step的起始地址
    start_addr = cur_addr


### chunk save: 每个branch汇总到一个row来保存
chunk_latency = {}
dram.reset()
total_cycles = 0
for step_idx, step in enumerate(workload['decode']['steps']):
    chunk_latency[step_idx] = 0
    selected_branch_idx = step['selected_branch_index']
    if selected_branch_idx == -1:
        break
    cur_addr = ceil(start_addr / ROW_SIZE) * ROW_SIZE
    branch_tokens = step['branch_tokens']
    branch_count = step['branch_count']
    addr_dict_row = {i: [] for i in range(branch_count)}

    branch_tokens = step['branch_tokens'].copy() 
    while all(token > ROW_COUNT for token in branch_tokens):
        for branch_idx in range(branch_count):
            if branch_tokens[branch_idx] > ROW_COUNT:
                addr_dict_row[branch_idx].append(cur_addr)
                branch_tokens[branch_idx] -= ROW_COUNT
                cur_addr += ROW_SIZE
    
    selected_addrs_row = addr_dict_row[selected_branch_idx]
    if selected_addrs_row:
        print(f"\n=== Step {step_idx} - Selected Branch {selected_branch_idx} (Chunk Save) ===")
        print(f"Token数量: {len(selected_addrs)}")
        
        for row_addr in selected_addrs_row:
            report_row = dram.read(row_addr, ROW_SIZE)
            chunk_latency[step_idx] += report_row.total_cycles
        # report = dram.read(selected_addrs[0], selected_addrs[1])
        # total_cycles += report.total_cycles
        print(f"总延迟: {total_cycles} ns")
    # 更新下一个step的起始地址
    start_addr = cur_addr
    
print(navie_latency)
print(chunk_latency)
print("total navie latency: ", sum(navie_latency.values()))
print("total chunk latency: ", sum(chunk_latency.values()))