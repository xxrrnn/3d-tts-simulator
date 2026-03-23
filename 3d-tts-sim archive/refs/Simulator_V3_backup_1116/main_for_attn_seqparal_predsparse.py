import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

import logging



from utils.log_tools import setup_logger

from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *
from src.Attn_Layer_SeqParal_PredSparse import Attn_Layer_SeqParal_PredSparse
from run.run_Attn_seqparal_predsparse import run_Attn_seqparal_predsparse



setup_logger(root_dir, "run_Attn_seqparal_predsparse") 
logger = logging.getLogger(__name__)



def main_for_Attn_seqparal_predsparse():
    logger.info("Attn序列并行模型（预测稀疏）运行开始")

    # 初始化Attention层
    my_attn_layer = Attn_Layer_SeqParal_PredSparse(
        stage_id=13,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )


    my_cim_config = CIM_config(
        subarray_height = 1, #存算比
        subarray_width  = 8,  #权重位宽
        macro_row_num   = 256,   #32*8宏行数量
        macro_col_num   = 32,    #16*2宏列数量
        
        working_frequency = 400, #工作频率，单位为MHz
        energy_efficiency = 24.7, #TOPS/W
    )


    # my_cim_num = 16*8
    my_cim_num = 8

    my_dram_config = DRAM_config(
        bandwidth=238.418, #带宽，单位为GB/s
        energy=0.88, #能量，单位为pJ/bit
    )
    my_noc_config = NoC_config(
        flit_width=512, #1024
        working_frequency=400, #工作频率，单位为MHz
        energy=6.70, #能量，单位为pJ/bit/hop
    )

    my_nnlut_config = NNLUT_config(
        entry_num= 16 * 16,
        # working_frequency: float = 1470, #单位：MHz
        cycle_period= 2.5*10**(-6), #单位：ms
        power= 2.1584, #单位：mW
    )

    my_topk_config = TopK_config(
        entry_num= 32,
        cycle_period= 2.5*10**(-6), #单位：ms
        power= 32*4.24727*100000, #单位：uW
    )

    my_sram_buffer_config = SRAM_Buffer_config(
        capacity = 256, #KB
        average_power= 3.3988528*32, #单位：mW
    )


    my_input_token_num = 4096
    my_cached_token_num = 6425
    
    # 设置Attn层的N，并初始化矩阵
    my_attn_layer.set_N(my_input_token_num)
    my_attn_layer.set_M(my_cached_token_num)
    my_attn_layer.init_matrices()

    
    # 运行Attn基线模型
    total_latency, total_energy_consumption = run_Attn_seqparal_predsparse(
        my_attn_layer=my_attn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        my_topk_config=my_topk_config,
        my_srambuffer_config=my_sram_buffer_config,
    )
    
    return total_latency, total_energy_consumption


if __name__ == "__main__":
    total_latency, total_energy_consumption = main_for_Attn_seqparal_predsparse()    
    with open ("/DISK1/data/zczhou_23/work2/Simulator_V3/results/energy/attn_seqparal_predsparse.txt", "w") as f:
        f.write(f"total_latency: {total_latency:.2f}ms\n")
        f.write(f"total_energy_consumption: {total_energy_consumption:.2f}mJ\n")

