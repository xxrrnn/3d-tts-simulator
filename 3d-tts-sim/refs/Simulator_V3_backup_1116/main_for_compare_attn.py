import os
import sys

from run import run_Attn_baseline2
from src import Attn_Layer_Baseline2, Attn_Layer_SeqParal

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

import logging




from utils.log_tools import setup_logger

from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *
from src.Attn_Layer_Baseline import Attn_Layer_Baseline
from src.Attn_Layer_Baseline2 import Attn_Layer_Baseline2
from src.Attn_Layer_SeqParal import Attn_Layer_SeqParal
from src.Attn_Layer_SeqParal_PredSparse import Attn_Layer_SeqParal_PredSparse

from run.run_Attn_baseline import run_Attn_baseline
from run.run_Attn_baseline2 import run_Attn_baseline2
from run.run_Attn_seqparal import run_Attn_seqparal
from run.run_Attn_seqparal_predsparse import run_Attn_seqparal_predsparse



setup_logger(root_dir, "compare_Attn")
logger = logging.getLogger(__name__)


input_token_num =  [1024, 1600, 2304, 4096]
cached_token_num = [1497, 2521, 4121, 6425]


def compare_attn(stage_idx, input_token_num, cached_token_num):
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


    baseline2D_attn_layer = Attn_Layer_Baseline2(
        stage_id=13,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )

    baseline_attn_layer = Attn_Layer_Baseline(
        stage_id=13,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )



    SP_attn_layer = Attn_Layer_SeqParal(
        stage_id=13,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )

    SPPS_attn_layer = Attn_Layer_SeqParal_PredSparse(
        stage_id=13,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )


    baseline2D_attn_layer.set_N(input_token_num)
    baseline2D_attn_layer.set_M(cached_token_num)
    baseline2D_attn_layer.set_stage_id(stage_idx)
    baseline2D_attn_layer.init_matrices()


    baseline_attn_layer.set_N(input_token_num)
    baseline_attn_layer.set_M(cached_token_num)
    baseline_attn_layer.set_stage_id(stage_idx)
    baseline_attn_layer.init_matrices()

   
    
    SP_attn_layer.set_N(input_token_num)
    SP_attn_layer.set_M(cached_token_num)
    SP_attn_layer.set_stage_id(stage_idx)
    SP_attn_layer.init_matrices()


    SPPS_attn_layer.set_N(input_token_num)
    SPPS_attn_layer.set_M(cached_token_num)
    SPPS_attn_layer.set_stage_id(stage_idx)
    SPPS_attn_layer.init_matrices()



    baseline2D_latency = run_Attn_baseline2(
        my_attn_layer=baseline2D_attn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        # my_srambuffer_config=my_sram_buffer_config,
    )

    baseline_latency, baseline_energy = run_Attn_baseline(
        my_attn_layer=baseline2D_attn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        my_srambuffer_config=my_sram_buffer_config,
    )
    

    SP_latency, SP_energy = run_Attn_seqparal(
        my_attn_layer=SP_attn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        my_sram_config=my_sram_buffer_config,
    )

    SPPS_latency, SPPS_energy = run_Attn_seqparal_predsparse(
        my_attn_layer=SPPS_attn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        my_topk_config=my_topk_config,
        my_srambuffer_config=my_sram_buffer_config,
    )

    latency = [baseline2D_latency, baseline_latency,  SP_latency, SPPS_latency]
    energy = [0 , baseline_energy,  SP_energy, SPPS_energy]

    return latency, energy

if __name__ == "__main__":
    with open ("/DISK1/data/zczhou_23/work2/Simulator_V3/results/compare_Attn.txt", "w") as f:
        f.write(f"比较Infinity2B中各种Attn层延迟和能耗\n")
        f.write(f"|   2D架构    |     3D+TP     |     3D+TP+RA     |     DuRA     |\n")

        for i in range(len(input_token_num)):
            f.write(f"第{i+10}层:\n")
            f.write(f"输入token数: {input_token_num[i]}\n")
            f.write(f"缓存token数: {cached_token_num[i]}\n")
            latency, energy = compare_attn(i+10, input_token_num[i], cached_token_num[i])
            f.write(f"延迟: {latency}\n")
            f.write(f"能耗: {energy}\n\n")
