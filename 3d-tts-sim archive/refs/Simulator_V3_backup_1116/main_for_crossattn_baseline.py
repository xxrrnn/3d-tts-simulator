import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

import logging

from utils.log_tools import setup_logger

from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *
from src.CrossAttn_Layer_Baseline import CrossAttn_Layer_Baseline
from run.run_CrossAttn_baseline import run_CrossAttn_baseline



setup_logger(root_dir, "run_CrossAttn_baseline")
logger = logging.getLogger(__name__)



def main_for_CrossAttn_baseline():
    logger.info("CrossAttn基线模型运行开始")

    # 初始化CrossAttn层
    my_crossattn_layer = CrossAttn_Layer_Baseline(
        stage_id=13,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )


    my_cim_config = CIM_config(
        subarray_height = 1, #存算比
        subarray_width  = 8,  #权重位宽
        macro_row_num   = 256,   #宏行数量
        macro_col_num   = 32,    #宏列数量
        
        working_frequency = 400, #工作频率，单位为MHz
        energy_efficiency = 24.7, #TOPS/W
    )


    my_cim_num = 8

    my_dram_config = DRAM_config(
        bandwidth=238.418, #带宽，单位为GB/s
        energy=0.88, #能量，单位为pJ/bit
    )
    my_noc_config = NoC_config(
        flit_width=512,
        working_frequency=400, #工作频率，单位为MHz
        energy=6.70, #能量，单位为pJ/bit/hop
    )

    my_nnlut_config = NNLUT_config(
        entry_num= 16 * 16,
        # working_frequency: float = 1470, #单位：MHz
        cycle_period= 2.5*10**(-6), #单位：ms
        power= 2.1584, #单位：mW
    )



    my_input1_token_num = 4096
    my_input2_token_num = 48
    
    # 设置Attn层的N，并初始化矩阵
    my_crossattn_layer.set_N(my_input1_token_num, my_input2_token_num)
    my_crossattn_layer.init_matrices()

    
    # 运行CrossAttn基线模型
    run_CrossAttn_baseline(
        my_attn_layer=my_crossattn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
    )
    


if __name__ == "__main__":
    main_for_CrossAttn_baseline()    

