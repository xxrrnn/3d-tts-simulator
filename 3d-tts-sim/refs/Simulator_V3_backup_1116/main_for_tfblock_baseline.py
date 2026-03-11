import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

import logging



# from run.run_TF_block_baseline import run_TF_block_baseline
from utils.log_tools import setup_logger

from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *

from src.Attn_Layer_Baseline import Attn_Layer_Baseline
from src.CrossAttn_Layer_Baseline import CrossAttn_Layer_Baseline
from src.FFN_Layer import FFN_Layer
from run.run_TF_block_baseline import run_TF_block_baseline


setup_logger(root_dir, "run_TF_block_baseline")
logger = logging.getLogger(__name__)



def main_for_TF_block_baseline():
    logger.info("TF块基线模型运行开始")

    


    my_cim_config = CIM_config(
        subarray_height = 1, #存算比
        subarray_width  = 8,  #权重位宽
        macro_row_num   = 32,   #宏行数量
        macro_col_num   = 16,    #宏列数量
        
        working_frequency = 400, #工作频率，单位为MHz
        energy_efficiency = 24.7, #TOPS/W
    )


    my_cim_num = 16*8

    my_dram_config = DRAM_config(
        bandwidth=238.418, #带宽，单位为GB/s
        energy=0.88, #能量，单位为pJ/bit
    )
    my_noc_config = NoC_config(
        flit_width=1024,
        working_frequency=400, #工作频率，单位为MHz
        energy=6.70, #能量，单位为pJ/bit/hop
    )

    my_nnlut_config = NNLUT_config(
        entry_num= 16 * 16,
        # working_frequency: float = 1470, #单位：MHz
        cycle_period= 2.5*10**(-6), #单位：ms
        power= 2.1584, #单位：mW
    )

    # 初始化Attention层
    my_crossattn_layer = CrossAttn_Layer_Baseline(
        stage_id=13,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )

    my_selfattn_layer = Attn_Layer_Baseline(
        stage_id=13,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )

    my_ffn_layer = FFN_Layer(
        stage_id=13,
        hidden_dim=2048,
    )

    text_token_num = 48
    my_input_token_num = 4096
    my_cached_token_num = 6425
    
    # 设置Attn层的N，并初始化矩阵
    my_crossattn_layer.set_N(my_input_token_num, text_token_num)

    my_crossattn_layer.init_matrices()

    my_selfattn_layer.set_N(my_input_token_num)
    my_selfattn_layer.set_M(my_cached_token_num)
    my_selfattn_layer.init_matrices()

    my_ffn_layer.set_N(my_input_token_num)
    my_ffn_layer.init_matrices()
    
    # 运行TF块基线模型
    total_latency = run_TF_block_baseline(  
        my_crossattn_layer=my_crossattn_layer,
        my_selfattn_layer=my_selfattn_layer,
        my_ffn_layer=my_ffn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
    )

    logger.info(f"TF块基线模型总延迟: {total_latency}")
    


if __name__ == "__main__":
    main_for_TF_block_baseline()    

