import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

import logging



# from run.run_Attn_baseline import run_Attn_baseline
# from run.run_TF_block_baseline import run_TF_block_baseline
from utils.log_tools import setup_logger

from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *
from src.FFN_Layer import FFN_Layer
from run.run_FFN_baseline2 import run_FFN_baseline2


setup_logger(root_dir, "run_FFN_baseline2")
logger = logging.getLogger(__name__)



def main_for_FFN_baseline2():
    logger.info("FFN基线模型2运行开始")

    # 初始化FFN层
    my_ffn_layer = FFN_Layer(
        stage_id=13,
        hidden_dim=2048,
    )
    
    # # 初始化Attention层
    # my_attn_layer = Attn_Layer_Baseline(
    #     block_id=1,
    #     head_num=16,
    #     head_dim=128,
    #     hidden_dim=2048,
    # )


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
        # bandwidth=238.418, #带宽，单位为GB/s

        bandwidth=912/64,  #GB/s
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



    my_input_token_num = 4096
    # my_cached_token_num = 4096
    
    # 设置FFN层的N，并初始化矩阵
    my_ffn_layer.set_N(my_input_token_num)
    my_ffn_layer.init_matrices()

    
    # 运行FFN基线模型
    run_FFN_baseline2(
        my_ffn_layer=my_ffn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
    )
    


if __name__ == "__main__":
    main_for_FFN_baseline2()    

