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

from src.Attn_Layer_Baseline2 import Attn_Layer_Baseline2
from src.CrossAttn_Layer_Baseline import CrossAttn_Layer_Baseline
from src.FFN_Layer import FFN_Layer
from run.run_TF_block_baseline2 import run_TF_block_baseline2


setup_logger(root_dir, "run_Infinity2B_baseline2")
logger = logging.getLogger(__name__)



Infinity2B_Model_config ={
    "D" : 2048,
    "head_num" : 16,
    "head_dim" : 128,
    "step_num" : 32,

    "layer_num" : 13,
    "token_map_size" : [1, 4, 16, 36, 64, 144, 256, 400, 576, 1024, 1600, 2304, 4096],
    "cached_kv_num"  : [0, 1,  5, 21, 57, 121, 265, 521, 921, 1497, 2521, 4121, 6425],   
}




def main_for_Infinity2B_baseline2():
    logger.info("Infinity2B模型-基线模型2-运行开始")

    stage_latency = []
    


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
        bandwidth=1180/64, #带宽，单位为GB/s
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
        stage_id=0,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )

    my_selfattn_layer = Attn_Layer_Baseline2(
        stage_id=0,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )

    my_ffn_layer = FFN_Layer(
        stage_id=0,
        hidden_dim=2048,
    )

    result_dir = "/DISK1/data/zczhou_23/work2/Simulator_V3/results/"
    result_file_dir = os.path.join(result_dir, "infinity2B_baseline2.txt")


    with open(result_file_dir, "w") as f:
        f.write(f"开始Infinity2B模型-基线模型2-运行\n")

    for i in range(Infinity2B_Model_config["layer_num"]):
        text_token_num = 48
        my_input_token_num = Infinity2B_Model_config["token_map_size"][i]
        my_cached_token_num = Infinity2B_Model_config["cached_kv_num"][i]
        
        # 设置Attn层的N，并初始化矩阵
        my_crossattn_layer.set_N(my_input_token_num, text_token_num)
        my_crossattn_layer.set_stage_id(i)

        my_crossattn_layer.init_matrices()

        my_selfattn_layer.set_N(my_input_token_num)
        my_selfattn_layer.set_M(my_cached_token_num)
        my_selfattn_layer.set_stage_id(i)
        my_selfattn_layer.init_matrices()

        my_ffn_layer.set_N(my_input_token_num)
        my_ffn_layer.set_stage_id(i)    
        my_ffn_layer.init_matrices()

        # 运行TF块基线模型
        stage_latency.append(
            run_TF_block_baseline2(  
            my_crossattn_layer=my_crossattn_layer,
            my_selfattn_layer=my_selfattn_layer,
            my_ffn_layer=my_ffn_layer,
            my_cim_config=my_cim_config,
            my_cim_num=my_cim_num,
            my_dram_config=my_dram_config,
            my_noc_config=my_noc_config,
            my_nnlut_config=my_nnlut_config,
        ))
        
        with open(result_file_dir, "a") as f:
            f.write(f"第{i}层延迟: {stage_latency[-1]}\n")
        
    logger.info(f"Infinity2B模型-基线模型2-总延迟: {sum(stage_latency)}")
    with open(result_file_dir, "a") as f:
        f.write(f"总延迟: {sum(stage_latency)}\n")
    


if __name__ == "__main__":
    main_for_Infinity2B_baseline2()    

