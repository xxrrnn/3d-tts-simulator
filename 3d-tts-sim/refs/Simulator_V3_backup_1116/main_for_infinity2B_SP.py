import os
import sys

from run import run_TF_block_baseline

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

import logging



# from run.run_TF_block_baseline import run_TF_block_baseline
from utils.log_tools import setup_logger

from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *

from src.Attn_Layer_SeqParal import Attn_Layer_SeqParal
from src.Attn_Layer_Baseline import Attn_Layer_Baseline
from src.CrossAttn_Layer_SeqParal import CrossAttn_Layer_SeqParal
from src.CrossAttn_Layer_Baseline import CrossAttn_Layer_Baseline

from src.FFN_Layer import FFN_Layer
from run.run_TF_block_SP import run_TF_block_SeqParal
from run.run_TF_block_baseline import run_TF_block_baseline


setup_logger(root_dir, "run_Infinity2B_SP")
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




def main_for_TF_block_SeqParal():
    logger.info("Infinity2B模型-序列并行-运行开始")

 
    


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

    my_sram_buffer_config = SRAM_Buffer_config(
        capacity = 256, #KB
        average_power= 3.3988528*32, #单位：mW
    )

    # 初始化Attention层
    my_crossattn_layer_sp = CrossAttn_Layer_SeqParal(
        stage_id=0,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )

    my_crossattn_layer_baseline = CrossAttn_Layer_Baseline(
        stage_id=0,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )


    my_selfattn_layer_sp = Attn_Layer_SeqParal(
        stage_id=0,
        head_num=16,
        head_dim=128,   
        hidden_dim=2048,
    )
    my_selfattn_layer_baseline = Attn_Layer_Baseline(
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
    result_file_dir = os.path.join(result_dir, "infinity2B_SP_1115.txt")
    
    with open(result_file_dir, "w") as f:   
        f.write(f"开始Infinity2B模型-序列并行-运行\n")

    stage_latency = []
    stage_energy_consumption = []

    for i in range(Infinity2B_Model_config["layer_num"]):
        text_token_num = 48
        my_input_token_num = Infinity2B_Model_config["token_map_size"][i]
        my_cached_token_num = Infinity2B_Model_config["cached_kv_num"][i]
        
        my_ffn_layer.set_N(my_input_token_num)
        my_ffn_layer.set_stage_id(i)    
        my_ffn_layer.init_matrices()

        if i <= 4:
            # 设置Attn层的N，并初始化矩阵
            my_crossattn_layer_baseline.set_N(my_input_token_num, text_token_num)
            my_crossattn_layer_baseline.set_stage_id(i)

            my_crossattn_layer_baseline.init_matrices()

            my_selfattn_layer_baseline.set_N(my_input_token_num)
            my_selfattn_layer_baseline.set_M(my_cached_token_num)
            my_selfattn_layer_baseline.set_stage_id(i)
            my_selfattn_layer_baseline.init_matrices()

            

            # 运行TF块基线模型
        
            
            latency, energy_consumption = run_TF_block_baseline(
                    my_crossattn_layer=my_crossattn_layer_baseline,
                    my_selfattn_layer=my_selfattn_layer_baseline,
                    my_ffn_layer=my_ffn_layer,
                    my_cim_config=my_cim_config,
                    my_cim_num=my_cim_num,
                    my_dram_config=my_dram_config,
                    my_noc_config=my_noc_config,
                    my_nnlut_config=my_nnlut_config,
                    my_sram_buffer_config=my_sram_buffer_config,
                )
            
        
        else:
            # 设置Attn层的N，并初始化矩阵
            my_crossattn_layer_sp.set_N(my_input_token_num, text_token_num)
            my_crossattn_layer_sp.set_stage_id(i)

            my_crossattn_layer_sp.init_matrices()

            my_selfattn_layer_sp.set_N(my_input_token_num)
            my_selfattn_layer_sp.set_M(my_cached_token_num)
            my_selfattn_layer_sp.set_stage_id(i)
            my_selfattn_layer_sp.init_matrices()

            latency, energy_consumption = run_TF_block_SeqParal(  
                my_crossattn_layer=my_crossattn_layer_sp,
                my_selfattn_layer=my_selfattn_layer_sp,
                my_ffn_layer=my_ffn_layer,
                my_cim_config=my_cim_config,
                my_cim_num=my_cim_num,
                my_dram_config=my_dram_config,
                my_noc_config=my_noc_config,
                my_nnlut_config=my_nnlut_config,
                my_sram_buffer_config=my_sram_buffer_config,    
        )
        
        stage_latency.append(latency)
        stage_energy_consumption.append(energy_consumption)
        
        with open(result_file_dir, "a") as f:
            f.write(f"第{i}层延迟: {stage_latency[-1]}; 能耗: {stage_energy_consumption[-1]}\n")
        
    logger.info(f"Infinity2B模型-序列并行-总延迟: {sum(stage_latency)}")
    logger.info(f"Infinity2B模型-序列并行-总能耗: {sum(stage_energy_consumption)}")
    
    with open(result_file_dir, "a") as f:
        f.write(f"总延迟: {sum(stage_latency)}\n")
        f.write(f"总能耗: {sum(stage_energy_consumption)}\n")
    


if __name__ == "__main__":
    main_for_TF_block_SeqParal()    

