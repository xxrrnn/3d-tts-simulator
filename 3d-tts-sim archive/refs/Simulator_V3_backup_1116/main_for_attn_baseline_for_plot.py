import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

import logging


from utils.log_tools import setup_logger

from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *
from src.Attn_Layer_Baseline2 import Attn_Layer_Baseline2
from run.run_Attn_baseline2 import run_Attn_baseline2

from utils.plot_tools import plot_points

setup_logger(root_dir, "run_Attn_baseline2")
logger = logging.getLogger(__name__)



def main_for_Attn_baseline2():
    logger.info("Attn基线模型运行开始")

    # 初始化Attention层
    my_attn_layer = Attn_Layer_Baseline2(
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



    my_input_token_num = 4096
    my_cached_token_num = 6425
    
    # 设置Attn层的N，并初始化矩阵
    my_attn_layer.set_N(my_input_token_num)
    my_attn_layer.set_M(my_cached_token_num)
    my_attn_layer.init_matrices()


    bandwidth_list = []
    latency_list = []
    initial_bandwidth = 14.25 #GDDR6X 对齐到3090Ti
    util_ratio = 0.5
    for i in range(20):
        bandwidth_list.append(util_ratio * (initial_bandwidth + (i * 10)))
        
        my_dram_config = DRAM_config(
            # bandwidth=238.418, #带宽，单位为GB/s

            # bandwidth=1180/64,  #GB/s  HBM3E
            bandwidth= bandwidth_list[-1],  #GB/s
            energy=0.88, #能量，单位为pJ/bit
        )


        # 运行Attn基线模型
        latency_list.append(
            run_Attn_baseline2(
                my_attn_layer=my_attn_layer,
                my_cim_config=my_cim_config,
                my_cim_num=my_cim_num,
                my_dram_config=my_dram_config,
                my_noc_config=my_noc_config,
                my_nnlut_config=my_nnlut_config,
        ))

    return bandwidth_list, latency_list
    


if __name__ == "__main__":
    bandwidth_list, latency_list = main_for_Attn_baseline2()    
    plot_points(
        x_coords=bandwidth_list,
        y_coords=latency_list,
        title = "Attn Baseline Latency vs Bandwidth (Util Ratio = 0.5)",
        xlabel = "Bandwidth (GB/s)",
        ylabel = "Latency (ms)",
        color  = "#FF5733",  # 十六进制颜色
        marker = "^",       # 三角形标记
        markersize=10,
        save_path="/DISK1/data/zczhou_23/work2/Simulator_V3/pngs/attn_baseline_latency_vs_bandwidth_utilratio_0.5.png"
    )
