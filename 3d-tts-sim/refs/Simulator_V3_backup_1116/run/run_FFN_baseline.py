import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import logging
logger = logging.getLogger(__name__)


from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *
from src.FFN_Layer import FFN_Layer



def run_FFN_baseline(
    my_ffn_layer: FFN_Layer,

    my_cim_config: CIM_config,
    my_cim_num: int,

    my_dram_config: DRAM_config,
    my_noc_config: NoC_config,
    my_nnlut_config: NNLUT_config,

    my_sram_config: SRAM_Buffer_config,

):
    """
    运行FFN基线模型
    """
    fc1_matmul,load_input_from_dram, load_weight1_from_dram, store_intermediate_to_dram = my_ffn_layer.fc1_per_core()

    fc2_matmul,load_intermediate_from_dram, load_weight2_from_dram, store_output_to_dram = my_ffn_layer.fc2_per_core()  

    o_allreduce_per_core = my_ffn_layer.o_allreduce_after_ffn_noc()

    gelu_latency_computer = NNLUT_Latency_Computer(
        input_matrix=my_ffn_layer.intermediate_matrix_per_core,
        nnlut_config=my_nnlut_config,
    )

    layer_norm_latency_computer = NNLUT_Latency_Computer(
        input_matrix=my_ffn_layer.output_matrix_per_core,
        nnlut_config=my_nnlut_config,
    )

    fc1_cim_latency_computer = CIM_Latency_Computer(
        matmul_workload=fc1_matmul,
        cim_config=my_cim_config,
        cim_num=my_cim_num,
    )

    fc2_cim_latency_computer = CIM_Latency_Computer(
        matmul_workload=fc2_matmul,
        cim_config=my_cim_config,
        cim_num=my_cim_num,
    )

    fc1_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_input_from_dram + load_weight1_from_dram + store_intermediate_to_dram,
        dram_config=my_dram_config,
    )

    fc2_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_intermediate_from_dram + load_weight2_from_dram + store_output_to_dram,
        dram_config=my_dram_config,
    )

    ffn_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=o_allreduce_per_core,
        noc_config=my_noc_config,
    )



    nonlinear_latency = layer_norm_latency_computer.layernorm_latency()/64 + gelu_latency_computer.gelu_latency()
    computing_latency = fc1_cim_latency_computer.compute_latency() + fc2_cim_latency_computer.compute_latency()
    dram_latency = fc1_dram_latency_computer.transfer_latency() + fc2_dram_latency_computer.transfer_latency()
    noc_latency = ffn_noc_latency_computer.transfer_latency()

    total_latency = computing_latency + dram_latency + noc_latency + nonlinear_latency

    logger.info(f"FFN基线模型计算延迟: {computing_latency}")
    logger.info(f"FFN基线模型DRAM延迟: {dram_latency}")
    logger.info(f"FFN基线模型NoC延迟: {noc_latency}")
    logger.info(f"FFN基线模型GELU+LayerNorm延迟: {nonlinear_latency}")
    
    logger.info(f"FFN基线模型总延迟: {total_latency}")


    nonlinear_energy_consumption = 64 * (layer_norm_latency_computer.layernorm_energy_consumption()/64 + gelu_latency_computer.gelu_energy_consumption())
    computing_energy_consumption = 64 * (fc1_cim_latency_computer.compute_energy_consumption() + fc2_cim_latency_computer.compute_energy_consumption())
    dram_energy_consumption = 64 * (fc1_dram_latency_computer.transfer_energy_consumption() + fc2_dram_latency_computer.transfer_energy_consumption())
    noc_energy_consumption = 64 * ffn_noc_latency_computer.transfer_energy_consumption()
    sram_energy_consumption = 64 * my_sram_config.average_power * total_latency

    total_energy_consumption = nonlinear_energy_consumption + computing_energy_consumption + dram_energy_consumption + noc_energy_consumption + sram_energy_consumption
    logger.info(f"FFN基线模型非线性能耗: {nonlinear_energy_consumption}")
    logger.info(f"FFN基线模型计算能耗: {computing_energy_consumption}")
    logger.info(f"FFN基线模型DRAM能耗: {dram_energy_consumption}")
    logger.info(f"FFN基线模型NoC能耗: {noc_energy_consumption}")
    logger.info(f"FFN基线模型SRAM能耗: {sram_energy_consumption}")
    logger.info(f"FFN基线模型总能耗: {total_energy_consumption}")


    return total_latency, total_energy_consumption