import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import logging
logger = logging.getLogger(__name__)


from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *
from src.Attn_Layer_Baseline import Attn_Layer_Baseline



def run_Attn_baseline(
    my_attn_layer: Attn_Layer_Baseline,

    my_cim_config: CIM_config,
    my_cim_num: int,

    my_dram_config: DRAM_config,
    my_noc_config: NoC_config,
    my_nnlut_config: NNLUT_config,

    my_srambuffer_config: SRAM_Buffer_config,

):
    """
    运行Attn基线模型
    """
    q_gen_matmul, load_intoken_from_dram_for_q_gen, load_weight_q_from_dram, store_gened_part_q_to_dram  = my_attn_layer.q_gen_per_core()
    k_gen_matmul, load_intoken_from_dram_for_k_gen, load_weight_k_from_dram, store_gened_part_k_to_dram  = my_attn_layer.k_gen_per_core()
    v_gen_matmul, load_intoken_from_dram_for_v_gen, load_weight_v_from_dram, store_gened_part_v_to_dram  = my_attn_layer.v_gen_per_core()
    s_gen_matmul, load_gened_part_q_from_dram, load_concat_part_k_from_dram, store_psum_s_to_dram = my_attn_layer.psum_s_gen_per_core()
    
    s_allreduce_on_noc_per_core = my_attn_layer.s_allreduce_on_noc_per_core()
    
    sv_gen_matmul, load_s_from_dram, load_concat_part_v_from_dram, store_part_sv_to_dram  = my_attn_layer.sv_gen_per_core()
    o_gen_matmul, load_part_sv_from_dram, load_weight_o_from_dram, store_psum_o_to_dram  = my_attn_layer.psum_o_gen_per_core()

    o_allreduce_per_core = my_attn_layer.o_allreduce_per_core()



    q_gen_latency_computer = CIM_Latency_Computer(
        matmul_workload=q_gen_matmul,
        cim_config=my_cim_config,
        cim_num=my_cim_num,
    )

    k_gen_latency_computer = CIM_Latency_Computer(
        matmul_workload=k_gen_matmul,
        cim_config=my_cim_config,
        cim_num=my_cim_num,
    )

    v_gen_latency_computer = CIM_Latency_Computer(
        matmul_workload=v_gen_matmul,
        cim_config=my_cim_config,
        cim_num=my_cim_num,
    )

    s_gen_latency_computer = CIM_Latency_Computer(
        matmul_workload=s_gen_matmul,
        cim_config=my_cim_config,
        cim_num=my_cim_num,
    )
    
    nnlut_for_softmax_latency = NNLUT_Latency_Computer(
        input_matrix=my_attn_layer.generated_s_per_core,
        nnlut_config=my_nnlut_config,
    )

    sv_gen_latency_computer = CIM_Latency_Computer(
        matmul_workload=sv_gen_matmul,
        cim_config=my_cim_config,
        cim_num=my_cim_num,
    )

    o_gen_latency_computer = CIM_Latency_Computer(
        matmul_workload=o_gen_matmul,
        cim_config=my_cim_config,
        cim_num=my_cim_num,
    )

    layer_norm_latency_computer = NNLUT_Latency_Computer(
        input_matrix=my_attn_layer.generated_psum_o_per_core,
        nnlut_config=my_nnlut_config,
    )




    q_gen_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_intoken_from_dram_for_q_gen + load_weight_q_from_dram + store_gened_part_q_to_dram,
        dram_config=my_dram_config,
    )
    k_gen_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_intoken_from_dram_for_k_gen + load_weight_k_from_dram + store_gened_part_k_to_dram,
        dram_config=my_dram_config,
    )
    v_gen_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_intoken_from_dram_for_v_gen + load_weight_v_from_dram + store_gened_part_v_to_dram,
        dram_config=my_dram_config,
    )
    s_gen_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_gened_part_q_from_dram + load_concat_part_k_from_dram + store_psum_s_to_dram,
        dram_config=my_dram_config,
    )



    sv_gen_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_s_from_dram + load_concat_part_v_from_dram + store_part_sv_to_dram,
        dram_config=my_dram_config,
    )
    o_gen_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_part_sv_from_dram + load_weight_o_from_dram + store_psum_o_to_dram,
        dram_config=my_dram_config,
    )



    s_allreduce_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=s_allreduce_on_noc_per_core,
        noc_config=my_noc_config,
    )

    o_allreduce_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=o_allreduce_per_core,
        noc_config=my_noc_config,
    )


    non_linear_latency = nnlut_for_softmax_latency.softmax_latency()/4 + layer_norm_latency_computer.layernorm_latency()/64
    computing_latency = q_gen_latency_computer.compute_latency() + k_gen_latency_computer.compute_latency() + v_gen_latency_computer.compute_latency() + s_gen_latency_computer.compute_latency() + sv_gen_latency_computer.compute_latency() + o_gen_latency_computer.compute_latency()
    dram_latency = q_gen_dram_latency_computer.transfer_latency() + k_gen_dram_latency_computer.transfer_latency() + v_gen_dram_latency_computer.transfer_latency() + s_gen_dram_latency_computer.transfer_latency() + sv_gen_dram_latency_computer.transfer_latency() + o_gen_dram_latency_computer.transfer_latency()
    noc_latency = s_allreduce_noc_latency_computer.transfer_latency() + o_allreduce_noc_latency_computer.transfer_latency()

    total_latency = computing_latency + dram_latency + noc_latency + non_linear_latency

    logger.info(f"Attn基线模型计算延迟: {computing_latency}")
    logger.info(f"Attn基线模型DRAM延迟: {dram_latency}")
    logger.info(f"Attn基线模型NoC延迟: {noc_latency}")
    logger.info(f"Attn基线模型非线性延迟: {non_linear_latency}")
    
    logger.info(f"Attn基线模型总延迟: {total_latency}")

    non_linear_energy_comsumption = 64 * (nnlut_for_softmax_latency.softmax_energy_consumption()/4 + layer_norm_latency_computer.layernorm_energy_consumption()/64)
    computing_energy_consumption = 64*(q_gen_latency_computer.compute_energy_consumption() + k_gen_latency_computer.compute_energy_consumption() + v_gen_latency_computer.compute_energy_consumption() + s_gen_latency_computer.compute_energy_consumption() + sv_gen_latency_computer.compute_energy_consumption() + o_gen_latency_computer.compute_energy_consumption())
    dram_energy_consumption = 64 * (q_gen_dram_latency_computer.transfer_energy_consumption() + k_gen_dram_latency_computer.transfer_energy_consumption() + v_gen_dram_latency_computer.transfer_energy_consumption() + s_gen_dram_latency_computer.transfer_energy_consumption() + sv_gen_dram_latency_computer.transfer_energy_consumption() + o_gen_dram_latency_computer.transfer_energy_consumption())
    noc_energy_consumption = 64 * (s_allreduce_noc_latency_computer.transfer_energy_consumption() + o_allreduce_noc_latency_computer.transfer_energy_consumption())
    
    sram_energy_consumption = 64 * total_latency * my_srambuffer_config.average_power #单位：J

    total_energy_consumption = non_linear_energy_comsumption + computing_energy_consumption + dram_energy_consumption + noc_energy_consumption
    
    logger.info(f"Attn基线模型计算能耗: {computing_energy_consumption}")
    logger.info(f"Attn基线模型DRAM能耗: {dram_energy_consumption}")
    logger.info(f"Attn基线模型NoC能耗: {noc_energy_consumption}")
    logger.info(f"Attn基线模型非线性能耗: {non_linear_energy_comsumption}")
    logger.info(f"Attn基线模型SRAM能耗: {sram_energy_consumption}")
    logger.info(f"Attn基线模型总能耗: {total_energy_consumption}")

    return total_latency, total_energy_consumption

