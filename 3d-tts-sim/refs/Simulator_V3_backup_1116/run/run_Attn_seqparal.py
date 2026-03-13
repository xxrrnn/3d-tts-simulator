import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import logging
logger = logging.getLogger(__name__)


from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *
from src.Attn_Layer_SeqParal import Attn_Layer_SeqParal



def run_Attn_seqparal(
    my_attn_layer: Attn_Layer_SeqParal,

    my_cim_config: CIM_config,
    my_cim_num: int,

    my_dram_config: DRAM_config,
    my_noc_config: NoC_config,
    my_nnlut_config: NNLUT_config,

    my_sram_config: SRAM_Buffer_config,

):
    """
    运行Attn序列并行模型
    """
    q_gen_matmul, load_intoken_from_dram_for_q_gen, load_weight_q_from_dram, store_gened_part_q_to_dram  = my_attn_layer.q_gen_per_core()
    k_gen_matmul, load_intoken_from_dram_for_k_gen, load_weight_k_from_dram, store_gened_part_k_to_dram  = my_attn_layer.k_gen_per_core()
    v_gen_matmul, load_intoken_from_dram_for_v_gen, load_weight_v_from_dram, store_gened_part_v_to_dram  = my_attn_layer.v_gen_per_core()

    q_all2all_noc_per_core = my_attn_layer.q_all2all_transfer()
    k_all2all_noc_per_core = my_attn_layer.k_all2all_transfer()
    v_all2all_noc_per_core = my_attn_layer.v_all2all_transfer()

    s_gen_matmul, load_gened_chunk_q_from_dram, load_cached_chunk_k_from_dram, transfer_concat_chunk_k_from_noc, store_chunk_s_to_dram = my_attn_layer.chunk_s_gen_per_core()
    
    sv_gen_matmul, load_chunk_s_from_dram, load_cached_chunk_v_from_dram, transfer_concat_chunk_v_from_noc, store_chunk_sv_to_dram  = my_attn_layer.sv_gen_per_core()
    
    sv_all2all_noc_per_core =  my_attn_layer.sv_all2all_transfer()
    
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
        input_matrix=my_attn_layer.generated_chunk_s_per_core,
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
        transfer_workload=load_gened_chunk_q_from_dram + load_cached_chunk_k_from_dram + store_chunk_s_to_dram,
        dram_config=my_dram_config,
    )
    s_gen_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=transfer_concat_chunk_k_from_noc,
        noc_config=my_noc_config,
    )

    sv_gen_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_chunk_s_from_dram + load_cached_chunk_v_from_dram + store_chunk_sv_to_dram,
        dram_config=my_dram_config,
    )

    sv_gen_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=transfer_concat_chunk_v_from_noc,
        noc_config=my_noc_config,
    )

    o_gen_dram_latency_computer = DRAM_Latency_Computer(
        transfer_workload=load_part_sv_from_dram + load_weight_o_from_dram + store_psum_o_to_dram,
        dram_config=my_dram_config,
    )


    q_all2all_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=q_all2all_noc_per_core,
        noc_config=my_noc_config,
    )

    k_all2all_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=k_all2all_noc_per_core,
        noc_config=my_noc_config,
    )

    v_all2all_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=v_all2all_noc_per_core,
        noc_config=my_noc_config,
    )

    sv_all2all_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=sv_all2all_noc_per_core,
        noc_config=my_noc_config,
    )

    o_allreduce_noc_latency_computer = NoC_Latency_Computer(
        transfer_workload=o_allreduce_per_core,
        noc_config=my_noc_config,
    )


    non_linear_latency = nnlut_for_softmax_latency.softmax_latency() + layer_norm_latency_computer.layernorm_latency()/64
    computing_latency = q_gen_latency_computer.compute_latency() + k_gen_latency_computer.compute_latency() + v_gen_latency_computer.compute_latency() + s_gen_latency_computer.compute_latency() + sv_gen_latency_computer.compute_latency() + o_gen_latency_computer.compute_latency()
    dram_latency = q_gen_dram_latency_computer.transfer_latency() + k_gen_dram_latency_computer.transfer_latency() + v_gen_dram_latency_computer.transfer_latency() + s_gen_dram_latency_computer.transfer_latency() + sv_gen_dram_latency_computer.transfer_latency() + o_gen_dram_latency_computer.transfer_latency()
    noc_latency = q_all2all_noc_latency_computer.transfer_latency() + k_all2all_noc_latency_computer.transfer_latency() + v_all2all_noc_latency_computer.transfer_latency() + s_gen_noc_latency_computer.transfer_latency() + sv_gen_noc_latency_computer.transfer_latency() + sv_all2all_noc_latency_computer.transfer_latency() + o_allreduce_noc_latency_computer.transfer_latency()

    total_latency = computing_latency + dram_latency + noc_latency + non_linear_latency

    logger.info(f"Attn序列并行模型非线性延迟: {non_linear_latency}")

    logger.info(f"Attn序列并行模型计算延迟: {computing_latency}")
    logger.info(f"Attn序列并行模型DRAM延迟: {dram_latency}")
    logger.info(f"Attn序列并行模型NoC延迟: {noc_latency}")
    
    logger.info(f"Attn序列并行模型总延迟: {total_latency}")




    non_linear_energy_consumption = 64 * (nnlut_for_softmax_latency.softmax_energy_consumption()+layer_norm_latency_computer.layernorm_energy_consumption()/64)
    computing_energy_consumption = 64 * (q_gen_latency_computer.compute_energy_consumption() + k_gen_latency_computer.compute_energy_consumption() + v_gen_latency_computer.compute_energy_consumption() + s_gen_latency_computer.compute_energy_consumption() + sv_gen_latency_computer.compute_energy_consumption() + o_gen_latency_computer.compute_energy_consumption())
    dram_energy_consumption = 64 * (q_gen_dram_latency_computer.transfer_energy_consumption() + k_gen_dram_latency_computer.transfer_energy_consumption() + v_gen_dram_latency_computer.transfer_energy_consumption() + s_gen_dram_latency_computer.transfer_energy_consumption() + sv_gen_dram_latency_computer.transfer_energy_consumption() + o_gen_dram_latency_computer.transfer_energy_consumption())
    noc_energy_consumption = 64 * (q_all2all_noc_latency_computer.transfer_energy_consumption() + k_all2all_noc_latency_computer.transfer_energy_consumption() + v_all2all_noc_latency_computer.transfer_energy_consumption() + s_gen_noc_latency_computer.transfer_energy_consumption() + sv_gen_noc_latency_computer.transfer_energy_consumption() + sv_all2all_noc_latency_computer.transfer_energy_consumption() + o_allreduce_noc_latency_computer.transfer_energy_consumption())
    sram_energy_consumption = 64 * my_sram_config.average_power * total_latency

    total_energy_consumption = non_linear_energy_consumption + computing_energy_consumption + dram_energy_consumption + noc_energy_consumption + sram_energy_consumption

    logger.info(f"Attn序列并行模型非线性能耗: {non_linear_energy_consumption}")
    logger.info(f"Attn序列并行模型计算能耗: {computing_energy_consumption}")
    logger.info(f"Attn序列并行模型DRAM能耗: {dram_energy_consumption}")
    logger.info(f"Attn序列并行模型NoC能耗: {noc_energy_consumption}")
    logger.info(f"Attn序列并行模型SRAM能耗: {sram_energy_consumption}")

    logger.info(f"Attn序列并行模型总能耗: {total_energy_consumption}")
    


    return total_latency, total_energy_consumption