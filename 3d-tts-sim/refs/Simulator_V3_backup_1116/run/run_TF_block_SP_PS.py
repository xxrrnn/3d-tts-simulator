import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import logging
logger = logging.getLogger(__name__)

from src.Latency_Computer import *
from src.workload_config import *
from src.hardware_config import *
from src.FFN_Layer import *
from src.Attn_Layer_SeqParal_PredSparse import *
from src.CrossAttn_Layer_SeqParal import *

from run.run_FFN_baseline import run_FFN_baseline
from run.run_Attn_seqparal_predsparse import run_Attn_seqparal_predsparse
from run.run_CrossAttn_seqparal import run_CrossAttn_seqparal

def run_TF_block_SeqParal_PredSparse(
    my_crossattn_layer: CrossAttn_Layer_SeqParal,
    my_selfattn_layer: Attn_Layer_SeqParal_PredSparse,
    my_ffn_layer: FFN_Layer,
    
    my_cim_config: CIM_config,
    my_cim_num: int,

    my_dram_config: DRAM_config,
    my_noc_config: NoC_config,
    my_nnlut_config: NNLUT_config,
    my_topk_config: TopK_config,
    my_sram_buffer_config: SRAM_Buffer_config,
    ):
    """
    运行TF块序列并行模型
    """

    crossattn_latency, crossattn_energy = run_CrossAttn_seqparal(
        my_attn_layer=my_crossattn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        my_sram_buffer_config=my_sram_buffer_config,
    )

    attn_latency, attn_energy = run_Attn_seqparal_predsparse(
        my_attn_layer=my_selfattn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,

        my_topk_config= my_topk_config,
        my_srambuffer_config=my_sram_buffer_config,
    )

    ffn_latency, ffn_energy = run_FFN_baseline(
        my_ffn_layer=my_ffn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        my_sram_config=my_sram_buffer_config,
    )

    total_latency = crossattn_latency + attn_latency + ffn_latency
    total_energy = crossattn_energy + attn_energy + ffn_energy

    return total_latency, total_energy
