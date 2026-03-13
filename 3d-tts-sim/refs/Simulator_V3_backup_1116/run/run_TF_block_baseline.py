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
from src.Attn_Layer_Baseline import *
from src.CrossAttn_Layer_Baseline import *

from run.run_FFN_baseline import run_FFN_baseline
from run.run_Attn_baseline import run_Attn_baseline
from run.run_CrossAttn_baseline import run_CrossAttn_baseline


def run_TF_block_baseline(
    my_crossattn_layer: CrossAttn_Layer_Baseline,
    my_selfattn_layer: Attn_Layer_Baseline,
    my_ffn_layer: FFN_Layer,
    
    my_cim_config: CIM_config,
    my_cim_num: int,

    my_dram_config: DRAM_config,
    my_noc_config: NoC_config,
    my_nnlut_config: NNLUT_config,
    my_sram_buffer_config: SRAM_Buffer_config,

    ):
    """
    运行TF块基线模型
    """

    crossattn_latency, crossattn_energy_consumption = run_CrossAttn_baseline(
        my_attn_layer=my_crossattn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        my_sram_buffer_config=my_sram_buffer_config,
    )

    attn_latency, attn_energy_consumption = run_Attn_baseline(
        my_attn_layer=my_selfattn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        my_srambuffer_config=my_sram_buffer_config,
    )

    ffn_latency, ffn_energy_consumption = run_FFN_baseline(
        my_ffn_layer=my_ffn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
        my_sram_config=my_sram_buffer_config,
    )

    total_latency = crossattn_latency + attn_latency + ffn_latency
    total_energy_consumption = crossattn_energy_consumption + attn_energy_consumption + ffn_energy_consumption

    return total_latency, total_energy_consumption
