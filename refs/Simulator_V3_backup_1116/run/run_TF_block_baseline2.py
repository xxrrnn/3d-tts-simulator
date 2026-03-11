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
from src.Attn_Layer_Baseline2 import *
from src.CrossAttn_Layer_Baseline import *

from run.run_FFN_baseline2 import run_FFN_baseline2
from run.run_Attn_baseline2 import run_Attn_baseline2
from run.run_CrossAttn_baseline2 import run_CrossAttn_baseline2


def run_TF_block_baseline2(
    my_crossattn_layer: CrossAttn_Layer_Baseline,
    my_selfattn_layer: Attn_Layer_Baseline2,
    my_ffn_layer: FFN_Layer,
    
    my_cim_config: CIM_config,
    my_cim_num: int,

    my_dram_config: DRAM_config,
    my_noc_config: NoC_config,
    my_nnlut_config: NNLUT_config,

    ):
    """
    运行TF块基线模型2
    """

    crossattn_latency = run_CrossAttn_baseline2(
        my_attn_layer=my_crossattn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
    )

    attn_latency = run_Attn_baseline2(
        my_attn_layer=my_selfattn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
    )

    ffn_latency = run_FFN_baseline2(
        my_ffn_layer=my_ffn_layer,
        my_cim_config=my_cim_config,
        my_cim_num=my_cim_num,
        my_dram_config=my_dram_config,
        my_noc_config=my_noc_config,
        my_nnlut_config=my_nnlut_config,
    )

    total_latency = crossattn_latency + attn_latency + ffn_latency

    return total_latency
