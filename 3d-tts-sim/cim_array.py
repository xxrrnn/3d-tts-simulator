from typing import List
import numpy as np
import math, pickle


class CIM_Array:
    PR_SCALING = 1.5 # scaling factor to account for post placement and routing

    ## The class constructor
    # @param model-name:    Name of the model to be evaluated.
    # @param i_prec:        The input activation precision.
    # @param kv_prec:       The KV-cache precision.
    # @param w_prec:        The weight precision.
    # @param batch_size:    The batch size.
    # @param cxt_len:       The input context length.
    # @param is_generation: Whether the simulation mode is generation stage or prefill stage.
    
    # @param subarray_height: The height of the subarray.
    # @param subarray_width: The width of the subarray.
    # @param macro_row_num: The number of macro rows.
    # @param macro_col_num: The number of macro columns.
    # @param working_frequency: The working frequency of the CIM.
    # @param energy_efficiency: The energy efficiency of the CIM.
    def __init__(
        self,
        model_name: str,
        i_prec: int=16, 
        kv_prec: int=8,
        w_prec: int=8, 
        batch_size:int=1,
        pe_dp_size: int=1,
        cxt_len: int=256,
        is_generation: bool=False,
        
        subarray_height: int=1,
        subarray_width: int=8,
        macro_row_num: int=256,
        macro_col_num: int=32,
        working_frequency: float=400,
        energy_efficiency: float=24.7,
    ):
        assert pe_energy != 0, "ERROR! You must provide the energy cost of a PE."
        assert len(pe_array_dim) == 2, f"ERROR! The dimension of PE array must be 2. But you gave {len(pe_array_dim)}."
        
        self.model_name    = model_name
        self.i_prec        = i_prec
        self.kv_prec       = kv_prec
        self.w_prec        = w_prec
        self.batch_size    = batch_size

        self.subarray_height = subarray_height
        self.subarray_width = subarray_width
        self.macro_row_num = macro_row_num
        self.macro_col_num = macro_col_num
        self.working_frequency = working_frequency
        self.energy_efficiency = energy_efficiency
        
        self._init_model_profiler(model_name, cxt_len, is_generation)
    
    def _init_model_profiler(self, model_name, cxt_len: int=256, is_generation: bool=False):
        model_name_dict = {
            
        }
        file_path = f'./model_shape_config/{model_name_dict[model_name]}.pickle'
        with open(file_path, 'rb') as f:
            model_config, layer_config = pickle.load(f)
        
        ########## FFN Dimension ##########
        batch_size = self.batch_size
        weight_dim = {}
        input_dim  = {}
        output_dim = {}
        for name, weight_shape in layer_config.items():
            weight_dim[name] = [1] + weight_shape
            if is_generation: # generation
                input_dim[name]  = [1, batch_size, weight_shape[1]]
                output_dim[name] = [1, batch_size, weight_shape[0]]
            else:
                input_dim[name]  = [batch_size, cxt_len, weight_shape[1]]
                output_dim[name] = [batch_size, cxt_len, weight_shape[0]]

        ########## Attention Dimension ##########
        num_hidden_layers   = model_config['num_hidden_layers']
        hidden_size         = model_config['hidden_size']
        num_attention_heads = model_config['num_attention_heads']
        head_size           = hidden_size / num_attention_heads
        if 'num_key_value_heads' in model_config.keys():
            num_key_value_heads = model_config['num_key_value_heads']
        else:
            num_key_value_heads = num_attention_heads
        query_share_factor = num_attention_heads / num_key_value_heads

        for l_idx in range(num_hidden_layers):
            op_name = f'model.layers.{l_idx}.self_attn.attn_qk'
            if is_generation: # generation
                weight_dim[op_name] = [batch_size * num_key_value_heads, cxt_len, head_size] # key dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor, head_size] # query dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, cxt_len] # score dimension
            else:
                weight_dim[op_name] = [batch_size * num_key_value_heads, cxt_len, head_size] # key dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor * cxt_len, head_size] # query dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, cxt_len] # score dimension
            
            op_name = f'model.layers.{l_idx}.self_attn.attn_v'
            if is_generation: # generation
                weight_dim[op_name] = [batch_size * num_key_value_heads, head_size, cxt_len] # value dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor, cxt_len] # score dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, head_size] # output dimension
            else:
                weight_dim[op_name] = [batch_size * num_key_value_heads, head_size, cxt_len] # value dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor * cxt_len, cxt_len] # score dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, head_size] # output dimension

        self.weight_dim = weight_dim
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.layer_name_list = list(weight_dim.keys())

    def _init_mem(self):
        raise NotImplementedError('ERROR! No implementation of function _init_mem()')

