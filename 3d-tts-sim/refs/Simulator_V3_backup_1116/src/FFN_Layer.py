"""
总配置：张量并行+序列并行

1. 16个core group（4 core）并行执行一个head
2. 每个core group内部，每个core负责1/4的head
3. 这里我们只关注单个core的负载（计算+传输）
"""

from src.workload_config import MatMul_Workload, myMatrix
import logging
logger = logging.getLogger(__name__)



# O:(N, D);  W1:(D,4D) ==> O1:(N, 4D)
#O1:(N, 4D); W2:(4D,D) ==> O2:(N, D)
class FFN_Layer:
    def __init__(self,
        stage_id:int,

        hidden_dim:int = 2048,
    ):
        
        self.stage_id = stage_id

        self.D = hidden_dim
        

    def set_N(self, input_token_num:int):
        self.N = input_token_num

    def set_stage_id(self, stage_id:int):
        self.stage_id = stage_id

    def init_matrices(self):
        """
        初始化出现在每个core上的矩阵
        注意：需要在调用set_N之后调用此方法
        """
        #每个core上都是完整的O:(N, D)
        self.input_matrix_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N,
                matrix_width  = self.D,
            )

        #1/64的W1，垂直切分，(D, 4D//64)
        self.weight1_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.D,
                matrix_width  = self.D//16,#4D//64
            )
    
        #上采样结果O1:(N, 4D//64)
        self.intermediate_matrix_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N,
                matrix_width  = self.D//16,
            )
    
        #1/64的W2，水平切分，(4D//64, D)
        self.weight2_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.D//16,#4D//64  
                matrix_width  = self.D,
            )
        
        #下采样结果O2:(N, D)，这里的O2仅仅是partial sum，需要全局的all reduce
        self.output_matrix_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N,
                matrix_width  = self.D,
            )
    #下采样结果，经过全局的all reduce，变为最终的结果(N, D)
#####################################################################################
    """
    每个core上承担的矩阵乘法负载(包括计算负载、dram访存负载和noc传输负载)
    """
    def fc1_per_core(self):
        matmul_workload = MatMul_Workload(
            matrix_A = self.input_matrix_per_core,
            matrix_B = self.weight1_per_core,
        )
        assert matmul_workload.matrix_C == self.intermediate_matrix_per_core, "上采样矩阵乘法结果错误"

        load_weight1_from_dram = self.weight1_per_core.total_KB() #KB
        load_input_from_dram = self.input_matrix_per_core.total_KB() #KB
        store_intermediate_to_dram = self.intermediate_matrix_per_core.total_KB() #KB
        
        return matmul_workload,load_input_from_dram, load_weight1_from_dram, store_intermediate_to_dram


    def fc2_per_core(self):
        matmul_workload = MatMul_Workload(
            matrix_A = self.intermediate_matrix_per_core,
            matrix_B = self.weight2_per_core,
        )
        assert matmul_workload.matrix_C == self.output_matrix_per_core, "下采样矩阵乘法结果错误"
        
        load_weight2_from_dram = self.weight2_per_core.total_KB() #KB
        load_intermediate_from_dram = self.intermediate_matrix_per_core.total_KB() #KB
        store_output_to_dram = self.output_matrix_per_core.total_KB() #KB
        
        return matmul_workload,load_intermediate_from_dram, load_weight2_from_dram, store_output_to_dram


#####################################################################################
    """
    每个core上承担的传输负载
    """
    
    #在每层FFN结束后，需要all reduce
    def o_allreduce_after_ffn_noc(self):
        return (63/64)*2*self.output_matrix_per_core.total_KB()

    
        