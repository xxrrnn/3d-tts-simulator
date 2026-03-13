"""
总配置：张量并行+序列并行

1. 16个core group（4 core）并行执行一个head
2. 每个core group内部，每个core负责1/4的head
3. 这里我们只关注单个core的负载（计算+传输）
"""
import logging

from src.workload_config import MatMul_Workload, myMatrix

logger = logging.getLogger(__name__)

#作为baseline
#仅考虑垂直切分，即张量并行，不考虑序列并行
#无预测，无稀疏
class CrossAttn_Layer_Baseline:
    def __init__(self,
        stage_id:int,

        head_num:int = 16,
        head_dim:int = 128,
        
        hidden_dim:int = 2048,
    ):
        assert head_num*head_dim == hidden_dim, "head_num*head_dim必须等于hidden_dim"

        self.stage_id = stage_id
        self.head_num = head_num
        self.head_dim = head_dim

        self.D = hidden_dim

    def set_N(self, input1_token_num:int, input2_token_num:int):
        self.N1 = input1_token_num
        self.N2 = input2_token_num

    def set_stage_id(self, stage_id:int):
        self.stage_id = stage_id

    def init_matrices(self):
        """
        初始化出现在每个core上的矩阵
        注意：需要在调用set_N和set_M之后调用此方法
        """
        self.input1_matrix_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N1,
                matrix_width  = self.D,
            )
        
        self.input2_matrix_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N2,
                matrix_width  = self.D,
            )
        

        self.weight_q_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.D,
                matrix_width  = self.head_dim//4,
            )

        self.weight_k_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.D,
                matrix_width  = self.head_dim//4,
            )

        self.weight_v_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.D,
                matrix_width  = self.head_dim//4,
            )


        #part表示本身是正确结果的一部分，需要拼接得到完整的结果
        self.generated_part_q_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N1,
                matrix_width  = self.head_dim//4,
            )

        self.generated_part_k_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N2,
                matrix_width  = self.head_dim//4,
            )
    
        self.generated_part_v_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N2,
                matrix_width  = self.head_dim//4,
            )




    #psum表示需要累加才能得到正确结果
        self.generated_psum_s_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N1,
                matrix_width  = self.N2,
            )
    
    #经过4core之间的all reduce，得到完整的s
        self.generated_s_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N1,
                matrix_width  = self.N2,
            )
    
        self.generated_part_sv_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N1,
                matrix_width  = self.head_dim//4,
            )

        self.weight_o_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.head_dim//4,
                matrix_width  = self.D,
            )
    
    #需要全局的64core之间的all reduce
        self.generated_psum_o_per_core = myMatrix(
                data_width    = 8,
                matrix_height = self.N1,
                matrix_width  = self.D,
            )

#####################################################################################
    """
    每个core承担的计算量
    """
    def q_gen_per_core(self):
        logger.info("start q_gen_per_core")
        matmul_workload = MatMul_Workload(
            matrix_A = self.input1_matrix_per_core,
            matrix_B = self.weight_q_per_core,
        )
        # logger.info("matmul_workload.matrix_C.matrix_height = %s", matmul_workload.matrix_C.matrix_height)
        # logger.info("self.generated_part_q_per_core.matrix_height = %s", self.generated_part_q_per_core.matrix_height)
        assert matmul_workload.matrix_C == self.generated_part_q_per_core, "q生成的矩阵C的维度必须等于generated_part_q_per_core"

        load_weight_q_from_dram = self.weight_q_per_core.total_KB()
        load_intoken1_from_dram = self.input1_matrix_per_core.total_KB()
        store_gened_part_q_to_dram = self.generated_part_q_per_core.total_KB()

        return matmul_workload, load_intoken1_from_dram, load_weight_q_from_dram, store_gened_part_q_to_dram

    def k_gen_per_core(self):
        logger.info("start k_gen_per_core")
        matmul_workload = MatMul_Workload(
            matrix_A = self.input2_matrix_per_core,
            matrix_B = self.weight_k_per_core,
        )
        assert matmul_workload.matrix_C == self.generated_part_k_per_core, "k生成的矩阵C的维度必须等于generated_part_k_per_core"
        
        load_weight_k_from_dram = self.weight_k_per_core.total_KB()
        load_intoken2_from_dram = self.input2_matrix_per_core.total_KB()
        store_gened_part_k_to_dram = self.generated_part_k_per_core.total_KB()

        return matmul_workload, load_intoken2_from_dram, load_weight_k_from_dram, store_gened_part_k_to_dram
    
    def v_gen_per_core(self):
        logger.info("start v_gen_per_core")
        matmul_workload = MatMul_Workload(
            matrix_A = self.input2_matrix_per_core,
            matrix_B = self.weight_v_per_core,
        )
        assert matmul_workload.matrix_C == self.generated_part_v_per_core, "v生成的矩阵C的维度必须等于generated_part_v_per_core"

        load_weight_v_from_dram = self.weight_v_per_core.total_KB()
        load_intoken2_from_dram = self.input2_matrix_per_core.total_KB()
        store_gened_part_v_to_dram = self.generated_part_v_per_core.total_KB()

        return matmul_workload, load_intoken2_from_dram, load_weight_v_from_dram, store_gened_part_v_to_dram
    
    def psum_s_gen_per_core(self):
        logger.info("start psum_s_gen_per_core")
        matmul_workload = MatMul_Workload(
            # matrix_A = self.generated_part_q_per_core,
            # matrix_B = self.concat_part_k_per_core.transposed(),
            matrix_A = self.generated_part_v_per_core,
            matrix_B = self.generated_part_q_per_core.transposed(), #存储Q。输入K
        )
        # logger.info("matmul_workload.matrix_C.matrix_height = %s", matmul_workload.matrix_C.matrix_height)
        # logger.info("matmul_workload.matrix_C.matrix_width = %s", matmul_workload.matrix_C.matrix_width)
        # logger.info("self.generated_psum_s_per_core().matrix_height = %s", self.generated_psum_s_per_core().matrix_height)
        # logger.info("self.generated_psum_s_per_core().matrix_width = %s", self.generated_psum_s_per_core().matrix_width)

        assert matmul_workload.matrix_C.transposed() == self.generated_psum_s_per_core, "s生成的矩阵C的维度必须等于generated_psum_s_per_core"

        load_gened_part_q_from_dram = self.generated_part_q_per_core.total_KB()
        load_gened_part_k_from_dram = self.generated_part_k_per_core.total_KB()
        store_psum_s_to_dram = self.generated_psum_s_per_core.total_KB()

        # return matmul_workload, load_gened_part_q_from_dram, load_cached_part_k_from_dram, tranload_generated_part_k_from_noc, store_psum_s_to_dram
        return matmul_workload, load_gened_part_q_from_dram, load_gened_part_k_from_dram, store_psum_s_to_dram

    #psum_s:(N, N+M) --> s:(N, N+M)
    #4 core all reduce
    def s_allreduce_on_noc_per_core(self):
        logger.info("start s_allreduce_on_noc_per_core")
        return 2 * (3/4) * self.generated_psum_s_per_core.total_KB()


    #noc传输之后
    def sv_gen_per_core(self):
        logger.info("start sv_gen_per_core")
        matmul_workload = MatMul_Workload(
            matrix_A = self.generated_s_per_core,
            matrix_B = self.generated_part_v_per_core,
        )
        assert matmul_workload.matrix_C == self.generated_part_sv_per_core, "sv生成的矩阵C的维度必须等于generated_part_sv_per_core"
        
        load_s_from_dram = self.generated_s_per_core.total_KB()
        load_gened_part_v_from_dram = self.generated_part_v_per_core.total_KB()
        store_part_sv_to_dram = self.generated_part_sv_per_core.total_KB()

        return matmul_workload, load_s_from_dram, load_gened_part_v_from_dram, store_part_sv_to_dram

    def psum_o_gen_per_core(self):
        logger.info("start psum_o_gen_per_core")
        matmul_workload = MatMul_Workload(
            matrix_A = self.generated_part_sv_per_core,
            matrix_B = self.weight_o_per_core,
        )
        assert matmul_workload.matrix_C == self.generated_psum_o_per_core, "o生成的矩阵C的维度必须等于generated_psum_o_per_core"

        load_part_sv_from_dram = self.generated_part_sv_per_core.total_KB()
        load_weight_o_from_dram = self.weight_o_per_core.total_KB()
        store_psum_o_to_dram = self.generated_psum_o_per_core.total_KB()

        return matmul_workload, load_part_sv_from_dram, load_weight_o_from_dram, store_psum_o_to_dram

#####################################################################################
    """
    每个core承担的传输量
    """

    def o_allreduce_per_core(self):
        logger.info("start o_allreduce_per_core")
        return 2*(63/64) * self.generated_psum_o_per_core.total_KB()
