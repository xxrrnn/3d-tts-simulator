from .hardware_config import *
from .workload_config import MatMul_Workload, myMatrix

import logging
import math

logger = logging.getLogger(__name__)


class CIM_Latency_Computer:
    def __init__(self, 
                matmul_workload: MatMul_Workload,
                cim_config: CIM_config,
                cim_num:int
                ):

        self.matmul_workload = matmul_workload
        self.cim_config = cim_config
        self.cim_num = cim_num

        assert self.cim_config.subarray_width == self.matmul_workload.matrix_B.data_width, "CIM子阵列宽度与矩阵B数据宽度不匹配"


    def map_weight_to_cim(self):
        """
        将权重映射到CIM子阵列
        """
        page_x = math.ceil(self.matmul_workload.matrix_B.matrix_width / self.cim_config.macro_col_num)
        page_y = math.ceil(self.matmul_workload.matrix_B.matrix_height / self.cim_config.macro_row_num)
        logger.info(f"matrix_B.matrix_width={self.matmul_workload.matrix_B.matrix_width}, matrix_B.matrix_height={self.matmul_workload.matrix_B.matrix_height}, cim_config.macro_col_num={self.cim_config.macro_col_num}, cim_config.macro_row_num={self.cim_config.macro_row_num}")
        
        utilization_percentage = (self.matmul_workload.matrix_B.matrix_width * self.matmul_workload.matrix_B.matrix_height) / (page_x * page_y * self.cim_config.macro_col_num * self.cim_config.macro_row_num) * 100
        logger.info(f"page_x={page_x}, page_y={page_y}, 利用率={utilization_percentage:.2f}%")

        return page_x, page_y

    def weight_update_cycles(self):
        """
        计算权重更新的周期数
        """
        page_x, page_y = self.map_weight_to_cim()
        page_turn = math.ceil((page_x * page_y) / self.cim_num)

        
        accurate_weight_update_cycles = page_turn * self.cim_config.macro_row_num * self.cim_config.subarray_height
        
        logger.info(f"page_turn={page_turn}, 准确的权重更新周期数={accurate_weight_update_cycles}")

        return accurate_weight_update_cycles


    def compute_cycles(self):
        """
        计算矩阵乘法的周期数
        """
        page_x, page_y = self.map_weight_to_cim()
        page_turn = math.ceil((page_x * page_y) / self.cim_num)
        accurate_computing_cycles = page_turn * self.matmul_workload.matrix_A.matrix_height * self.matmul_workload.matrix_A.data_width
        
        logger.info(f"page_turn={page_turn}, 准确的计算周期数={accurate_computing_cycles}")

        return accurate_computing_cycles

    def weight_update_latency(self):
        """
        计算权重更新的周期数
        """
        latency = self.weight_update_cycles() / (1000 * self.cim_config.working_frequency)  #单位：毫秒
        logger.info(f"权重更新延迟={latency:.2f}ms")
        return latency

    def compute_latency(self):
        """
        计算矩阵乘法的延迟
        """
        latency = self.compute_cycles() / (1000 * self.cim_config.working_frequency)  #单位：毫秒
        logger.info(f"CIM计算延迟={latency:.2f}ms")
        return latency

    def compute_energy_consumption(self):
        """
        计算CIM计算能耗
        """
        computing_cycles = self.compute_cycles() #cycles
        cim_macs_per_cycle = self.cim_config.compute_ability() * self.cim_num #MAC/cycle
        energy_per_op = self.cim_config.energy #J/op

        energy_consumption = 2 * computing_cycles * cim_macs_per_cycle * energy_per_op *1000 #单位：mJ

        logger.info(f"CIM计算能耗={energy_consumption:.2f}mJ")
        return energy_consumption


class NoC_Latency_Computer:
    def __init__(self, 
                
                transfer_workload: float, #单位：KB

                noc_config: NoC_config,
                ):
        self.transfer_workload = transfer_workload #单位：KB
        self.noc_config = noc_config
        
    def transfer_latency(self):
        """
        计算NoC传输延迟
        """
        latency = self.transfer_workload * 1000 / self.noc_config.bandwidth  #单位：毫秒
        
        logger.info(f"NoC传输延迟={latency:.2f}ms")
        
        return latency

    def transfer_energy_consumption(self):
        """
        计算NoC传输能耗
        """
        energy_consumption = self.noc_config.energy * self.transfer_workload * 1000 #单位：mJ
        logger.info(f"NoC传输能耗={energy_consumption:.2f}mJ")
        return energy_consumption



class DRAM_Latency_Computer:
    def __init__(self, 
                transfer_workload: float, #单位：KB
                dram_config: DRAM_config,

                ):
        self.transfer_workload = transfer_workload #单位：KB
        self.dram_config = dram_config

    def transfer_latency(self):
        """
        计算DRAM传输延迟
        """

        latency = self.transfer_workload * 1000 / self.dram_config.bandwidth  #单位：毫秒
        
        logger.info(f"DRAM传输延迟={latency:.2f}ms")
        
        return latency

    def transfer_energy_consumption(self):
        """
        计算DRAM传输能耗
        """
        energy_consumption = self.dram_config.energy * self.transfer_workload * 1000 #单位：mJ
        logger.info(f"DRAM传输能耗={energy_consumption:.2f}mJ")
        return energy_consumption


class NNLUT_Latency_Computer:
    def __init__(self, 
                input_matrix: myMatrix,
                nnlut_config: NNLUT_config,
                ):

        self.input_matrix = input_matrix
        self.nnlut_config = nnlut_config

    def softmax_latency(self):
        """
        计算NNULT传输延迟
        """ 
        #(N, N+M)进行softmax
        #N（N+M）//16

        per_row_turns = math.ceil(self.input_matrix.matrix_width / self.nnlut_config.entry_num)
        row_num = self.input_matrix.matrix_height

        latency = row_num * (per_row_turns * 2 + 2)* self.nnlut_config.cycle_period #单位：毫秒



        # scale = max(1, (self.input_matrix.matrix_height * self.input_matrix.matrix_width) // self.nnlut_config.entry_num )
        # #(N+M)个数，进行exp+div,各4 cycle
        # #忽略 exp, 只计算div的cycle
        # latency = scale * 2 * self.nnlut_config.cycle_period #单位：毫秒
        
        logger.info(f"softmax in_matrix shape={self.input_matrix.matrix_height}*{self.input_matrix.matrix_width}")
        logger.info(f"per_row_turns={per_row_turns}, row_num={row_num}")
        logger.info(f"NNULT softmax延迟={latency:.2f}ms")
        
        return latency
        
    def softmax_energy_consumption(self):
        """
        计算NNULT softmax能耗
        """
        energy_consumption = self.softmax_latency() * self.nnlut_config.power #单位：mJ
        logger.info(f"NNULT softmax能耗={energy_consumption:.2f}mJ")
        return energy_consumption

    def gelu_latency(self):
        """
        计算NNULT传输延迟
        """

        scale = max(1, (self.input_matrix.matrix_height * self.input_matrix.matrix_width) // self.nnlut_config.entry_num )
        
        #每个数，gelu计算2 cycle
        latency = scale * 2 * self.nnlut_config.cycle_period #单位：毫秒
        
        logger.info(f"NNULT gelu延迟={latency:.2f}ms")
        
        return latency
        
    def gelu_energy_consumption(self):
        """
        计算NNULT gelu能耗
        """
        energy_consumption = self.gelu_latency() * self.nnlut_config.power #单位：mJ
        logger.info(f"NNULT gelu能耗={energy_consumption:.2f}mJ")
        return energy_consumption

    def layernorm_latency(self):
        """
        计算NNULT layernorm延迟
        """

        #(seq_len, d_model)
        
        scale = max(1, (self.input_matrix.matrix_height * self.input_matrix.matrix_width) // self.nnlut_config.entry_num )
        #每个数，layernorm, 核心部分是每个数一个除法，4 cycle
        latency = (scale * 2 + 2)* self.nnlut_config.cycle_period #单位：毫秒
        
        logger.info(f"NNULT layernorm延迟={latency:.2f}ms")
        
        return latency
        
    def layernorm_energy_consumption(self):
        """
        计算NNULT layernorm能耗
        """
        energy_consumption = self.layernorm_latency() * self.nnlut_config.power #单位：mJ
        logger.info(f"NNULT layernorm能耗={energy_consumption:.2f}mJ")
        return energy_consumption



class TopK_Latency_Computer:
    def __init__(self, 
                input_array_nums:int,
                input_array_length:int,
                topk_config: TopK_config,
                output_array_length:int,
                ):

        self.input_array_nums = input_array_nums
        self.input_array_length = input_array_length
        self.topk_config = topk_config
        self.output_array_length = output_array_length

    def topk_latency(self):
        """
        计算TopK延迟
        """
        per_column_turns = math.ceil(self.input_array_nums / self.topk_config.entry_num)
        
        latency =per_column_turns * (self.input_array_length + math.log2(self.output_array_length)) * self.topk_config.cycle_period #单位：毫秒
        
        logger.info(f"topk input_array_nums={self.input_array_nums}")
        logger.info(f"topk input_array_length={self.input_array_length}")
        logger.info(f"topk output_array_length={self.output_array_length}")
        logger.info(f"per_column_turns={per_column_turns}")
        logger.info(f"TopK延迟={latency:.2f}ms")
        
        return latency

    def topk_energy_consumption(self):
        """
        计算TopK能耗
        """
        energy_consumption = self.topk_latency() * self.topk_config.power #单位：mJ
        logger.info(f"TopK能耗={energy_consumption:.2f}mJ")
        return energy_consumption
