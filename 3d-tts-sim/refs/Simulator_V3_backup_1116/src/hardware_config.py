import logging
from tkinter import Toplevel
logger = logging.getLogger(__name__)


class CIM_config:
    def __init__(self, 
        subarray_height: int, #存算比
        subarray_width: int,  #权重位宽
        macro_row_num: int,   #宏行数量
        macro_col_num: int,    #宏列数量

        working_frequency: float, #工作频率，单位为MHz
        energy_efficiency: float, #TOPS/W
        ):
        """
        TODO:
            读写功耗，
            计算功耗
        """
        
        self.subarray_height = subarray_height
        self.subarray_width = subarray_width
        self.macro_row_num = macro_row_num
        self.macro_col_num = macro_col_num

        self.working_frequency = working_frequency

        self.energy = (1/energy_efficiency) * 10**(-12) #J/op (注意这里的MAC应该算成2个OP)

        logger.info(f"CIM配置: 存算比={self.subarray_height}, 权重位宽={self.subarray_width}, 宏：{self.macro_row_num}行 x {self.macro_col_num}列")
        logger.info(f"工作频率={self.working_frequency}MHz, 功耗={self.energy}J/op")



    def storage_capacity(self):
        """返回CIM的存储容量，单位为KB"""
        return (self.subarray_height * self.subarray_width * self.macro_row_num * self.macro_col_num) / (8 * 1024)

    def compute_ability(self):
        """返回CIM的计算能力，归一化至INT8，单位为MAC/cycle"""
        return (self.macro_row_num * self.macro_col_num * self.subarray_width) / (8*8)



class DRAM_config:
    def __init__(self,
        bandwidth: float, #带宽，单位为GB/s

        energy: float, #pJ/bit
        ):
        """
        TODO:
            读写功耗
        """
        self.bandwidth = bandwidth * 1024*1024 #单位：KB/s
        self.energy = energy*8192* (10**(-12)) #单位：J/KB

        logger.info(f"DRAM配置: 带宽={self.bandwidth}KB/s, 功耗={self.energy}J/KB")

        # logger.info(f"DRAM配置: 带宽={self.bandwidth}KB/s")


class NoC_config:
    def __init__(self,
        flit_width: int, #包宽度，单位为bit
        working_frequency: float, #工作频率，单位为MHz

        energy: float, #pJ/bit/hop
        ):
        """
        TODO:
            传输功耗
        """
        self.flit_width = flit_width
        self.working_frequency = working_frequency

        self.bandwidth = self.flit_width * self.working_frequency*1000000 / (1024 * 8)
        #单位：KB/s

        #这里的hop我们默认是1，因为只用到了相邻core之间的互传
        self.energy = energy*8192* (10**(-12)) #单位：J/KB

        logger.info(f"NoC配置: 包宽度={self.flit_width}bit, 带宽={self.bandwidth}KB/s, 工作频率={self.working_frequency}MHz, 功耗={self.energy}J/KB")


class SRAM_Buffer_config:
    def __init__(self,
        capacity: float, #容量，单位为KB
        average_power: float, #3.3988528 mW
        ):
        """
        TODO:
            读写功耗
        """
        self.capacity = capacity
        self.average_power = average_power/1000 #单位：W

        logger.info(f"SRAM Buffer配置: 容量={self.capacity}KB, 平均功耗={self.average_power}W")


class NNLUT_config:
    def __init__(self,
        entry_num: int = 256,
        # working_frequency: float = 1470, #单位：MHz
        cycle_period: float = 2.5*10**(-6), #单位：ms
        power: float = 2.1584, #单位：mW
        ):
        self.entry_num = entry_num
        self.cycle_period = cycle_period #单位：ms
        self.power = power/1000 #单位：W

        logger.info(f"NNULT配置: entry_num={entry_num}, cycle_period={cycle_period}ms, power={power}W")


class TopK_config:
    def __init__(self,
        entry_num: int = 32,
        cycle_period: float = 2.5*10**(-6), #单位：ms
        power: float = 32*4.24727*100000 , #单位：uW
        ):
        self.entry_num = entry_num
        self.cycle_period = cycle_period #单位：ms
        self.power = power/1000000 #单位：W

        logger.info(f"topk配置: entry_num={entry_num}, cycle_period={cycle_period}ms, power={power}W")


