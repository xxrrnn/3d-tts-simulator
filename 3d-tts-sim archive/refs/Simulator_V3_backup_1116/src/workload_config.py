import math
import logging

logger = logging.getLogger(__name__)

class myMatrix:
    def __init__(self,
            data_width: int, #数据宽度，单位为bit
            matrix_height: int, #矩阵高度
            matrix_width: int, #矩阵宽度
        ):
        self.data_width = data_width
        self.matrix_height = matrix_height
        self.matrix_width = matrix_width

    def __eq__(self, other):
        if not isinstance(other, myMatrix):
            return False
        return (
            # self.data_width == other.data_width and
            self.matrix_height == other.matrix_height and
            self.matrix_width == other.matrix_width
        )

    def total_KB(self):
        return (self.matrix_height * self.matrix_width * self.data_width) / (8 * 1024)

    def transposed(self):
        return myMatrix(data_width=self.data_width, matrix_height=self.matrix_width, matrix_width=self.matrix_height)

    


# 矩阵乘法任务配置
# A*B=C
class MatMul_Workload:
    def __init__(self,
        matrix_A: myMatrix,
        matrix_B: myMatrix,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        # logger.info("self.matrix_A.matrix_height = %s", self.matrix_A.matrix_height)
        # logger.info("self.matrix_A.matrix_width = %s", self.matrix_A.matrix_width)
        
        # logger.info("self.matrix_B.matrix_height = %s", self.matrix_B.matrix_height)
        # logger.info("self.matrix_B.matrix_width = %s", self.matrix_B.matrix_width)

        assert self.matrix_A.matrix_width == self.matrix_B.matrix_height, "矩阵A的宽度必须等于矩阵B的高度"

        self.matrix_C = myMatrix(
            data_width=(self.matrix_A.data_width + self.matrix_B.data_width + math.log2(self.matrix_B.matrix_height)),
            matrix_height=self.matrix_A.matrix_height,
            matrix_width=self.matrix_B.matrix_width,
        )

    def total_INT8_MACs(self):
        # 归一到INT8
        total_macs = self.matrix_A.matrix_height * self.matrix_A.matrix_width * self.matrix_B.matrix_width
        scale = (self.matrix_A.data_width/8)*(self.matrix_B.data_width/8)
        return total_macs * scale
    
