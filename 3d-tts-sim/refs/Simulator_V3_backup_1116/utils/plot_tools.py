import matplotlib.pyplot as plt
from typing import List, Optional, Union
import numpy as np

def plot_points(
    x_coords: Union[List[float], np.ndarray],
    y_coords: Union[List[float], np.ndarray],
    title: str = "Points Visualization",
    xlabel: str = "X Axis",
    ylabel: str = "Y Axis",
    color: str = "blue",
    marker: str = "o",
    markersize: int = 6,
    grid: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    绘制由两个坐标向量（x坐标 + y坐标）定义的点集

    参数说明：
    ----------
    x_coords: Union[List[float], np.ndarray]
        x坐标向量，可传入列表（如[1,3,5]）或numpy数组，元素为数字
    y_coords: Union[List[float], np.ndarray]
        y坐标向量，与x_coords长度必须一致，格式同上
    title: str (默认: "Points Visualization")
        图表标题
    xlabel: str (默认: "X Axis")
        X轴标签
    ylabel: str (默认: "Y Axis")
        Y轴标签
    color: str (默认: "blue")
        点的颜色（支持matplotlib颜色格式：命名色、十六进制、RGB）
    marker: str (默认: "o")
        点的标记样式（"o"=圆，"s"=正方形，"^"=三角形，"*"=星号等）
    markersize: int (默认: 6)
        点的大小
    grid: bool (默认: True)
        是否显示网格线
    save_path: Optional[str] (默认: None)
        图片保存路径（如"points.png"），为None时仅显示不保存
    """
    # --------------------------
    # 关键输入校验（避免错误）
    # --------------------------
    # 1. 校验输入类型（支持列表或numpy数组）
    if not isinstance(x_coords, (list, np.ndarray)) or not isinstance(y_coords, (list, np.ndarray)):
        raise TypeError("x_coords和y_coords必须是列表或numpy数组")
    
    # 2. 转换为numpy数组（统一处理逻辑，兼容列表和数组输入）
    x = np.asarray(x_coords)
    y = np.asarray(y_coords)
    
    # 3. 校验坐标为数字类型
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        raise ValueError("x_coords和y_coords的元素必须是数字（int/float）")
    
    # 4. 校验x和y长度一致（核心！否则无法一一对应成点）
    if len(x) != len(y):
        raise ValueError(f"x_coords和y_coords长度不一致：x长度={len(x)}，y长度={len(y)}")
    
    # 5. 校验非空
    if len(x) == 0:
        raise ValueError("x_coords和y_coords不能为空")
    
    # --------------------------
    # 绘图核心逻辑
    # --------------------------
    plt.figure(figsize=(8, 6))  # 画布大小（宽8，高6英寸）
    
    # 绘制散点（直接使用两个坐标向量）
    plt.scatter(
        x=x,
        y=y,
        color=color,
        marker=marker,
        s=markersize,
        alpha=0.8  # 透明度（避免点重叠遮挡）
    )
    
    # 图表样式优化
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(grid, alpha=0.3, linestyle="--")  # 浅灰色虚线网格
    plt.tight_layout()  # 自动调整布局，防止标签被截断
    
    # 保存图片（指定路径时）
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,  # 高分辨率（适合打印/投稿）
            bbox_inches="tight"  # 裁剪多余空白
        )
        print(f"图片已保存至：{save_path}")
    
    # 显示图片（本地运行弹出窗口，Jupyter中直接嵌入）
    plt.show()

# ------------------------------
# 示例：测试函数（两个坐标向量输入）
# ------------------------------
if __name__ == "__main__":
    # 1. 测试用坐标向量（列表格式）
    x = [1.2, 2.8, 3.4, 4.1, 5.5, 6.2]
    y = [3.5, 5.1, 2.9, 6.7, 4.3, 1.8]
    
    # 2. 基础用法（默认样式）
    print("=== 基础样式绘图 ===")
    plot_points(x_coords=x, y_coords=y)
    
    # 3. 自定义样式（红色三角形点，保存图片）
    print("\n=== 自定义样式绘图 ===")
    plot_points(
        x_coords=x,
        y_coords=y,
        title="Custom Points Plot (Two Vectors Input)",
        xlabel="X Value",
        ylabel="Y Value",
        color="#FF5733",  # 十六进制颜色
        marker="^",       # 三角形标记
        markersize=10,
        save_path="/DISK1/data/zczhou_23/work2/Simulator_V3/pngs/two_vectors_points.png"
    )
    
    # 4. 支持numpy数组输入（可选）
    print("\n=== numpy数组输入测试 ===")
    x_np = np.array([0, 1, 2, 3, 4])
    y_np = np.array([0, 2, 4, 6, 8])
    plot_points(x_coords=x_np, y_coords=y_np, color="green", marker="s")