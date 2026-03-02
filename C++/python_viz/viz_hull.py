# import pyvista as pv

# # ---------------------- 核心配置 ----------------------
# VTK_FILE_PATH = "/home/robot/AAA/UUV_Model/data/UUVHull.vtk"  # 你的VTK文件路径（替换为实际路径）
# # -------------------------------------------------------

# # 1. 加载VTK文件（自动识别文件格式，无需手动选择读取器）
# mesh = pv.read(VTK_FILE_PATH)

# # 2. 打印网格核心信息（查看点、面数量等，贴合你的建模需求）
# print("=" * 50)
# print("VTK文件加载成功，网格信息如下：")
# print(f"网格类型：{mesh.type}")
# print(f"点（Vertices）数量：{mesh.n_points}")
# print(f"面（Faces）数量：{mesh.n_faces}")
# print(f"网格边界框：\n{mesh.bounds}")  # [x_min, x_max, y_min, y_max, z_min, z_max]
# print(f"网格中心坐标：{mesh.center}")
# print("=" * 50)

# # 3. 3D可视化显示（交互式窗口，支持旋转、缩放、平移）
# plotter = pv.Plotter(title="UUV Hull VTK Mesh View")
# # 添加网格（显示面+边线，便于查看细节）
# plotter.add_mesh(
#     mesh,
#     color="lightblue",  # 网格颜色
#     show_edges=True,    # 显示面边线
#     edge_color="black", # 边线颜色
#     opacity=0.8         # 透明度（0-1）
# )
# # 添加坐标系（便于定位，X红、Y绿、Z蓝）
# plotter.add_axes()
# # 添加网格信息标签
# plotter.add_text(f"Points: {mesh.n_points}\nFaces: {mesh.n_faces}", position="lower_left")
# # 显示窗口（阻塞式，关闭窗口后程序结束）
# plotter.show()


import pyvista as pv
import numpy as np
import os

def visualize_mesh(vtk_path="uuv_mesh.vtk"):
    """
    读取并可视化 C++ 生成的 UUV 网格
    """
    # 1. 检查文件是否存在
    if not os.path.exists(vtk_path):
        print(f"错误: 找不到文件 '{vtk_path}'")
        print("请先运行 C++ 程序生成网格文件！")
        return

    print(f"正在读取: {vtk_path} ...")
    
    try:
        # 2. 读取 VTK 文件
        mesh = pv.read(vtk_path)
        
        # 打印基本信息用于核对
        print("-" * 30)
        print(f"网格统计信息:")
        print(f"  - 顶点数 (Points): {mesh.n_points}")
        print(f"  - 面元数 (Cells):  {mesh.n_cells}")
        
        # 检查数据字段
        print(f"  - 包含数据: {mesh.array_names}")
        bounds = mesh.bounds
        print(f"  - 尺寸范围 (Bounds):")
        print(f"    X: [{bounds[0]:.3f}, {bounds[1]:.3f}] (长度: {bounds[1]-bounds[0]:.3f})")
        print(f"    Y: [{bounds[2]:.3f}, {bounds[3]:.3f}]")
        print(f"    Z: [{bounds[4]:.3f}, {bounds[5]:.3f}]")
        print("-" * 30)

        # 3. 初始化绘图窗口
        # shape=(1, 2) 表示左右两个子图
        p = pv.Plotter(shape=(1, 2), window_size=(1600, 800), title="UUV Mesh Inspector")

        # === 左窗口：实体模型检查 ===
        p.subplot(0, 0)
        p.add_text("Solid View (Check Topology)", font_size=10)
        
        # 绘制网格
        # show_edges=True 非常重要，可以看到三角形的划分结构
        p.add_mesh(mesh, 
                   color="orange", 
                   show_edges=True, 
                   edge_color="black", 
                   line_width=1,
                   smooth_shading=False) # 关闭平滑着色，为了看清面元棱角
        
        p.add_axes()
        p.show_grid()
        p.camera_position = 'xz' # 侧视图

        # === 右窗口：法向量检查 ===
        p.subplot(0, 1)
        p.add_text("Normals Check (Blue=Outward)", font_size=10)

        # 绘制半透明的主体
        p.add_mesh(mesh, color="white", opacity=0.3, show_edges=False)

        # 尝试绘制法向量箭头
        # 如果 C++ 里写入了 "Normals" 这个 vector 数据
        if "Normals" in mesh.cell_data:
            # 获取面心
            centers = mesh.cell_centers()
            # 绘制箭头：位置在面心，方向取自 C++ 数据
            p.add_arrows(centers.points, mesh.cell_data["Normals"], 
                         mag=0.1,  # 箭头长度缩放，根据你的模型尺寸调整 (L=4m, mag=0.1比较合适)
                         color="blue")
        else:
            # 如果 C++ 没写 Normals 数据，让 PyVista 自己算一个来对比
            print("提示: 文件中未找到 'Normals' 数据，使用 PyVista 自动计算显示。")
            mesh_computed = mesh.compute_normals(cell_normals=True, point_normals=False)
            centers = mesh.cell_centers()
            p.add_arrows(centers.points, mesh_computed.cell_data["Normals"], mag=0.1, color="red")

        p.add_axes()
        
        # 链接两个窗口的相机，动一个另一个跟着动
        p.link_views()
        
        print("可视化窗口已打开。")
        print("按 'q' 退出，按 'w' 切换线框模式。")
        p.show()

    except Exception as e:
        print(f"读取或可视化过程中发生错误: {e}")

if __name__ == "__main__":
    # 确保这里的路径和你 C++ 代码中 saveToVTK 的路径一致
    visualize_mesh("/home/robot/AAA/UUV_Model/data/UUVHull.vtk")