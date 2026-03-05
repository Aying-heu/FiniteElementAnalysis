import pyvista as pv
import numpy as np

# 1. 导入STL文件
def load_stl_file(file_path):
    """加载STL文件并返回网格对象"""
    mesh = pv.read(file_path)
    print(f"已成功导入STL文件: {file_path}")
    print(f"网格信息: {mesh}")
    return mesh

# 2. 计算每个面的重心和法向量
def calculate_face_properties(mesh):
    """计算每个面的重心和法向量"""
    # 确保网格是三角网格
    # if not mesh.is_all_triangles():
    #     mesh = mesh.triangulate()
    #     print("已将网格转换为三角网格")
    
    # 获取所有三角形面
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # 忽略每个面开头的顶点计数
    
    # 存储结果：每个面的重心和法向量
    face_centers = []
    face_normals = []
    
    # 计算每个面的重心和法向量
    for face in faces:
        # 获取三角形的三个顶点
        v0, v1, v2 = mesh.points[face]
        
        # 计算重心 (三个顶点的平均值)
        center = (v0 + v1 + v2) / 3
        face_centers.append(center)
        
        # 计算法向量 (两个边的叉积)
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal /= np.linalg.norm(normal)  # 归一化
        face_normals.append(normal)
    
    return np.array(face_centers), np.array(face_normals)

# 3. 可视化STL模型和面属性
def visualize_stl(mesh, face_centers, face_normals):
    """可视化STL模型及其面的重心和法向量"""
    plotter = pv.Plotter()
    
    # 添加STL模型
    plotter.add_mesh(mesh, color='lightblue', opacity=0.8, label='STL Model')
    
    # 添加重心点
    centers = pv.PolyData(face_centers)
    plotter.add_mesh(centers, color='red', point_size=10, render_points_as_spheres=True, label='Face Centers')
    
    # 添加法向量箭头
    arrows = pv.Arrow()
    for center, normal in zip(face_centers, face_normals):
        # 创建箭头表示法向量
        arrow = arrows.copy()
        arrow.rotate_z(0)  # 清除旋转
        arrow.rotate_vector(normal, angle=0)  # 对齐方向
        arrow.translate(center - arrow.center)  # 移动到重心位置
        plotter.add_mesh(arrow, color='green', opacity=0.7)
    
    # 添加图例
    plotter.add_legend()
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 显示
    plotter.show()

# 主函数
def main():
    # 替换为你的STL文件路径
    stl_file_path = "/home/robot/AAA/FiniteElementAnalysis/use_model/model/1.STL"
    
    # 1. 加载STL文件
    mesh = load_stl_file(stl_file_path)
    
    # 2. 计算面属性
    face_centers, face_normals = calculate_face_properties(mesh)
    
    # 打印前5个面的结果
    print("\n前5个面的重心和法向量:")
    for i in range(min(5, len(face_centers))):
        print(f"面 {i+1}:")
        print(f"  重心: {face_centers[i]}")
        print(f"  法向量: {face_normals[i]}")
    
    # 3. 可视化
    visualize_stl(mesh, face_centers, face_normals)

if __name__ == "__main__":
    main()