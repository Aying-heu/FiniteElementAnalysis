import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# 加载 STL 文件
stl_file_path = "/home/robot/AAA/FiniteElementAnalysis/use_model/model/2.STL"
mesh = pv.read(stl_file_path)
roi_mesh=pv.read(stl_file_path)

center = mesh.points.mean(axis=0)
print("STL模型重心坐标：", center)

# 获取形心（重心）
centroid = mesh.center_of_mass()
print("STL模型形心坐标：", centroid)


def compute_volume_weighted_centroid(mesh):
    cells = mesh.faces.reshape(-1, 4)[:, 1:]
    points = mesh.points
    centroids = []
    volumes = []
    
    for cell in cells:
        v0, v1, v2 = points[cell]
        # 计算三角形面积矢量
        cross_prod = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross_prod)
        # 三角形质心
        tri_centroid = (v0 + v1 + v2) / 3.0
        centroids.append(tri_centroid)
        volumes.append(area)
    
    volumes = np.array(volumes)
    centroids = np.array(centroids)
    # 体积加权平均
    return np.average(centroids, axis=0, weights=volumes)

true_centroid = compute_volume_weighted_centroid(mesh)
print("精确形心:", true_centroid)


mesh.points-=true_centroid
roi_mesh.points-=true_centroid
print("将mesh重心移动到精确形心中心。")

# 计算模型尺寸
bounds = mesh.bounds
x_length = bounds[1] - bounds[0]
y_length = bounds[3] - bounds[2]
z_length = bounds[5] - bounds[4]
print(f"模型尺寸: X={x_length:.2f}, Y={y_length:.2f}, Z={z_length:.2f}")
# 检查模型完整性
if not mesh.is_manifold:
    print("警告：模型非流形！")
if mesh.n_open_edges > 0:
    print(f"警告：有{mesh.n_open_edges}条开放边！")
# 创建可视化窗口
plotter = pv.Plotter()
actor = plotter.add_mesh(mesh, color='lightgray', opacity=0.8, smooth_shading=True)
plotter.show(auto_close=False)  # 打开窗口但不关闭
plotter.add_axes(line_width=3, labels_off=False)
print(mesh.volume)

# 动态旋转并展示
n_frames = 1000
for i in tqdm(range(n_frames)):
    # 旋转角度
    angle = i * 2 * np.pi / n_frames
    rot = R.from_euler('y', angle).as_matrix()
    # 对所有点做旋转
    mesh.points = (rot @ roi_mesh.points.T).T
    plotter.render()

    if i == 2:
        camera_position = plotter.camera.position
        camera_focus = plotter.camera.focal_point
        # 将相机的z坐标取负
        new_camera_position = (camera_position[1], camera_position[0], -camera_position[2])
        # 更新相机位置，保持关注点不变
        plotter.camera.position = new_camera_position
        plotter.camera.focal_point = camera_focus
        plotter.camera.up = (0, 0, -1)  # 设置z轴向下


plotter.close()


# import numpy as np
# import trimesh
# import sys
# import pyvista as pv
# from scipy.spatial import Delaunay, ConvexHull
# import networkx as nx
# from collections import deque
# import logging

# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def read_stl_file(file_path):
#     """读取STL文件并返回trimesh对象"""
#     try:
#         mesh = trimesh.load(file_path)
#         logger.info(f"成功读取STL文件: {file_path}")
#         return mesh
#     except Exception as e:
#         logger.error(f"读取STL文件时出错: {e}")
#         sys.exit(1)

# def visualize_stl(mesh, title="STL模型可视化"):
#     """可视化STL模型"""
#     try:
#         plotter = pv.Plotter()
#         pv_mesh = pv.wrap(mesh)
#         plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.8, 
#                         smooth_shading=True, show_edges=True)
#         plotter.add_title(title)
#         plotter.show()
#     except Exception as e:
#         logger.error(f"可视化失败: {e}")

# def analyze_stl(mesh):
#     """分析STL模型的基本信息"""
#     vertex_count = len(mesh.vertices)
#     face_count = len(mesh.faces)
#     edge_counts = [len(face) for face in mesh.faces]
#     all_triangles = all(count == 3 for count in edge_counts)
#     surface_area = mesh.area
    
#     logger.info("STL模型分析结果:")
#     logger.info(f"顶点个数: {vertex_count}")
#     logger.info(f"面个数: {face_count}")
#     logger.info(f"面的边数分布: {set(edge_counts)}")
#     logger.info(f"所有面都是三角形: {'是' if all_triangles else '否'}")
#     logger.info(f"表面积: {surface_area:.4f} 平方单位")
    
#     return surface_area

# def calculate_face_quality(face_vertices):
#     """计算三角形面元的质量（0-1，1为等边三角形）"""
#     if len(face_vertices) != 3:
#         return 0.0  # 仅处理三角形
    
#     v0, v1, v2 = face_vertices
#     a = np.linalg.norm(v1 - v0)
#     b = np.linalg.norm(v2 - v1)
#     c = np.linalg.norm(v0 - v2)
    
#     # 使用三角形质量公式
#     area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    
#     # 避免除零错误
#     if area < 1e-10:
#         return 0.0
        
#     # 质量公式：4√3 * area / (a² + b² + c²)
#     quality = 4 * np.sqrt(3) * area / (a**2 + b**2 + c**2)
#     return max(0.0, min(1.0, quality))

# def split_triangle(face_vertices):
#     """将三角形分割成4个小三角形"""
#     v0, v1, v2 = face_vertices
#     mid01 = (v0 + v1) / 2
#     mid12 = (v1 + v2) / 2
#     mid20 = (v2 + v0) / 2
    
#     return [
#         [v0, mid01, mid20],
#         [v1, mid01, mid12],
#         [v2, mid12, mid20],
#         [mid01, mid12, mid20]
#     ]

# def split_elongated_triangle(face_vertices):
#     """将细长三角形分割成两个三角形"""
#     v0, v1, v2 = face_vertices
    
#     # 计算各边长度
#     edges = [
#         np.linalg.norm(v1 - v0),
#         np.linalg.norm(v2 - v1),
#         np.linalg.norm(v0 - v2)
#     ]
    
#     # 找到最长边
#     longest_idx = np.argmax(edges)
    
#     if longest_idx == 0:  # 边0-1
#         mid = (v0 + v1) / 2
#         return [[v0, mid, v2], [mid, v1, v2]]
#     elif longest_idx == 1:  # 边1-2
#         mid = (v1 + v2) / 2
#         return [[v0, v1, mid], [v0, mid, v2]]
#     else:  # 边2-0
#         mid = (v2 + v0) / 2
#         return [[v0, v1, mid], [v1, v2, mid]]

# def build_face_adjacency(mesh):
#     """构建面邻接关系图"""
#     logger.info("构建面邻接关系图...")
#     # 创建边到面的映射
#     edge_to_faces = {}
#     for i, face in enumerate(mesh.faces):
#         # 三角形的三条边
#         edges = [
#             tuple(sorted((face[0], face[1]))),
#             tuple(sorted((face[1], face[2]))),
#             tuple(sorted((face[2], face[0])))
#         ]
#         for edge in edges:
#             if edge not in edge_to_faces:
#                 edge_to_faces[edge] = []
#             edge_to_faces[edge].append(i)
    
#     # 构建邻接图
#     adjacency = {i: set() for i in range(len(mesh.faces))}
#     for edge, faces in edge_to_faces.items():
#         if len(faces) >= 2:  # 共享边的面
#             for i in range(len(faces)):
#                 for j in range(i+1, len(faces)):
#                     adjacency[faces[i]].add(faces[j])
#                     adjacency[faces[j]].add(faces[i])
#     return adjacency

# def robust_delaunay_triangulation(points):
#     """鲁棒的Delaunay三角化，处理共面点情况"""
#     try:
#         # 尝试直接进行3D Delaunay
#         return Delaunay(points)
#     except Exception as e:
#         logger.warning(f"3D Delaunay失败，尝试2D投影: {e}")
#         try:
#             # 使用PCA找到最佳投影平面
#             centroid = np.mean(points, axis=0)
#             centered = points - centroid
#             _, _, vh = np.linalg.svd(centered)
#             proj_plane = vh[:2]
#             proj_points = np.dot(centered, proj_plane.T)
#             return Delaunay(proj_points)
#         except Exception as e2:
#             logger.error(f"2D投影Delaunay失败: {e2}")
#             return None

# def merge_small_faces(mesh, small_face_indices, adjacency, min_area):
#     """合并相邻的小面元"""
#     logger.info(f"合并 {len(small_face_indices)} 个小面元...")
#     # 创建图用于区域增长
#     graph = nx.Graph()
#     for face_idx in small_face_indices:
#         graph.add_node(face_idx)
#         for neighbor in adjacency[face_idx]:
#             if neighbor in small_face_indices:
#                 graph.add_edge(face_idx, neighbor)
    
#     # 找到所有连通分量
#     merged_faces = []
#     visited = set()
#     merge_failures = 0
    
#     for comp in nx.connected_components(graph):
#         if len(comp) < 2:  # 单个面不合并
#             continue
            
#         # 收集所有顶点
#         all_vertices = []
#         for face_idx in comp:
#             face_vertices = mesh.vertices[mesh.faces[face_idx]]
#             all_vertices.append(face_vertices)
#             visited.add(face_idx)
        
#         # 合并顶点
#         merged_vertices = np.vstack(all_vertices)
        
#         # 跳过点太少的组件
#         if len(merged_vertices) < 3:
#             logger.warning(f"连通分量只有 {len(merged_vertices)} 个顶点，跳过合并")
#             merge_failures += 1
#             continue
        
#         # 尝试Delaunay三角化
#         delaunay = robust_delaunay_triangulation(merged_vertices)
        
#         if delaunay is not None:
#             merged_triangles = []
#             for simplex in delaunay.simplices:
#                 if len(simplex) == 3:  # 只处理三角形
#                     triangle = merged_vertices[simplex]
#                     merged_triangles.append(triangle)
#             if merged_triangles:
#                 merged_faces.extend(merged_triangles)
#                 continue
        
#         # 如果Delaunay失败，尝试凸包
#         try:
#             hull = ConvexHull(merged_vertices)
#             hull_vertices = merged_vertices[hull.vertices]
#             # 将凸包多边形三角化
#             triangles = []
#             for i in range(1, len(hull_vertices)-1):
#                 triangles.append([hull_vertices[0], hull_vertices[i], hull_vertices[i+1]])
#             merged_faces.extend(triangles)
#         except Exception as e:
#             logger.error(f"凸包计算失败: {e}")
#             merge_failures += 1
#             # 合并失败，保留原始小面
#             for face_idx in comp:
#                 merged_faces.append(mesh.vertices[mesh.faces[face_idx]])
    
#     if merge_failures > 0:
#         logger.warning(f"有 {merge_failures} 个小面元组合并失败")
    
#     return merged_faces, visited

# def process_facets(mesh, target_area, target_face_count):
#     """自适应处理面元以保持面数在目标范围内"""
#     logger.info("开始处理面元...")
#     # 计算每个面的面积和质量
#     face_areas = mesh.area_faces
#     face_qualities = [calculate_face_quality(mesh.vertices[face]) for face in mesh.faces]
    
#     # 识别需要处理的面
#     large_faces = []
#     small_faces = []
#     elongated_faces = []
    
#     for i, (area, quality) in enumerate(zip(face_areas, face_qualities)):
#         if area > 2.0 * target_area:
#             large_faces.append(i)
#         elif area < 0.5 * target_area:
#             small_faces.append(i)
#         elif quality < 0.3:  # 质量低于0.3视为细长面
#             elongated_faces.append(i)
    
#     logger.info(f"检测到大面: {len(large_faces)}，小面: {len(small_faces)}，细长面: {len(elongated_faces)}")
    
#     # 构建邻接关系
#     adjacency = build_face_adjacency(mesh)
    
#     # 合并小面
#     merged_faces = []
#     small_face_set = set(small_faces)
#     merged_regions, merged_indices = merge_small_faces(mesh, small_face_set, adjacency, 0.5 * target_area)
#     merged_faces.extend(merged_regions)
    
#     # 处理未合并的小面和需要处理的面
#     processed_faces = []
#     handled_faces = set(merged_indices)
    
#     for i in range(len(mesh.faces)):
#         if i in handled_faces:
#             continue
            
#         face_vertices = mesh.vertices[mesh.faces[i]]
        
#         # 处理大面
#         if i in large_faces:
#             if face_areas[i] > 4.0 * target_area:
#                 # 非常大的面，分割两次
#                 for sub_face in split_triangle(face_vertices):
#                     processed_faces.extend(split_triangle(sub_face))
#             else:
#                 processed_faces.extend(split_triangle(face_vertices))
        
#         # 处理细长面
#         elif i in elongated_faces:
#             processed_faces.extend(split_elongated_triangle(face_vertices))
        
#         # 保留其他面
#         else:
#             processed_faces.append(face_vertices)
    
#     # 添加合并后的面
#     processed_faces.extend(merged_faces)
    
#     # 重建网格
#     processed_mesh = create_mesh_from_faces(processed_faces)
    
#     if processed_mesh is None:
#         logger.error("网格重建失败，返回原始网格")
#         return mesh
    
#     # 调整面数到目标范围
#     current_face_count = len(processed_mesh.faces)
#     logger.info(f"初步处理后: {current_face_count} 个面")
    
#     # 面数过多时简化
#     if current_face_count > target_face_count * 1.2:
#         logger.info(f"简化网格至 {target_face_count} 个面")
#         try:
#             # 正确调用简化函数 - 使用face_count参数
#             processed_mesh = processed_mesh.simplify_quadric_decimation(face_count=target_face_count)
#         except Exception as e:
#             logger.error(f"网格简化失败: {e}")
#             # 简化失败时尝试使用边折叠简化
#             try:
#                 logger.warning("尝试使用边折叠简化")
#                 processed_mesh = processed_mesh.simplify(target_count=target_face_count)
#             except Exception as e2:
#                 logger.error(f"边折叠简化也失败: {e2}")
#                 logger.warning("使用处理后的网格不简化")
    
#     # 面数过少时细分
#     elif current_face_count < target_face_count * 0.8:
#         logger.info(f"细分网格至约 {target_face_count} 个面")
#         max_subdivisions = 2
#         subdivisions = 0
#         while (len(processed_mesh.faces) < target_face_count * 0.8 and 
#                subdivisions < max_subdivisions):
#             processed_mesh = processed_mesh.subdivide()
#             subdivisions += 1
#             logger.info(f"细分后: {len(processed_mesh.faces)} 个面")
    
#     return processed_mesh

# def create_mesh_from_faces(processed_faces):
#     """正确重建网格"""
#     logger.info("重建网格...")
#     if not processed_faces:
#         logger.error("没有面可处理")
#         return None
    
#     all_vertices = []
#     all_faces = []
#     vertex_map = {}  # 顶点坐标到索引的映射
    
#     # 顶点去重和面重建
#     for face in processed_faces:
#         if len(face) < 3:
#             logger.warning(f"跳过无效面（顶点数 < 3）")
#             continue
            
#         face_indices = []
#         for vertex in face:
#             # 生成顶点哈希值（精度为6位小数）
#             key = tuple(np.round(vertex, 6))
            
#             if key not in vertex_map:
#                 vertex_map[key] = len(all_vertices)
#                 all_vertices.append(vertex)
            
#             face_indices.append(vertex_map[key])
        
#         # 三角化非三角形面
#         if len(face_indices) == 3:
#             all_faces.append(face_indices)
#         else:
#             # 对多边形进行三角化
#             poly_vertices = np.array([all_vertices[i] for i in face_indices])
#             try:
#                 # 使用2D投影进行Delaunay三角化
#                 centroid = np.mean(poly_vertices, axis=0)
#                 vectors = poly_vertices - centroid
#                 # 使用主成分分析找到最佳投影平面
#                 _, _, vh = np.linalg.svd(vectors)
#                 proj_plane = vh[:2]
#                 proj_points = np.dot(vectors, proj_plane.T)
                
#                 # 进行Delaunay三角化
#                 tri = Delaunay(proj_points)
#                 for simplex in tri.simplices:
#                     if len(simplex) == 3:  # 确保是三角形
#                         all_faces.append([
#                             face_indices[simplex[0]],
#                             face_indices[simplex[1]],
#                             face_indices[simplex[2]]
#                         ])
#             except Exception as e:
#                 logger.warning(f"多边形三角化失败: {e}")
#                 # 如果三角化失败，使用简单方法
#                 for i in range(1, len(face_indices) - 1):
#                     all_faces.append([face_indices[0], face_indices[i], face_indices[i+1]])
    
#     # 确保顶点和面是numpy数组
#     if not all_vertices or not all_faces:
#         logger.error("重建网格失败：没有顶点或面")
#         return None
        
#     vertices_array = np.array(all_vertices)
#     faces_array = np.array(all_faces)
    
#     # 创建网格前进行验证
#     if len(faces_array) == 0:
#         logger.error("没有创建任何面！")
#         return None
    
#     logger.info(f"重建网格: {len(vertices_array)} 个顶点, {len(faces_array)} 个面")
#     return trimesh.Trimesh(vertices=vertices_array, faces=faces_array)

# def visualize_processed_mesh(processed_mesh, title="处理后的网格模型"):
#     """可视化处理后的网格"""
#     try:
#         pv_mesh = pv.wrap(processed_mesh)
#         plotter = pv.Plotter()
        
#         # 添加网格
#         plotter.add_mesh(pv_mesh, show_edges=True, color='lightblue', 
#                         opacity=0.9, smooth_shading=False)
        
#         # 添加网格信息标注
#         if processed_mesh:
#             info = f"顶点数: {len(processed_mesh.vertices)}\n面数: {len(processed_mesh.faces)}"
#             plotter.add_text(info, position='lower_right', color='black', font_size=12)
        
#         plotter.add_title(title)
#         plotter.show()
#     except Exception as e:
#         logger.error(f"可视化失败: {e}")

# def main():
#     if len(sys.argv) != 2:
#         print("用法: python stl_processor.py <stl_file_path>")
#         print("示例: python stl_processor.py model.stl")
#         sys.exit(1)
    
#     stl_file_path = sys.argv[1]
    
#     # 读取STL文件
#     logger.info(f"正在读取STL文件: {stl_file_path}")
#     mesh = read_stl_file(stl_file_path)
    
#     # 可视化原始模型
#     visualize_stl(mesh, "原始STL模型")
    
#     # 分析STL模型
#     surface_area = analyze_stl(mesh)
    
#     # 设置目标面元数量范围
#     min_faces = 500
#     max_faces = 800
#     target_face_count = (min_faces + max_faces) // 2  # 目标650个面
    
#     # 计算目标面元面积
#     target_area = surface_area / target_face_count
#     logger.info(f"目标面元面积: {target_area:.6f} 平方单位")
#     logger.info(f"目标面元数量: {min_faces}-{max_faces} (目标: {target_face_count})")
    
#     # 处理面元
#     logger.info("正在处理面元...")
#     processed_mesh = process_facets(mesh, target_area, target_face_count)
    
#     if processed_mesh is None:
#         logger.error("网格处理失败！使用原始网格")
#         processed_mesh = mesh
    
#     # 确保最终面数在范围内
#     final_face_count = len(processed_mesh.faces)
#     if final_face_count < min_faces or final_face_count > max_faces:
#         logger.info(f"调整面数: {final_face_count} → {target_face_count}")
#         try:
#             # 正确调用简化函数 - 使用face_count参数
#             if final_face_count > target_face_count:
#                 processed_mesh = processed_mesh.simplify_quadric_decimation(face_count=target_face_count)
#             else:
#                 # 面数过少，我们无法通过简化减少，所以只能保持
#                 logger.warning("面数过少，无法通过简化达到目标，保持当前网格")
#         except Exception as e:
#             logger.error(f"最终网格简化失败: {e}")
#             # 尝试使用边折叠简化
#             try:
#                 logger.warning("尝试使用边折叠简化")
#                 processed_mesh = processed_mesh.simplify(target_count=target_face_count)
#             except Exception as e2:
#                 logger.error(f"边折叠简化也失败: {e2}")
#                 logger.warning("使用当前网格")
    
#     # 可视化处理结果
#     visualize_processed_mesh(processed_mesh, "优化后的网格模型")
    
#     # 计算处理后的表面积
#     processed_area = processed_mesh.area
#     logger.info(f"处理后的总表面积: {processed_area:.4f} 平方单位")
#     logger.info(f"面积变化率: {(processed_area/surface_area-1)*100:.2f}%")
    
#     # 导出处理后的模型
#     output_path = "optimized_model.stl"
#     processed_mesh.export(output_path)
#     logger.info(f"优化后的模型已导出至: {output_path}")
    
#     # 最终报告
#     logger.info("\n优化完成:")
#     logger.info(f"- 原始面数: {len(mesh.faces)}")
#     logger.info(f"- 优化后面数: {len(processed_mesh.faces)}")
#     logger.info(f"- 原始面积: {surface_area:.4f}")
#     logger.info(f"- 优化后面积: {processed_area:.4f}")

# if __name__ == "__main__":
#     main()