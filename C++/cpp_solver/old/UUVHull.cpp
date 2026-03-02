#include "UUVHull.hpp"
using namespace std;

std::vector<Face> generateAlignedIcosahedron(double r) {
    if (r <= 0.0) {
        throw std::invalid_argument("外接球半径r必须大于0");
    }

    const double PI = std::acos(-1.0);
    // 正20面体几何常数
    // 纬度角的正切值 tan(theta) = 0.5, 也就是 arctan(0.5) ≈ 26.565度
    // 对应的 X 坐标位置为 r / sqrt(5)
    const double x_ring = r / std::sqrt(5.0);
    const double r_ring = 2.0 * r / std::sqrt(5.0); // 环在 YZ 平面上的半径

    std::vector<Eigen::Vector3d> vertices;
    vertices.reserve(12);

    // 1. 生成12个顶点 (X轴为极轴)
    
    // 顶点 0: 前极点 (+X)
    vertices.push_back(Eigen::Vector3d(r, 0.0, 0.0));

    // 顶点 1-5: 前环 (Ring 1), X > 0
    // 起始角度 0
    for (int i = 0; i < 5; ++i) {
        double angle = 2.0 * PI * i / 5.0;
        double y = r_ring * std::cos(angle);
        double z = r_ring * std::sin(angle);
        vertices.push_back(Eigen::Vector3d(x_ring, y, z));
    }

    // 顶点 6-10: 后环 (Ring 2), X < 0
    // 起始角度 36度 (PI/5)，与前环交错
    for (int i = 0; i < 5; ++i) {
        double angle = 2.0 * PI * i / 5.0 + PI / 5.0;
        double y = r_ring * std::cos(angle);
        double z = r_ring * std::sin(angle);
        vertices.push_back(Eigen::Vector3d(-x_ring, y, z));
    }

    // 顶点 11: 后极点 (-X)
    vertices.push_back(Eigen::Vector3d(-r, 0.0, 0.0));

    // 2. 定义三角形拓扑结构 (右手定则，法线向外)
    std::vector<std::tuple<int, int, int>> indices;
    indices.reserve(20);

    // Group 1: 连接前极点 (0) 和前环 (1-5) 的5个面
    // 顺序: 0 -> Ring[i] -> Ring[next]
    for (int i = 0; i < 5; ++i) {
        int v1 = 0;
        int v2 = 1 + i;
        int v3 = 1 + ((i + 1) % 5);
        indices.emplace_back(v1, v2, v3);
    }

    // Group 2: 中间带 (连接前环和后环) 的10个面
    // 形成锯齿状带
    for (int i = 0; i < 5; ++i) {
        int r1_curr = 1 + i;
        int r1_next = 1 + ((i + 1) % 5);
        int r2_curr = 6 + i;
        int r2_next = 6 + ((i + 1) % 5);

        // 三角形 A (倒三角): Ring1_curr -> Ring2_curr -> Ring1_next
        indices.emplace_back(r1_curr, r2_curr, r1_next);
        
        // 三角形 B (正三角): Ring2_curr -> Ring2_next -> Ring1_next
        indices.emplace_back(r2_curr, r2_next, r1_next);
    }

    // Group 3: 连接后极点 (11) 和后环 (6-10) 的5个面
    // 顺序: 11 -> Ring2[next] -> Ring2[curr] (注意顺序以保证法向朝外)
    for (int i = 0; i < 5; ++i) {
        int v1 = 11;
        int v2 = 6 + ((i + 1) % 5);
        int v3 = 6 + i;
        indices.emplace_back(v1, v2, v3);
    }

    // 3. 构建 Face 对象 (计算属性)
    std::vector<Face> faces;
    faces.reserve(20);

    for (size_t i = 0; i < indices.size(); ++i) {
        auto [idx1, idx2, idx3] = indices[i];
        
        Face f;
        f.id = static_cast<int>(i);
        f.v1 = vertices[idx1];
        f.v2 = vertices[idx2];
        f.v3 = vertices[idx3];

        // 计算中心
        f.center = (f.v1 + f.v2 + f.v3) / 3.0;

        // 计算法向量
        Eigen::Vector3d e1 = f.v2 - f.v1;
        Eigen::Vector3d e2 = f.v3 - f.v1;
        Eigen::Vector3d normal = e1.cross(e2);
        
        f.area = 0.5 * normal.norm(); // 面积
        
        f.normal = normal.normalized();

        // 鲁棒性检查：确保法线指向外侧 (Center dot Normal > 0)
        // 因为这是一个凸包，且原点在内部
        if (f.normal.dot(f.center) < 0) {
            f.normal = -f.normal;
            // 实际上如果顶点顺序正确不需要这一步，但为了保险起见保留
        }

        faces.push_back(f);
    }

    return faces;
}


UUVHull::UUVHull(const std::string& path) : filePath(path) {
    // 构造函数逻辑：存在则读取，不存在则生成
    std::ifstream ifs(filePath);
    if (ifs.good()) {
        ifs.close();
        loadFromVTK();
    } else {
        generateMesh();
        saveToVTK();
    }
}

// 投影点到胶囊体表面 (Capsule Surface Projection)
Vector3d UUVHull::projectToCapsule(const Vector3d& p, double r, double h_half) {
    Vector3d new_p = p;
    
    // 1. 前半球 (x > h_half)
    if (p.x() > h_half) {
        Vector3d center(h_half, 0, 0);
        Vector3d dir = p - center;
        new_p = center + dir.normalized() * r;
    } 
    // 2. 后半球 (x < -h_half)
    else if (p.x() < -h_half) {
        Vector3d center(-h_half, 0, 0);
        Vector3d dir = p - center;
        new_p = center + dir.normalized() * r;
    }
    // 3. 圆柱体 ( -h_half <= x <= h_half )
    else {
        // 保持 x 不变，只把 yz 投影到圆上
        double yz_len = std::sqrt(p.y()*p.y() + p.z()*p.z());
        if (yz_len > 1e-9) {
            double scale = r / yz_len;
            new_p.y() *= scale;
            new_p.z() *= scale;
        }
    }
    return new_p;
}

std::vector<std::tuple<Vector3d, Vector3d, Vector3d>> UUVHull::generateFourTriangles(const Vector3d& p1, const Vector3d& p2, const Vector3d& p3, double r){
    Vector3d line1_middle=(p1+p2)/2;
    Vector3d line2_middle=(p1+p3)/2;
    Vector3d line3_middle=(p3+p2)/2;
    line1_middle=line1_middle/line1_middle.norm()*r;
    line2_middle=line2_middle/line2_middle.norm()*r;
    line3_middle=line3_middle/line3_middle.norm()*r;
    std::tuple<Vector3d, Vector3d, Vector3d>triangle1{p1,line1_middle,line2_middle};
    std::tuple<Vector3d, Vector3d, Vector3d>triangle2{line1_middle,p2,line3_middle};
    std::tuple<Vector3d, Vector3d, Vector3d>triangle3{line1_middle,line2_middle,line3_middle};
    std::tuple<Vector3d, Vector3d, Vector3d>triangle4{line2_middle,line3_middle,p3};
    std::vector<std::tuple<Vector3d, Vector3d, Vector3d>> result{triangle1,triangle2,triangle3,triangle4};
    return result;
}
void UUVHull::generateFourTriangles(const Vector3d& p1, const Vector3d& p2, const Vector3d& p3, 
                                        int depth, double r, double h_half){
    if(depth==0){
        faces.push_back(calculateFace(p1,p2,p3));
        return;
    }
    // 取中点
    Vector3d m1 = (p1 + p2) * 0.5;
    Vector3d m2 = (p2 + p3) * 0.5;
    Vector3d m3 = (p3 + p1) * 0.5;
    // 关键：将中点投影回表面
    m1 = projectToCapsule(m1, r, h_half);
    m2 = projectToCapsule(m2, r, h_half);
    m3 = projectToCapsule(m3, r, h_half);

    // 递归
    generateFourTriangles(p1, m1, m3, depth - 1, r, h_half);
    generateFourTriangles(m1, p2, m2, depth - 1, r, h_half);
    generateFourTriangles(m3, m2, p3, depth - 1, r, h_half);
    generateFourTriangles(m1, m2, m3, depth - 1, r, h_half);
}

std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> 
    UUVHull::processStretchedTriangle(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3, 
                                    int nums_h_cut, double r, double h_half) {
    
    std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> result;
    
    // 1. 顶点分类：找出底边(2个点)和顶点(1个点)
    std::vector<Eigen::Vector3d> group_pos, group_neg;
    if (p1.x() > 0) group_pos.push_back(p1); else group_neg.push_back(p1);
    if (p2.x() > 0) group_pos.push_back(p2); else group_neg.push_back(p2);
    if (p3.x() > 0) group_pos.push_back(p3); else group_neg.push_back(p3);

    Eigen::Vector3d base1, base2, apex;
    bool base_is_positive; // 标记底边是否在正半轴

    if (group_pos.size() == 2) {
        base1 = group_pos[0];
        base2 = group_pos[1];
        apex = group_neg[0];
        base_is_positive = true;
    } else {
        base1 = group_neg[0];
        base2 = group_neg[1];
        apex = group_pos[0];
        base_is_positive = false;
    }

    // 2. 拉伸顶点 (应用 h_cyl 偏移)
    // 原始点来自于半径为r的球，x坐标不为0
    if (base_is_positive) {
        base1.x() += h_half;
        base2.x() += h_half;
        apex.x()  -= h_half;
    } else {
        base1.x() -= h_half;
        base2.x() -= h_half;
        apex.x()  += h_half;
    }

    // 3. 循环切分生成三角形
    // 我们从底边 (base1, base2) 开始，向 apex 逼近
    // Level 0 是 base, Level nums_h_cut 是 apex
    
    Eigen::Vector3d prev_v1 = base1;
    Eigen::Vector3d prev_v2 = base2;

    for (int k = 1; k <= nums_h_cut; ++k) {
        double t = static_cast<double>(k) / nums_h_cut;
        
        // 线性插值计算当前层的高度点
        Eigen::Vector3d curr_v1 = base1 + (apex - base1) * t;
        Eigen::Vector3d curr_v2 = base2 + (apex - base2) * t;

        // 【关键步骤】圆柱面投影修正
        // 插值点是弦上的点（在圆柱内部），必须推到表面
        // 只有当不是顶点(k=nums_h_cut)时才需要计算(顶点本身就在球面上，虽然投影一下也没错)
        auto projectToCylinder = [&](Eigen::Vector3d& v) {
            double yz_norm = std::sqrt(v.y()*v.y() + v.z()*v.z());
            if (yz_norm > 1e-6) {
                double scale = r / yz_norm;
                v.y() *= scale;
                v.z() *= scale;
            }
        };

        // 最后一点汇聚于apex，apex本身就在半径r上，不需要过度修正，但为了数值统一可以做
        projectToCylinder(curr_v1);
        projectToCylinder(curr_v2);

        // 4. 生成面元
        if (k == nums_h_cut) {
            // 最后一层：是一个三角形（顶端汇聚）
            // 注意：此时 curr_v1 和 curr_v2 理论上非常接近 apex，但在数值上就是 apex
            // 为了避免生成退化三角形，直接使用 apex
            result.emplace_back(prev_v1, prev_v2, apex);
        } else {
            // 中间层：是一个四边形，切分为两个三角形
            // 三角形 1: (Prev_L, Prev_R, Curr_R)
            result.emplace_back(prev_v1, prev_v2, curr_v2);
            // 三角形 2: (Prev_L, Curr_R, Curr_L)
            result.emplace_back(prev_v1, curr_v2, curr_v1);
        }

        // 更新上一层节点
        prev_v1 = curr_v1;
        prev_v2 = curr_v2;
    }

    return result;
}

// 辅计算三角形属性
Face UUVHull::calculateFace(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3){
    Face f;
    f.id = (int)faces.size();
    f.v1 = p1; f.v2 = p2; f.v3 = p3;
    f.center = (p1 + p2 + p3) / 3.0;

    Vector3d edge1 = p2 - p1;
    Vector3d edge2 = p3 - p1;
    Vector3d cross = edge1.cross(edge2);
    f.area = 0.5 * cross.norm();
    f.normal = cross.normalized();
    if (f.normal.dot(f.center) < 0) {
        f.normal = -f.normal;
        // 实际上如果顶点顺序正确不需要这一步，但为了保险起见保留
    }
    return f;
};

void UUVHull::generateMesh() {
    std::cout << "Generating UUV Mesh..." << std::endl;
    // 参数定义
    double L = 4.0;
    double r = 0.2858;
    double h_cyl = 3.4284;

    faces.clear();
    faces.reserve(20*4^4+211*96);

    // 先获取正20面体, a = 4r / sqrt(10+2*sqrt(5)) = 0.3005
    vector<Face> Regular_Icosahedron{generateAlignedIcosahedron(r)};
    vector<tuple<Vector3d,Vector3d,Vector3d>>face_prepare;
    // h_cyl / (a/2*sqrt(3)) = 26.3479
    // 划分出 27 份
    double Icosahedron_sidelen=4*r/sqrt(10+2*sqrt(5));
    double h_Icosahedron_sidelen=Icosahedron_sidelen / 2 * sqrt(3);
    int nums_h_cut= static_cast<int>(h_cyl / h_Icosahedron_sidelen);
    if(nums_h_cut%2==0)nums_h_cut+=1;

    for (auto& face:Regular_Icosahedron){
        if(face.v1[0]>0 && face.v2[0]>0 && face.v3[0]>0){
            face.v1[0]+=h_cyl/2;
            face.v2[0]+=h_cyl/2;
            face.v3[0]+=h_cyl/2;
            face_prepare.push_back({face.v1,face.v2,face.v3});
            continue;
        }
        if(face.v1[0]<0 && face.v2[0]<0 && face.v3[0]<0){
            face.v1[0]-=h_cyl/2;
            face.v2[0]-=h_cyl/2;
            face.v3[0]-=h_cyl/2;
            face_prepare.push_back({face.v1,face.v2,face.v3});
            continue;
        }
        // 情况 3: 跨越中间的面 (连接环) -> 调用新函数进行拉伸和切分
        // 这一步会生成圆柱面上的多个小三角形
        auto strip_faces = processStretchedTriangle(face.v1, face.v2, face.v3, nums_h_cut, r, h_cyl/2);
        
        // 将切分好的小三角形加入列表
        face_prepare.insert(face_prepare.end(), strip_faces.begin(), strip_faces.end());
        
    }

    for (const auto& face : face_prepare) {
        generateFourTriangles(std::get<0>(face), std::get<1>(face), std::get<2>(face), 4, r, h_cyl/2.0);
    }

    std::cout << "Mesh Generated. Total Faces: " << faces.size() << std::endl;
}


// 2. 保存为 Legacy VTK 格式 (ASCII)
void UUVHull::saveToVTK() {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot save to " << filePath << std::endl;
        return;
    }

    int n_cells = faces.size();
    int n_points = n_cells * 3; // 简单模式：不共享顶点

    // Header
    file << "# vtk DataFile Version 3.0\n";
    file << "UUV Hydrodynamic Mesh\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // Points
    file << "POINTS " << n_points << " float\n";
    for (const auto& f : faces) {
        file << f.v1.x() << " " << f.v1.y() << " " << f.v1.z() << "\n";
        file << f.v2.x() << " " << f.v2.y() << " " << f.v2.z() << "\n";
        file << f.v3.x() << " " << f.v3.y() << " " << f.v3.z() << "\n";
    }

    // Cells
    file << "CELLS " << n_cells << " " << n_cells * 4 << "\n";
    for (int i = 0; i < n_cells; ++i) {
        int base = i * 3;
        file << "3 " << base << " " << base + 1 << " " << base + 2 << "\n";
    }

    // Cell Types (5 = Triangle)
    file << "CELL_TYPES " << n_cells << "\n";
    for (int i = 0; i < n_cells; ++i) file << "5\n";

    // Field Data (保存我们关心的属性)
    file << "CELL_DATA " << n_cells << "\n";
    
    // 1. ID
    file << "SCALARS FaceID int 1\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto& f : faces) file << f.id << "\n";

    // 2. Area
    file << "SCALARS Area float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto& f : faces) file << f.area << "\n";

    // 3. Normal (Vectors)
    file << "VECTORS Normals float\n";
    for (const auto& f : faces) {
        file << f.normal.x() << " " << f.normal.y() << " " << f.normal.z() << "\n";
    }
    
    // 4. Center (Vectors) - 用于调试中心位置是否正确
    file << "VECTORS Centers float\n";
    for (const auto& f : faces) {
        file << f.center.x() << " " << f.center.y() << " " << f.center.z() << "\n";
    }

    file.close();
    std::cout << "Saved mesh to " << filePath << std::endl;
}


// 3. 读取 VTK 文件
void UUVHull::loadFromVTK() {
    // 解析 VTK 文件稍微麻烦一点，但为了保持一致性是值得的。
    // 这里需要编写一个简单的解析器，读取 POINTS 和 CELLS。
    // 然后重新计算 center, normal, area (或者从文件读)
    // 建议：C++ 仿真时其实只依赖 generate 出来的内存数据。
    // load 功能主要是为了验证或者复用。
    // 简便做法：既然 generate很快，仿真端可以每次都 generate，
    // 只有 Python 端需要 load 文件。
    std::cout << "Loading mesh from " << filePath << " (Not fully implemented, generating instead...)" << std::endl;
    generateMesh(); // 偷懒做法：直接重新生成，反正毫秒级
}