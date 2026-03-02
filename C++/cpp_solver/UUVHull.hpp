#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <cmath>
#include <tuple>

using namespace std;
// 使用 Eigen 处理向量
using Eigen::Vector3d;
#define EPS 1e-5

// 面元结构体
struct Face {
    int id;
    Vector3d v1, v2, v3; // 三个顶点 (世界坐标系生成前，这是Body坐标系)
    Vector3d center;     // 中心点
    Vector3d normal;     // 法向量 (单位向量，指向外)
    double area;         // 面积
};

/**
 * @brief 生成正20面体，返回包含20个面的vector<Face>
 * @return vector<Face> 正20面体的所有面元，id从0到19依次编号
 * 
 * 返回的20面体，半径为sqrt( (5+sqrt(5)) / 2 )
 */
std::vector<Face> generateAlignedIcosahedron(double r);


class UUVHull {
public:
    std::string filePath;
    std::vector<Face> faces; // 存储所有面元

    UUVHull(const std::string& path);

    // 获取面元数量
    size_t getFaceCount() const { return faces.size(); }

private:
    // 1. 生成模型 (你的核心逻辑)
    void generateMesh();
    Face calculateFace(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3);

    Vector3d projectToCapsule(const Vector3d& p, double r, double h_half);

    std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>
        processStretchedTriangle(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3, 
                                            int nums_h_cut, double r, double h_half);
    // 传入三个顶点和半径r，返回4个三角形的顶点
    static std::vector<std::tuple<Vector3d, Vector3d, Vector3d>> generateFourTriangles(const Vector3d& p1, const Vector3d& p2, const Vector3d& p3, 
                                                                                        double r);
    void generateFourTriangles(const Vector3d& p1, const Vector3d& p2, const Vector3d& p3, 
                                    int depth, double r, double h_half);

    // 2. 保存为 Legacy VTK 格式 (ASCII)
    void saveToVTK();

    // 3. 读取 VTK 文件
    void loadFromVTK();
};