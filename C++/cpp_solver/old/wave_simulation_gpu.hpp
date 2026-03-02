#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include"UUVHull.hpp"

// 检查文件系统支持
namespace fs = std::filesystem;

const double PI = std::acos(-1.0);


extern "C" void launch_wave_kernel(
    const double* d_amp, 
    const double* d_omega, 
    const double* d_k, 
    const double* d_phase,
    const double* d_theta,
    int num_waves,

    const double* d_face_centers_x,
    const double* d_face_centers_y,
    const double* d_face_centers_z,
    const double* d_face_normals_x,
    const double* d_face_normals_y,
    const double* d_face_normals_z,
    const double* d_face_areas,
    int num_faces,

    double t,double x,double y,double z,
    double roll,double pitch,double yaw,
    double* d_result
);

class WaveForceEstimator{
public:
    // 构造函数：初始化波浪参数，生成网格    
    WaveForceEstimator(std::vector<Face>& faces, double Hs=1.5,double T1=5,
                        int wave_N=30,int wave_M=20,int s=5,
                        double main_theta=0,double depth=100.0,std::string wave_type="JONSWAP");

    // === 新增：禁止拷贝构造和赋值 ===
    // 这能防止显存被意外释放，如果代码中有按值传递的地方，编译器会报错提示你
    WaveForceEstimator(const WaveForceEstimator&) = delete;
    WaveForceEstimator& operator=(const WaveForceEstimator&) = delete;

    ~WaveForceEstimator(); 
    
    double compute_wave_elevation(double x, double y, double t);

    std::vector<double> compute_wave_force(double t,
                                            double x,double y,double z,
                                            double roll,double pitch,double yaw);  // double EnvironForceMoment[6]
private:
    // 波浪参数
    int wave_N; // 频率离散点数
    int wave_M; // 方向离散点数
    int s;       // 方向集中度   0-10
    double Hs;
    double T1;
    double g{9.81};
    double main_theta;
    double omega_min{0.1};
    double omega_max{3.0};
    double depth;
    std::string wave_type;
    double norm_factor; // 预计算的归一化系数 C(s)
    std::string wave_params_root_path="/home/robot/AAA/UUV_Model_New/data/wave_parameter/";
    std::vector<Face>& uuv_hull_faces;
    double uuv_total_area;
    
    std::vector<double> amplitudes;
    std::vector<double> omega;
    std::vector<double> delta_omega;
    std::vector<double> k;
    std::vector<double> phases;
    std::vector<double> wave_theta;
    // 辅助函数：二维转一维索引
    inline int idx(int i, int j) const {
        return i * wave_M + j;
    }

    // 初始化流程
    void check_init_or_load(); // 策略入口
    void convert_data_to_gpu_format();
    std::string generate_filename(); // 生成唯一文件名
    void init_wave_spectrum(); // 计算谱
    void resize_data(); // 分配内存

    // 辅助计算
    double S(double omega_i);  // JONSWAP    pierson-Moskowitz
    double D(double wave_theta_j);
    double wave_number(double omega_i, double depth);

    // IO
    void load_wave_params(const std::string& filename);
    void save_wave_params(const std::string& filename);
    

    // 预处理：将 Face 列表转为 GPU 友好的数组
    std::vector<double> face_centers_x, face_centers_y, face_centers_z;
    std::vector<double> face_normals_x, face_normals_y, face_normals_z;
    std::vector<double> face_areas;

    // === 新增：GPU 显存指针 ===
    // 静态数据（初始化后不变）
    double* d_amp = nullptr;
    double* d_omega = nullptr;
    double* d_k = nullptr;
    double* d_phase = nullptr;
    double* d_theta = nullptr;
    
    double* d_face_centers_x = nullptr;
    double* d_face_centers_y = nullptr;
    double* d_face_centers_z = nullptr;
    double* d_face_normals_x = nullptr;
    double* d_face_normals_y = nullptr;
    double* d_face_normals_z = nullptr;
    double* d_face_areas = nullptr;

    // 动态数据（结果，每帧变化）
    double* d_result = nullptr; // 存放 Force(3)+Moment(3)+Area(1)

    // 释放显存的辅助函数
    void free_gpu_memory();
    // 分配并拷贝静态数据的函数
    void init_gpu_memory();

};