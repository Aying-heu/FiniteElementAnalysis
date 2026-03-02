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

class WaveForceEstimator{
public:
    WaveForceEstimator(double Hs=1.5,double T1=5,
                        int wave_N=30,int wave_M=20,int s=5,
                        double main_theta=0,std::string wave_type="JONSWAP");
    std::vector<double> compute_wave_force(double t,std::vector<Face>& uuv_hull_faces,
                                            double x,double y,double z,
                                            double roll,double pitch,double yaw);
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
    std::string wave_type;
    double norm_factor; // 预计算的归一化系数 C(s)
    std::string wave_params_root_path="/home/robot/AAA/UUV_Model/data/wave_parameter/";
    std::vector<Face>& uuv_hull_faces;

    std::vector<double>amplitudes;
    std::vector<double>omega;
    std::vector<double>delta_omega;
    std::vector<double>k;
    std::vector<double>phases;
    std::vector<double>wave_theta;



    void check_init_or_load();
    void resize_data();
    string generate_filename();
    void load_wave_params(const std::string& filename);
    void init_wave_spectrum();
    void save_wave_params(const std::string& filename);

    double wave_number(double omega_i, double depth=100.0);
    double S(double omega_i);
    double D(double theta_val);
    double compute_wave_elevation(double x, double y, double t);

    int idx(int i, int j) { return i * wave_M + j; };
};
