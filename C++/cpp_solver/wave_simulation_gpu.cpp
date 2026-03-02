#include "wave_simulation_gpu.hpp"
#include <iostream>
#include <random>

#define CHECK_CUDA_ERROR(expr) { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 辅助：生成随机数
double getRand(double min = 0.0, double max = 2 * PI) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

// 构造函数
WaveForceEstimator::WaveForceEstimator(std::vector<Face>& faces,double Hs, double T1,
                                       int wave_N, int wave_M, int s,
                                       double main_theta,double depth, std::string wave_type)
    : Hs(Hs), T1(T1), wave_N(wave_N), wave_M(wave_M), s(s),
      main_theta(main_theta/180*PI), wave_type(wave_type),
      uuv_hull_faces(faces),depth{depth}
{
    // 1. 确保目录存在
    if (!fs::exists(wave_params_root_path)) {
        fs::create_directories(wave_params_root_path);
    }
    
    // 2. 执行加载或初始化策略
    check_init_or_load();

    // 3. send to gpu
    convert_data_to_gpu_format();
    init_gpu_memory();
}
WaveForceEstimator::~WaveForceEstimator() {
    free_gpu_memory();
}
void WaveForceEstimator::init_gpu_memory() {
    int n_waves = wave_N * wave_M;
    int n_faces = uuv_hull_faces.size();

    // --- 波浪数据 ---
    CHECK_CUDA_ERROR(cudaMalloc(&d_amp, n_waves * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_omega, n_waves * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k, n_waves * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_phase, n_waves * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_theta, n_waves * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_amp, amplitudes.data(), n_waves * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_omega, omega.data(), n_waves * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k, k.data(), n_waves * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_phase, phases.data(), n_waves * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_theta, wave_theta.data(), n_waves * sizeof(double), cudaMemcpyHostToDevice));

    // --- 网格数据 ---
    CHECK_CUDA_ERROR(cudaMalloc(&d_face_centers_x, n_faces * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_face_centers_y, n_faces * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_face_centers_z, n_faces * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_face_normals_x, n_faces * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_face_normals_y, n_faces * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_face_normals_z, n_faces * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_face_areas, n_faces * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_face_centers_x, face_centers_x.data(), n_faces * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_face_centers_y, face_centers_y.data(), n_faces * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_face_centers_z, face_centers_z.data(), n_faces * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_face_normals_x, face_normals_x.data(), n_faces * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_face_normals_y, face_normals_y.data(), n_faces * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_face_normals_z, face_normals_z.data(), n_faces * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_face_areas, face_areas.data(), n_faces * sizeof(double), cudaMemcpyHostToDevice));

    // --- 结果数据 ---
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, 7 * sizeof(double)));
    // CHECK_CUDA_ERROR(cudaMemset(d_result, 0, 7 * sizeof(double)));
}
// 4. 释放显存
void WaveForceEstimator::free_gpu_memory() {
    if (d_amp) cudaFree(d_amp);
    if (d_omega) cudaFree(d_omega);
    if (d_k) cudaFree(d_k);
    if (d_phase) cudaFree(d_phase);
    if (d_theta) cudaFree(d_theta);
    if (d_face_centers_x) cudaFree(d_face_centers_x);
    if (d_face_centers_y) cudaFree(d_face_centers_y);
    if (d_face_centers_z) cudaFree(d_face_centers_z);
    if (d_face_normals_x) cudaFree(d_face_normals_x);
    if (d_face_normals_y) cudaFree(d_face_normals_y);
    if (d_face_normals_z) cudaFree(d_face_normals_z);
    if (d_face_areas) cudaFree(d_face_areas);
    if (d_result) cudaFree(d_result);
}


// 分配内存
void WaveForceEstimator::resize_data() {
    int total_size = wave_N * wave_M;
    if (total_size <= 0) {
        std::cerr << "Error: Invalid wave grid size (" << wave_N << "x" << wave_M << ")" << std::endl;
        return;
    }
    cout<<"wave_N * wave_M = "<<wave_N<<" * "<<wave_M<<" = "<<total_size<<endl;
    amplitudes.resize(total_size);
    omega.resize(total_size);
    delta_omega.resize(total_size);
    k.resize(total_size);
    phases.resize(total_size);
    wave_theta.resize(total_size);
}

// 生成基于参数的唯一文件名
std::string WaveForceEstimator::generate_filename() {
    std::stringstream ss;
    ss << wave_params_root_path << "wave_" 
       << wave_type << "_"
       << "Hs" << std::fixed << std::setprecision(1) << Hs << "_"
       << "T" << std::fixed << std::setprecision(1) << T1 << "_"
       << "N" << wave_N << "_"
       << "M" << wave_M << "_"
       << "s" << s << "_"
       << "th" << std::fixed << std::setprecision(2) << main_theta/PI*180<< "_"
       << "depth" << std::fixed << std::setprecision(2) << depth
       << ".csv";
    return ss.str();
}

// 策略入口
void WaveForceEstimator::check_init_or_load() {
    std::string filename = generate_filename();
    
    if (fs::exists(filename)) {
        std::cout << "[Wave] Loading params from: " << filename << std::endl;
        load_wave_params(filename);
    } else {
        std::cout << "[Wave] Initializing new spectrum..." << std::endl;
        init_wave_spectrum();
        std::cout << "[Wave] Saving params to: " << filename << std::endl;
        save_wave_params(filename);
    }
}

void WaveForceEstimator::convert_data_to_gpu_format(){
    face_centers_x.clear();
    face_centers_y.clear();
    face_centers_z.clear();
    face_normals_x.clear();
    face_normals_y.clear();
    face_normals_z.clear();
    face_areas.clear();
    uuv_total_area=0.0;
    for(auto face : uuv_hull_faces){
        face_centers_x.push_back(face.center.x());
        face_centers_y.push_back(face.center.y());
        face_centers_z.push_back(face.center.z());
        face_normals_x.push_back(face.normal.x());
        face_normals_y.push_back(face.normal.y());
        face_normals_z.push_back(face.normal.z());
        face_areas.push_back(face.area);
        uuv_total_area+=face.area;
    }
}

// 迭代求解波数
double WaveForceEstimator::wave_number(double omega_i, double depth) {
    double k_val = omega_i * omega_i / g; // 深水近似初值
    for (size_t i = 0; i < 5; i++) {
        k_val = omega_i * omega_i / (g * std::tanh(k_val * depth));
    }
    return k_val;
}

// 谱函数 S(omega)
double WaveForceEstimator::S(double omega_i) {
    if (wave_type == "JONSWAP") {
        double gamma = 3.3;
        double sigma = (omega_i <= 5.24 / T1) ? 0.07 : 0.09;
        double arg = (0.191 * omega_i * T1 - 1.0) / (std::sqrt(2.0) * sigma);
        double Y = std::exp(-arg * arg);
        return 155.0 * std::pow(Hs, 2) / std::pow(T1, 4) * std::pow(omega_i, -5.0)
               * std::exp(-944.0 / std::pow(T1, 4) * std::pow(omega_i, -4.0))
               * std::pow(gamma, Y);
    } else if (wave_type == "pierson-Moskowitz") {
        double B = 3.11 / std::pow(Hs, 2);
        double A = 8.1e-3 * std::pow(g, 2);
        return (A / std::pow(omega_i, 5.0)) * std::exp(-B / std::pow(omega_i, 4.0));
    } else {
        std::cerr << "Error: Unknown wave type: " << wave_type << std::endl;
        return 0.0;
    }
}

// 方向函数 D(theta)
double WaveForceEstimator::D(double theta_val) {
    double delta = theta_val - main_theta;
    // 归一化到 [-PI, PI]
    while (delta > PI) delta -= 2 * PI;
    while (delta < -PI) delta += 2 * PI;

    if (delta >= -PI / 2.0 && delta <= PI / 2.0) {
        // 使用方向集中度 s (原代码里是固定2/PI * cos^2，这里加入 s 的支持)
        // D(theta) = C * cos(theta)^s
        // 简单起见保持你原来的 cos^2 逻辑，如果需要 s 生效，公式需调整
        return (2.0 / PI) * std::pow(std::cos(delta), 2); 
    }
    return 0.0;
}

// 初始化谱
void WaveForceEstimator::init_wave_spectrum() {
    resize_data(); // 必须先分配内存！

    double log_min = std::log10(omega_min);
    double log_max = std::log10(omega_max);
    double step_log_val = (wave_N > 1) ? (log_max - log_min) / (wave_N - 1) : 0;
    double delta_theta_val = (2 * PI) / wave_M;

    double prev_omega = std::pow(10.0, log_min); // 用于计算 delta_omega

    for (int i = 0; i < wave_N; i++) {
        double log_val = log_min + i * step_log_val;
        double omega_i = std::pow(10.0, log_val);
        
        // 计算 delta_omega
        double d_omega_i = (i == 0) ? (omega_i - prev_omega) : (omega_i - prev_omega); 
        // 修正：通常 delta_omega 是当前与前一个的差，或者中心差分。
        // 这里为了简单，第一项设为非常小或者取步长
        if(i==0) d_omega_i = std::pow(10.0, log_min + step_log_val) - omega_i;
        
        prev_omega = omega_i; // 更新供下一次使用

        double k_i = wave_number(omega_i,depth);

        for (int j = 0; j < wave_M; j++) {
            int index = idx(i, j);

            double theta_val = 0.0 + j * delta_theta_val;
            double phase_val = getRand();
            
            // 能量守恒: Amp = sqrt(2 * S * D * dw * dtheta)
            double amp_val = std::sqrt(2.0 * S(omega_i) * D(theta_val) * d_omega_i * delta_theta_val);

            amplitudes[index] = amp_val;
            omega[index]      = omega_i;
            delta_omega[index]= d_omega_i;
            k[index]          = k_i;
            phases[index]     = phase_val;
            wave_theta[index] = theta_val;
        }
    }
}

// 保存参数 CSV
void WaveForceEstimator::save_wave_params(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not save wave params to " << filename << std::endl;
        return;
    }

    // Header 1: Scalars
    file << "Hs,T1,wave_N,wave_M,s,main_theta,wave_type,depth,\n";
    file << Hs << "," << T1 << "," << wave_N << "," << wave_M << "," 
         << s << "," << main_theta << "," << wave_type <<","<<depth<< "\n"; 
    
    // Header 2: Arrays
    // 展平存储：每一行代表一个波分量 (i, j)
    file << "index,amplitude,omega,delta_omega,k,phase,theta,\n";
    
    for (int i = 0; i < wave_N; ++i) {
        for (int j = 0; j < wave_M; ++j) {
            int index = idx(i, j);
            file << index << ","
                 << amplitudes[index] << "," 
                 << omega[index] << "," 
                 << delta_omega[index] << "," 
                 << k[index] << "," 
                 << phases[index] << "," 
                 << wave_theta[index] << "\n";
        }
    }
    file.close();
}

// // 加载参数 CSV
// void WaveForceEstimator::load_wave_params(const std::string& filename) {
//     std::ifstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "Error: Could not load wave params from " << filename << std::endl;
//         // 如果加载失败，回退到初始化
//         init_wave_spectrum();
//         return;
//     }

//     std::string line;
//     // 1. 读取 Scalar Header
//     std::getline(file, line); 
//     // 2. 读取 Scalar Values (这里其实只做校验，或者可以跳过，因为文件名已经包含参数)
//     std::getline(file, line); 
    
//     // 3. 读取 Array Header
//     std::getline(file, line); 

//     // 4. 准备内存
//     resize_data();

//     // 5. 读取数据行
//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::string cell;
//         std::vector<std::string> row;
        
//         while (std::getline(ss, cell, ',')) {
//             row.push_back(cell);
//         }

//         if (row.size() < 8) continue;

//         int index = std::stoi(row[0]);

//         if (index < wave_M*wave_M) {
//             amplitudes[index]  = std::stod(row[1]);
//             omega[index]       = std::stod(row[2]);
//             delta_omega[index] = std::stod(row[3]);
//             k[index]           = std::stod(row[4]);
//             phases[index]      = std::stod(row[5]);
//             wave_theta[index]  = std::stod(row[6]);
//         }
//     }
//     file.close();
// }


void WaveForceEstimator::load_wave_params(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not load wave params from " << filename << std::endl;
        init_wave_spectrum();
        return;
    }

    std::string line;
    
    // 1. 读取 Header 1 (标题行) -> 跳过
    std::getline(file, line); 

    // 2. 读取 Scalar Values (参数数值行)
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> scalar_row;
        while (std::getline(ss, cell, ',')) {
            scalar_row.push_back(cell);
        }
        
        // 关键修复：从文件内容更新类成员变量，确保内存分配大小正确
        if (scalar_row.size() >= 7) {
            this->Hs = std::stod(scalar_row[0]);
            this->T1 = std::stod(scalar_row[1]);
            this->wave_N = std::stoi(scalar_row[2]); // 更新 N
            this->wave_M = std::stoi(scalar_row[3]); // 更新 M
            this->s = std::stoi(scalar_row[4]);
            this->main_theta = std::stod(scalar_row[5]);
            // wave_type 是字符串，如果需要也可以更新
        }
    }
    
    // 3. 读取 Array Header (index, amplitude...) -> 跳过
    std::getline(file, line); 

    // 4. 准备内存 (现在 wave_N 和 wave_M 已经是文件中的正确值了)
    resize_data();
    
    // 计算总数据量，防止越界
    int total_waves = this->wave_N * this->wave_M;

    // 5. 读取数据行
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        if (row.size() < 7) continue; // 至少要有7列数据

        int index = std::stoi(row[0]);

        // 关键修复：使用正确的边界判断 (wave_N * wave_M)
        if (index >= 0 && index < total_waves) {
            amplitudes[index]  = std::stod(row[1]);
            omega[index]       = std::stod(row[2]);
            delta_omega[index] = std::stod(row[3]);
            k[index]           = std::stod(row[4]);
            phases[index]      = std::stod(row[5]);
            wave_theta[index]  = std::stod(row[6]);
        }
    }
    file.close();
    
    std::cout << "[Wave] Params loaded successfully. N=" << wave_N << ", M=" << wave_M << std::endl;
}


// // 计算波面高度
// double WaveForceEstimator::compute_wave_elevation(double x, double y, double t) {
//     double eta = 0.0;
//     int total = wave_N * wave_M;
//     // 这里如果为了性能，应该把 vector 展平或者用 OpenMP
//     #pragma omp parallel for reduction(+:eta)
//     for (int n = 0; n < total; ++n) {
//         double phase_val = k[n] * (x * std::cos(wave_theta[n]) + y * std::sin(wave_theta[n]))
//                             - omega[n] * t + phases[n];
//         eta += amplitudes[n] * std::cos(phase_val);
//     }
//     return eta;
// }

// std::vector<double> WaveForceEstimator::compute_wave_force(double t,std::vector<Face>& uuv_hull_faces,
//                                                             double x,double y,double z,
//                                                             double roll,double pitch,double yaw){
std::vector<double> WaveForceEstimator::compute_wave_force(double t,
                                            double x,double y,double z,
                                            double roll,double pitch,double yaw){
    // 这一步非常轻量级，只传指针和几个标量
    // cout<<"start"<<endl;
    int num_faces_int = static_cast<int>(uuv_hull_faces.size());
    int num_waves_int = wave_N * wave_M;
    // 检查指针是否为空 (防御性编程)
    if (d_result == nullptr || d_amp == nullptr) {
        std::cerr << "Error: GPU memory not initialized!" << std::endl;
        return std::vector<double>(7, 0.0);
    }
    CHECK_CUDA_ERROR(cudaMemset(d_result, 0, 7 * sizeof(double)));

    launch_wave_kernel(
        d_amp, d_omega, d_k, d_phase, d_theta, num_waves_int,
        d_face_centers_x, d_face_centers_y, d_face_centers_z,
        d_face_normals_x, d_face_normals_y, d_face_normals_z, d_face_areas, num_faces_int,
        (double)t, (double)x, (double)y, (double)z, (double)roll, (double)pitch, (double)yaw,
        d_result
    );
    // 检查Kernel启动错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    // 等待Kernel执行完成
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<double> res(7);
    CHECK_CUDA_ERROR(cudaMemcpy(res.data(), d_result, 7 * sizeof(double), cudaMemcpyDeviceToHost));

    if (uuv_total_area > 1e-6) {
        res[6] = res[6] / uuv_total_area;
    } else {
        res[6] = 0.0;
    }
    // res[6]=0.4+res[6]*0.6;

    return res;
}