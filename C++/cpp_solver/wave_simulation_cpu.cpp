#include "wave_simulation_cpu.hpp"
#include <iostream>
#include <random>

// 辅助：生成随机数
double getRand(double min = 0.0, double max = 2 * PI) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

// 构造函数
WaveForceEstimator::WaveForceEstimator(double Hs, double T1,
                                       int wave_N, int wave_M, int s,
                                       double main_theta, std::string wave_type)
    : Hs(Hs), T1(T1), wave_N(wave_N), wave_M(wave_M), s(s),
      main_theta(main_theta), wave_type(wave_type) 
{
    // 1. 确保目录存在
    if (!fs::exists(wave_params_root_path)) {
        fs::create_directories(wave_params_root_path);
    }
    
    // 2. 执行加载或初始化策略
    check_init_or_load();
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
       << "th" << std::fixed << std::setprecision(2) << main_theta
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

        double k_i = wave_number(omega_i);

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
    file << "Hs,T1,wave_N,wave_M,s,main_theta,wave_type\n";
    file << Hs << "," << T1 << "," << wave_N << "," << wave_M << "," 
         << s << "," << main_theta << "," << wave_type << "\n"; 
    
    // Header 2: Arrays
    // 展平存储：每一行代表一个波分量 (i, j)
    file << "index,amplitude,omega,delta_omega,k,phase,theta\n";
    
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

// 加载参数 CSV
void WaveForceEstimator::load_wave_params(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not load wave params from " << filename << std::endl;
        // 如果加载失败，回退到初始化
        init_wave_spectrum();
        return;
    }

    std::string line;
    // 1. 读取 Scalar Header
    std::getline(file, line); 
    // 2. 读取 Scalar Values (这里其实只做校验，或者可以跳过，因为文件名已经包含参数)
    std::getline(file, line); 
    
    // 3. 读取 Array Header
    std::getline(file, line); 

    // 4. 准备内存
    resize_data();

    // 5. 读取数据行
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        if (row.size() < 8) continue;

        int index = std::stoi(row[0]);

        if (index < wave_M*wave_M) {
            amplitudes[index]  = std::stod(row[1]);
            omega[index]       = std::stod(row[2]);
            delta_omega[index] = std::stod(row[3]);
            k[index]           = std::stod(row[4]);
            phases[index]      = std::stod(row[5]);
            wave_theta[index]  = std::stod(row[6]);
        }
    }
    file.close();
}

// 计算波面高度
double WaveForceEstimator::compute_wave_elevation(double x, double y, double t) {
    double eta = 0.0;
    int total = wave_N * wave_M;
    // 这里如果为了性能，应该把 vector 展平或者用 OpenMP
    #pragma omp parallel for reduction(+:eta)
    for (int n = 0; n < total; ++n) {
        double phase_val = k[n] * (x * std::cos(wave_theta[n]) + y * std::sin(wave_theta[n]))
                            - omega[n] * t + phases[n];
        eta += amplitudes[n] * std::cos(phase_val);
    }
    return eta;
}

// std::vector<double> WaveForceEstimator::compute_wave_force(double t,std::vector<Face>& uuv_hull_faces,
//                                                             double x,double y,double z,
//                                                             double roll,double pitch,double yaw){
std::vector<double> WaveForceEstimator::compute_wave_force(double t,std::vector<Face>& uuv_hull_faces,
                                            double x,double y,double z,
                                            double roll,double pitch,double yaw){
    // 物理常数
    const double RHO = 1025.0; // 海水密度 kg/m^3
    const double G = 9.81;     // 重力加速度 m/s^2

    // 1. 构建旋转矩阵 R_body_to_world (ZYX 顺序)
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(yaw, Vector3d::UnitZ()) * 
        Eigen::AngleAxisd(pitch, Vector3d::UnitY()) * 
        Eigen::AngleAxisd(roll, Vector3d::UnitX());

    // UUV 重心位置向量
    Vector3d pos_cg(x, y, z);

    // 准备归约变量
    Vector3d total_force = Vector3d::Zero();
    Vector3d total_moment = Vector3d::Zero();

    // 获取面元数量
    int num_faces = uuv_hull_faces.size();

    double total_area{0.0};
    double wet_area{0.0};

    #pragma omp parallel
    {
        // 线程局部累加器 (避免锁竞争)
        Vector3d thread_force = Vector3d::Zero();
        Vector3d thread_moment = Vector3d::Zero();
        double thread_total_area=0.0;
        double thread_wet_area=0.0;

        #pragma omp for schedule(static)
        for (int i = 0; i < num_faces; ++i) {
            const auto& face = uuv_hull_faces[i];
            thread_total_area+=face.area;

            // A. 坐标变换：将面元中心和法向量转到世界坐标系
            Vector3d center_world = R * face.center + pos_cg;
            Vector3d normal_world = R * face.normal;

            // B. 计算波浪参数 (波面高度 eta 和 动压力)
            double eta = 0.0;             // 波面高度
            double dynamic_pressure = 0.0; // 动压力项 (含 Smith 效应)

            // 遍历所有波浪分量 (假设数据已展平为一维，或者使用双重循环)
            // 这里为了物理精确，必须对每个面元遍历波谱
            for( int k_idx = 0;k_idx<wave_M*wave_N;k_idx++){
                double A = amplitudes[k_idx];
                double w = omega[k_idx];
                double k = this->k[k_idx];
                double eps = phases[k_idx];
                double theta = wave_theta[k_idx];
                // 计算相位: k * (x*cos + y*sin) - w*t + phi
                double phase = k * (center_world.x() * std::cos(theta) + 
                                    center_world.y() * std::sin(theta)) 
                                - w * t + eps;
                double cos_val = std::cos(phase);

                // 1. 累加波面高度 (用于判断是否入水)
                eta += A * cos_val;

                // 2. 累加动压力 (Smith Effect: A * e^(-kz) * cos)
                // 注意：NED坐标系下，z向下为正。
                // 动压力衰减因子 e^(-kz)。当 z 为正(深水)时，e指数极小。
                // 为了防止数值爆炸(如果z是很大的负数)，通常限制一下。
                // 仅当面元在平衡位置附近或水下时计算。
                double decay = 1.0;
                if (center_world.z() > -10.0) { // 简单保护，防止飞到天上时溢出
                    decay = std::exp(-k * center_world.z());
                }
                // 伯努利方程动压项近似: rho * g * eta_local * decay
                dynamic_pressure += RHO * G * A * decay * cos_val;
            }

            // C. 浸没判断与受力计算
            // NED坐标系: 水面 Z_surface = -eta
            // 如果 面元中心 Z > -eta，则面元在水下
            if (center_world.z() > -eta) {
                thread_wet_area+=face.area;
                
                // 计算总压力 P_total = P_hydrostatic + P_dynamic
                // 1. 静水压力 P_stat = rho * g * z
                double static_pressure = RHO * G * center_world.z();
                
                // 2. 总压力
                // 注意：如果 z < 0 (在波谷与静水面之间)，static_pressure 是负的，
                // 但加上 dynamic_pressure 后总体应该是正的（物理上压力不为负）。
                // 为了数值鲁棒性，取 max(0, P)
                double total_pressure = static_pressure + dynamic_pressure;
                
                // 修正：完全出水的情况已经在 if (z > -eta) 排除了
                // 但为了防止波浪表面附近的数值误差，可以加个 clamp
                if (total_pressure < 0) total_pressure = 0.0;

                // D. 计算力和力矩
                // dF = -P * n * Area (压力垂直于表面指向内，故取负号)
                Vector3d dF = -total_pressure * normal_world * face.area;
                
                // 力臂 r = P_face - P_cg (在世界坐标系下)
                Vector3d r = center_world - pos_cg;
                Vector3d dM = r.cross(dF);

                // E. 累加
                thread_force += dF;
                thread_moment += dM;
            }
        }

        // 线程汇总
        #pragma omp critical
        {
            total_force += thread_force;
            total_moment += thread_moment;
            total_area+=thread_total_area;
            wet_area+=thread_wet_area;
        }
    }

    // 转换结果为 std::vector
    std::vector<double> result(6);
    result[0] = total_force.x();
    result[1] = total_force.y();
    result[2] = total_force.z();
    result[3] = total_moment.x();
    result[4] = total_moment.y();
    result[5] = total_moment.z();
    result[6] = wet_area/total_area;

    return result;
}