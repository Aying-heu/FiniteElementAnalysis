#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>


#define MAX_WAVES 4096 

__device__ void compute_zyx_rotation_matrix(double roll, double pitch, double yaw, double R[9]) {
    // 计算各角度的正余弦值（使用CUDA设备端数学函数，不可用std::sin/std::cos）
    double cr = cos(roll);
    double sr = sin(roll);
    double cp = cos(pitch);
    double sp = sin(pitch);
    double cy = cos(yaw);
    double sy = sin(yaw);

    // 计算R = R_X(roll) * R_Y(pitch) * R_Z(yaw) （行优先存储）
    R[0] = cy * cp;                     // R11
    R[1] = cy * sp * sr - sy * cr;     // R12
    R[2] = cy * sp * cr + sy * sr;     // R13
    R[3] = sy * cp;                     // R21
    R[4] = sy * sp * sr + cy * cr;     // R22
    R[5] = sy * sp * cr - cy * sr;     // R23
    R[6] = -sp;                         // R31
    R[7] = cp * sr;                     // R32
    R[8] = cp * cr;                     // R33
}

// // ---------------------------
// // CUDA Kernel: 核心计算函数
// // ---------------------------
// __global__ void compute_face_force_kernel(
//     const double* __restrict__ amp,    // 波幅数组
//     const double* __restrict__ omega,  // 频率数组
//     const double* __restrict__ k_arr,  // 波数数组
//     const double* __restrict__ phase,  // 相位数组
//     const double* __restrict__ theta,  // 方向数组
//     int num_waves,                     // 波浪分量总数
    
//     const double* __restrict__ face_centers_x,
//     const double* __restrict__ face_centers_y,
//     const double* __restrict__ face_centers_z,
//     const double* __restrict__ face_normals_x,
//     const double* __restrict__ face_normals_y,
//     const double* __restrict__ face_normals_z,
//     const double* __restrict__ face_areas,         // 面元面积
//     int num_faces,                                 // 面元总数
    
//     double t,                          // 当前时间
//     double x,
//     double y,
//     double z,
//     double roll,
//     double pitch,
//     double yaw,
//     double* out_result_array
// ) {
//     __shared__ double s_data[256 * 7]; 

//     __shared__ double R[9];
//     compute_zyx_rotation_matrix(roll, pitch, yaw, R);
    
//     int tid = threadIdx.x;
//     int idx = blockIdx.x * blockDim.x + tid;
//     double local_force[7] = {0.0};
//     if (idx >= num_faces) return;

//     // 常量定义
//     const double RHO = 1025.0;
//     const double G = 9.81;
//     __syncthreads(); // 必须同步，确保初始化完成

//     double center_x =    R[0]*face_centers_x[idx] 
//                         +R[1]*face_centers_y[idx] 
//                         +R[2]*face_centers_z[idx]
//                         +x;
//     double center_y =    R[3]*face_centers_x[idx] 
//                         +R[4]*face_centers_y[idx] 
//                         +R[5]*face_centers_z[idx]
//                         +y;
//     double center_z =    R[6]*face_centers_x[idx] 
//                         +R[7]*face_centers_y[idx] 
//                         +R[8]*face_centers_z[idx]
//                         +z;
//     double normal_x =    R[0]*face_normals_x[idx] 
//                         +R[1]*face_normals_y[idx] 
//                         +R[2]*face_normals_z[idx];
//     double normal_y =    R[3]*face_normals_x[idx] 
//                         +R[4]*face_normals_y[idx] 
//                         +R[5]*face_normals_z[idx];
//     double normal_z =    R[6]*face_normals_x[idx] 
//                         +R[7]*face_normals_y[idx] 
//                         +R[8]*face_normals_z[idx];
//     double area = face_areas[idx];


//     // 2. 计算波面高度 (eta) 和 动压力
//     double eta = 0.0;
//     double dynamic_pressure = 0.0;

//     #pragma unroll
//     for (int w = 0; w < num_waves; w++) {
//         double arg = k_arr[w] * (center_x * cos(theta[w]) + center_y * sin(theta[w])) - omega[w] * t + phase[w];
//         double s, c;
//         sincos(arg, &s, &c); 

//         eta += amp[w] * c; 

//         double decay = exp(-k_arr[w] * fmax(center_z, 0.0));

//         dynamic_pressure += RHO * G *  amp[w]  * decay * c;
//     }

//     // 3. 判断是否在水下 (NED坐标系，Z向下为正，水面 Z = -eta)
//     // 修正逻辑：如果 center.z > -eta (即在波面之下)
//     if (center_z > -eta) {
//         double static_pressure = RHO * G * center_z;
//         double total_pressure = static_pressure + dynamic_pressure;
//         if (total_pressure < 0) total_pressure = 0.0;

//         // dF = -P * n * Area
//         double fx = -total_pressure * normal_x * area;
//         double fy = -total_pressure * normal_y * area;
//         double fz = -total_pressure * normal_z * area;

//         // 力矩计算 (假设重心在原点，如果不是需传入CG并相减)
//         // M = r x F = center x F
//         double mx = (center_y - y) * fz - (center_z - z) * fy;
//         double my = (center_z - z) * fx - (center_x - x) * fz;
//         double mz = (center_x - x) * fy - (center_y - y) * fx;

//         // 4. 原子累加到全局内存 (这是最简单的做法，虽然有性能损耗但代码量小)
//         // 注意：double类型的atomicAdd在旧GPU (Pascal架构之前) 需要特殊实现
//         local_force[0] = fx; 
//         local_force[1] = fy; 
//         local_force[2] = fz; 
//         local_force[3] = mx; 
//         local_force[4] = my; 
//         local_force[5] = mz; 
//         local_force[6] = area;
//     }
//     for(int k=0; k<7; k++) {
//         s_data[tid * 7 + k] = local_force[k];
//     }

//     // === 5. 块内同步 ===
//     // 等待所有线程把结果写到小黑板上
//     __syncthreads();

//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             for(int k=0; k<7; k++) {
//                 s_data[tid * 7 + k] += s_data[(tid + s) * 7 + k];
//             }
//         }
//         __syncthreads();
//     }
//     if (tid == 0) {
//         for(int k=0; k<7; k++) {
//             atomicAdd(&out_result_array[k], s_data[k]);
//         }
//     }
// }

__global__ void compute_face_force_kernel(
    const double* __restrict__ amp,
    const double* __restrict__ omega,
    const double* __restrict__ k_arr,
    const double* __restrict__ phase,
    const double* __restrict__ theta,
    int num_waves,
    const double* __restrict__ face_centers_x,
    const double* __restrict__ face_centers_y,
    const double* __restrict__ face_centers_z,
    const double* __restrict__ face_normals_x,
    const double* __restrict__ face_normals_y,
    const double* __restrict__ face_normals_z,
    const double* __restrict__ face_areas,
    int num_faces,
    double t, double x, double y, double z,
    double roll, double pitch, double yaw,
    double* out_result_array
) {
    // 1. Shared Memory
    __shared__ double s_data[256 * 7]; 
    __shared__ double R[9];

    // 2. Thread 0 calculates Rotation Matrix for the whole block (Optimization)
    if (threadIdx.x == 0) {
        compute_zyx_rotation_matrix(roll, pitch, yaw, R);
    }
    __syncthreads(); // Wait for R to be ready

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize to 0.0 (Important for padding threads)
    double local_force[7] = {0.0}; 

    // 3. Only calculate physics if idx is valid
    // REMOVED: if (idx >= num_faces) return;  <-- CAUSES DEADLOCK
    if (idx < num_faces) {
        const double RHO = 1025.0;
        const double G = 9.81;

        double center_x = R[0]*face_centers_x[idx] + R[1]*face_centers_y[idx] + R[2]*face_centers_z[idx] + x;
        double center_y = R[3]*face_centers_x[idx] + R[4]*face_centers_y[idx] + R[5]*face_centers_z[idx] + y;
        double center_z = R[6]*face_centers_x[idx] + R[7]*face_centers_y[idx] + R[8]*face_centers_z[idx] + z;

        double eta = 0.0;
        double dynamic_pressure = 0.0;

        #pragma unroll
        for (int w = 0; w < num_waves; w++) {
            if(amp[w]==0 || amp[w]<1e-6)continue;
            double arg = k_arr[w] * (center_x * cos(theta[w]) + center_y * sin(theta[w])) - omega[w] * t + phase[w];
            double s_val, c_val;
            sincos(arg, &s_val, &c_val);
            eta += amp[w] * c_val;
            double decay = exp(-k_arr[w] * fmax(center_z, 0.0));
            dynamic_pressure += RHO * G * amp[w] * decay * c_val;
        }

        if (center_z > -eta) {
            double normal_x = R[0]*face_normals_x[idx] + R[1]*face_normals_y[idx] + R[2]*face_normals_z[idx];
            double normal_y = R[3]*face_normals_x[idx] + R[4]*face_normals_y[idx] + R[5]*face_normals_z[idx];
            double normal_z = R[6]*face_normals_x[idx] + R[7]*face_normals_y[idx] + R[8]*face_normals_z[idx];
            double area = face_areas[idx];

            double total_pressure = (RHO * G * center_z) + dynamic_pressure;
            if (total_pressure < 0) total_pressure = 0.0;

            double fx = -total_pressure * normal_x * area;
            double fy = -total_pressure * normal_y * area;
            double fz = -total_pressure * normal_z * area;

            local_force[0] = fx; 
            local_force[1] = fy; 
            local_force[2] = fz; 
            local_force[3] = (center_y - y) * fz - (center_z - z) * fy;
            local_force[4] = (center_z - z) * fx - (center_x - x) * fz;
            local_force[5] = (center_x - x) * fy - (center_y - y) * fx;
            local_force[6] = area;
        }
    }

    // 4. Load into Shared Memory
    // Even threads with idx >= num_faces participate here (writing 0.0)
    for(int k=0; k<7; k++) {
        s_data[tid * 7 + k] = local_force[k];
    }
    __syncthreads();

    // 5. Parallel Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for(int k=0; k<7; k++) {
                s_data[tid * 7 + k] += s_data[(tid + s) * 7 + k];
            }
        }
        __syncthreads();
    }

    // 6. Write to Global Memory
    if (tid == 0) {
        for(int k=0; k<7; k++) {
            atomicAdd(&out_result_array[k], s_data[k]);
        }
    }
}

// C++ 调用的包装函数
extern "C" void launch_wave_kernel(
    const double* d_amp, 
    const double* d_omega, 
    const double* d_k_arr, 
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
) {
    // 1. 分配 GPU 内存 (cudaMalloc)
    // 2. 拷贝数据 Host -> Device (cudaMemcpy)
    // 3. 计算 Grid 和 Block 大小
    //    int blockSize = 256;
    //    int gridSize = (num_faces + blockSize - 1) / blockSize;
    // 4. 启动 compute_face_force_kernel<<<gridSize, blockSize>>>(...);
    // 5. 拷贝结果回 Host
    // 6. 释放内存

    // 1. 打印调试信息 (都在 Host 端执行)
    // const double* d_amp, 
    // const double* d_omega, 
    // const double* d_k_arr, 
    // const double* d_phase,
    // const double* d_theta,
    // int num_waves,

    // const double* d_face_centers_x,
    // const double* d_face_centers_y,
    // const double* d_face_centers_z,
    // const double* d_face_normals_x,
    // const double* d_face_normals_y,
    // const double* d_face_normals_z,
    // const double* d_face_areas,
    // int num_faces,

    // double t,double x,double y,double z,
    // double roll,double pitch,double yaw,
    // double* d_result

    cudaError_t initErr = cudaFree(0);
    if (initErr != cudaSuccess) {
        printf("CRITICAL: CUDA Context Init Failed: %s\n", cudaGetErrorString(initErr));
        return;
    }

    // printf("--- DEBUG LAUNCH ---\n");
    // printf("num_waves=%d\n",num_waves);
    // printf("num_faces=%d\n",num_faces);
    // printf("t=%f\n",t);
    // printf("x=%f\n",x);
    // printf("y=%f\n",y);
    // printf("z=%f\n",z);
    // printf("roll=%f\n",roll);
    // printf("pitch=%f\n",pitch);
    // printf("pitch=%f\n",pitch);
    // printf("yaw=%f\n",yaw);
    // printf("d_amp addr: %p\n", d_amp);
    // printf("d_omega addr: %p\n", d_omega);
    // printf("d_k_arr addr: %p\n", d_k_arr);
    // printf("d_phase addr: %p\n", d_phase);
    // printf("d_theta addr: %p\n", d_theta);
    // printf("d_face_centers_x addr: %p\n", d_face_centers_x);
    // printf("d_face_centers_y addr: %p\n", d_face_centers_y);
    // printf("d_face_centers_z addr: %p\n", d_face_centers_z);
    // printf("d_face_normals_x addr: %p\n", d_face_normals_x);
    // printf("d_face_normals_y addr: %p\n", d_face_normals_y);
    // printf("d_face_normals_z addr: %p\n", d_face_normals_z);
    // printf("d_result addr: %p\n", d_result);


    if (d_result == nullptr || d_amp == nullptr || d_face_centers_x == nullptr) {
        printf("[CUDA Error] Null pointer passed to launch_wave_kernel!\n");
        return;
    }
    // 1. 清零结果内存 (非常重要，因为Kernel里是 atomicAdd)
    cudaMemset(d_result, 0, 7 * sizeof(double));

    // 2. 详细的空指针检查 (帮助定位 0x10 错误的根源)
    if (!d_amp || !d_k_arr || !d_face_centers_x || !d_result) {
        printf("[CUDA Error] Null pointer detected in launch_wave_kernel!\n");
        printf("d_amp: %p, d_k: %p, d_face_x: %p, d_result: %p\n", d_amp, d_k_arr, d_face_centers_x, d_result);
        return;
    }


    // 2. 计算线程块
    int blockSize = 256;
    int gridSize = (num_faces + blockSize - 1) / blockSize;
    // printf("Grid: %d, Block: %d\n", gridSize, blockSize);

    // 3. 启动内核
    // 注意：输出结果分开指针传进去比较麻烦，建议在显存里开一个长度为7的数组 d_result
    // 修改 Kernel 参数列表以适配 d_result 数组，或者在这里做指针偏移
    compute_face_force_kernel<<<gridSize, blockSize>>>(
        d_amp, d_omega, d_k_arr, d_phase, d_theta, num_waves,
        d_face_centers_x, d_face_centers_y, d_face_centers_z, 
        d_face_normals_x, d_face_normals_y, d_face_normals_z, 
        d_face_areas, num_faces,
        t, x, y, z, roll, pitch, yaw,
        d_result
    );
    // printf("Grid: %d, Block: %d\n", gridSize, blockSize);
    // 5. 【关键】强制同步并检查错误
    // 如果没有这句话，Kernel里的 printf 还没出来程序就崩了
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Failed: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaDeviceSynchronize(); // 等待 GPU 跑完
    if (err != cudaSuccess) {
        printf("Kernel Execution Failed: %s\n", cudaGetErrorString(err));
    } 
    // else {
    //     // printf("Kernel Finished Successfully.\n");
    // }
}