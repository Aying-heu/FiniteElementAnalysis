import math
# import plotly.graph_objects as go
import numpy as np
# from plotly.subplots import make_subplots
from math import pi, cos, sin, sqrt
import time
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from matplotlib.colors import LinearSegmentedColormap

import concurrent.futures
from functools import partial
from threading import Lock
from numba import njit, prange
import multiprocessing

import os
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
from tqdm import tqdm
# 预生成随机相位数组
np.random.seed(42)  # 固定随机种子确保结果可重现
from numba import cuda, float32

TPB = 256 
@cuda.jit
def fast_fsi_kernel(
    # 输入：初始几何信息 (只读，驻留显存)
    init_points,     # [N, 3] 初始未旋转的顶点
    face_ids,        # [N] 每个点所属的面ID
    face_normals,    # [6, 3] 6个面的初始法向量
    
    # 输入：当前运动状态 (每帧更新，仅12个float)
    rot_matrix,      # [3, 3] 旋转矩阵
    trans_vec,       # [3] 平移向量
    g_center,        # [3] 当前重心位置
    
    # 输入：波浪参数
    wave_k, wave_omega, wave_theta, wave_amp, wave_phi, 
    sim_time,        # 当前时间 t
    
    # 输入：物理参数
    rho, g, d_uuv,
    
    # 输出：力和力矩
    out_force,       # [3]
    out_moment       # [3]
):
    # 1. 声明共享内存用于块归约 (Block Reduction)
    # shared_mem 布局: [TPB, 6] -> 前3个存Fx,Fy,Fz，后3个存Mx,My,Mz
    s_data = cuda.shared.array((TPB, 6), dtype=float32)
    
    # 线程索引
    tid = cuda.threadIdx.x
    gid = cuda.grid(1)
    
    # 初始化共享内存
    s_data[tid, 0] = 0.0
    s_data[tid, 1] = 0.0
    s_data[tid, 2] = 0.0
    s_data[tid, 3] = 0.0
    s_data[tid, 4] = 0.0
    s_data[tid, 5] = 0.0
    
    # 2. 计算几何变换和受力 (每个线程处理一个点)
    if gid < init_points.shape[0]:
        # --- A. 几何变换 (Geometry Transformation) ---
        # 读取初始点
        p0_x = init_points[gid, 0]
        p0_y = init_points[gid, 1]
        p0_z = init_points[gid, 2]
        
        # 旋转 (R * p)
        cur_x = rot_matrix[0,0]*p0_x + rot_matrix[0,1]*p0_y + rot_matrix[0,2]*p0_z
        cur_y = rot_matrix[1,0]*p0_x + rot_matrix[1,1]*p0_y + rot_matrix[1,2]*p0_z
        cur_z = rot_matrix[2,0]*p0_x + rot_matrix[2,1]*p0_y + rot_matrix[2,2]*p0_z
        
        # 平移 (+ T)
        cur_x += trans_vec[0]
        cur_y += trans_vec[1]
        cur_z += trans_vec[2]
        
        # --- B. 计算波高 (Wave Elevation) ---
        # 直接在此处计算，不读取 global memory
        eta = 0.0
        # 循环展开或直接遍历波浪分量
        for i in range(wave_k.shape[0]): # wave_N
            k_val = wave_k[i]
            w_val = wave_omega[i]
            for j in range(wave_theta.shape[0]): # wave_M
                theta_val = wave_theta[j]
                
                # 相位计算
                phase = (k_val * (cur_x * math.cos(theta_val) + 
                                  cur_y * math.sin(theta_val))
                         - w_val * sim_time 
                         + wave_phi[i, j])
                
                eta += wave_amp[i, j] * math.cos(phase)
        
        # --- C. 计算压力和力矩 (Pressure & Force) ---
        if eta < cur_z: 
            # 压力深度 = 当前深度 - 波面位置
            depth = cur_z - eta
            pressure = rho * g * depth * d_uuv * d_uuv
            
            fid = face_ids[gid]
            n0_x = face_normals[fid, 0]
            n0_y = face_normals[fid, 1]
            n0_z = face_normals[fid, 2]
            
            # 旋转法向量
            nx = rot_matrix[0,0]*n0_x + rot_matrix[0,1]*n0_y + rot_matrix[0,2]*n0_z
            ny = rot_matrix[1,0]*n0_x + rot_matrix[1,1]*n0_y + rot_matrix[1,2]*n0_z
            nz = rot_matrix[2,0]*n0_x + rot_matrix[2,1]*n0_y + rot_matrix[2,2]*n0_z
            
            # --- 关键点：压力方向 ---
            # 在 Z轴向下坐标系中，计算受力方向需要特别小心
            # 压力始终垂直于表面指向物体内部。
            # 如果你的法向量(nx,ny,nz)是指向物体【内部】的：
            # Force = Pressure * Normal
            # 如果你的法向量是指向物体【外部】的：
            # Force = -Pressure * Normal
            
            # 之前你说 directions = center - face，这是指向内部的。
            # 所以这里保持正号
            fx = pressure * nx
            fy = pressure * ny
            fz = pressure * nz
            
            # 力矩计算保持不变 (r x F)
            rx = cur_x - g_center[0]
            ry = cur_y - g_center[1]
            rz = cur_z - g_center[2]
            
            mx = ry * fz - rz * fy
            my = rz * fx - rx * fz
            mz = rx * fy - ry * fx
            
            s_data[tid, 0] = fx
            s_data[tid, 1] = fy
            s_data[tid, 2] = fz
            s_data[tid, 3] = mx
            s_data[tid, 4] = my
            s_data[tid, 5] = mz

    cuda.syncthreads()

    # --- D. 块内归约 (Parallel Reduction in Shared Memory) ---
    # 使用二叉树归约法将 s_data[tid] 累加到 s_data[0]
    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            s_data[tid, 0] += s_data[tid + stride, 0]
            s_data[tid, 1] += s_data[tid + stride, 1]
            s_data[tid, 2] += s_data[tid + stride, 2]
            s_data[tid, 3] += s_data[tid + stride, 3]
            s_data[tid, 4] += s_data[tid + stride, 4]
            s_data[tid, 5] += s_data[tid + stride, 5]
        cuda.syncthreads()
        stride //= 2

    # --- E. 全局原子加 (只由每个Block的线程0执行) ---
    if tid == 0:
        cuda.atomic.add(out_force, 0, s_data[0, 0])
        cuda.atomic.add(out_force, 1, s_data[0, 1])
        cuda.atomic.add(out_force, 2, s_data[0, 2])
        
        cuda.atomic.add(out_moment, 0, s_data[0, 3])
        cuda.atomic.add(out_moment, 1, s_data[0, 4])
        cuda.atomic.add(out_moment, 2, s_data[0, 5])

@cuda.jit
def simulation_kernel(
    # --- 初始状态 ---
    init_state,      # [12] (x,y,z, phi,theta,psi, u,v,w, p,q,r)
    mass_matrix_inv, # [6, 6] 质量矩阵的逆 (预先在CPU算好传进来!)
    damping_lin,     # [6] 线性阻尼系数 (对角线简化版)
    damping_quad,    # [6] 二次阻尼系数
    
    # --- 几何 ---
    init_points,     # [N, 3]
    face_ids,        # [N]
    face_normals,    # [6, 3]
    g_center_local,  # [3] 重心在局部坐标系的位置
    
    # --- 波浪 & 环境 ---
    wave_k, wave_omega, wave_theta, wave_amp, wave_phi,
    current_force,   # [3] (u, v, w 方向的流力)
    tau_control,     # [6] 控制力
    
    # --- 仿真参数 ---
    dt,              # 时间步长
    total_steps,     # 总步数
    rho, g, d_uuv,
    
    # --- 输出 ---
    out_trajectory   # [total_steps, 13] (t, x, y, z, phi, theta, psi...)
):
    # 共享内存：存储当前步的 力 和 力矩
    # 0-2: Force, 3-5: Moment
    s_force = cuda.shared.array(6, dtype=float32)
    # 共享内存：存储当前步的 状态 (12个变量)
    s_state = cuda.shared.array(12, dtype=float32)
    # 共享内存：临时变量用于归约
    s_reduce = cuda.shared.array((TPB, 6), dtype=float32)

    tid = cuda.threadIdx.x
    gid = cuda.grid(1)
    
    # --- 1. 初始化状态 (由线程0负责加载到共享内存) ---
    if tid < 12:
        s_state[tid] = init_state[tid]
    
    cuda.syncthreads()

    # ====== 时间循环开始 ======
    for step in range(total_steps):
        current_time = step * dt
        
        # A. 清空受力累加器
        if tid < 6:
            s_force[tid] = 0.0
        
        # 初始化归约数组
        s_reduce[tid, 0] = 0.0
        s_reduce[tid, 1] = 0.0
        s_reduce[tid, 2] = 0.0
        s_reduce[tid, 3] = 0.0
        s_reduce[tid, 4] = 0.0
        s_reduce[tid, 5] = 0.0
        
        cuda.syncthreads() # 确保状态和力都被初始化/清空

        # B. 每个线程计算一部分点的受力 (并行计算 Froude-Krylov)
        # 读取当前状态 (所有线程都读共享内存，极快)
        x = s_state[0]; y = s_state[1]; z = s_state[2]
        phi = s_state[3]; theta = s_state[4]; psi = s_state[5]
        
        # 预计算旋转矩阵 (每个线程都要用，或者由线程0算好放在shared里)
        # 这里为了并行度，每个线程算自己的坐标变换
        # 构建旋转矩阵 R (ZYX顺序 或 XYZ顺序，需与Python一致)
        cph, sph = math.cos(phi), math.sin(phi)
        cth, sth = math.cos(theta), math.sin(theta)
        cps, sps = math.cos(psi), math.sin(psi)
        
        # R = Rz * Ry * Rx (示例)
        r00 = cps*cth; r01 = cps*sth*sph - sps*cph; r02 = cps*sth*cph + sps*sph
        r10 = sps*cth; r11 = sps*sth*sph + cps*cph; r12 = sps*sth*cph - cps*sph
        r20 = -sth;    r21 = cth*sph;               r22 = cth*cph

        # 遍历分配给该线程的点
        # 假设点数 N > TPB，使用 stride 循环
        for i in range(tid, init_points.shape[0], TPB):
            # 1. 坐标变换 Local -> Global
            p0x, p0y, p0z = init_points[i, 0], init_points[i, 1], init_points[i, 2]
            
            # 旋转
            cur_x = r00*p0x + r01*p0y + r02*p0z + x
            cur_y = r10*p0x + r11*p0y + r12*p0z + y
            cur_z = r20*p0x + r21*p0y + r22*p0z + z
            
            # 2. 计算波高 (完全内联)
            eta = 0.0
            for w_idx in range(wave_k.shape[0]):
                k = wave_k[w_idx]
                w = wave_omega[w_idx]
                for d_idx in range(wave_theta.shape[0]):
                    th = wave_theta[d_idx]
                    phase = k * (cur_x * math.cos(th) + cur_y * math.sin(th)) - w * current_time + wave_phi[w_idx, d_idx]
                    eta += wave_amp[w_idx, d_idx] * math.cos(phase)
            
            # 3. 计算压力 (Z轴向下逻辑: cur_z > eta 为入水)
            if cur_z > eta: # Z轴向下，数值大为深
                depth = cur_z - eta
                pressure = rho * g * depth * d_uuv * d_uuv
                
                # 法向量变换
                fid = face_ids[i]
                nx0, ny0, nz0 = face_normals[fid, 0], face_normals[fid, 1], face_normals[fid, 2]
                
                nx = r00*nx0 + r01*ny0 + r02*nz0
                ny = r10*nx0 + r11*ny0 + r12*nz0
                nz = r20*nx0 + r21*ny0 + r22*nz0
                
                # 累加力 (假设法向量向内)
                fx = -pressure * nx
                fy = -pressure * ny
                fz = -pressure * nz
                
                # 累加力矩 (相对于重心)
                # 重心在世界坐标的位置
                gx = r00*g_center_local[0] + r01*g_center_local[1] + r02*g_center_local[2] + x
                gy = r10*g_center_local[0] + r11*g_center_local[1] + r12*g_center_local[2] + y
                gz = r20*g_center_local[0] + r21*g_center_local[1] + r22*g_center_local[2] + z
                
                rx = cur_x - gx
                ry = cur_y - gy
                rz = cur_z - gz
                
                s_reduce[tid, 0] += fx
                s_reduce[tid, 1] += fy
                s_reduce[tid, 2] += fz
                s_reduce[tid, 3] += ry*fz - rz*fy
                s_reduce[tid, 4] += rz*fx - rx*fz
                s_reduce[tid, 5] += rx*fy - ry*fx

        cuda.syncthreads()

        # C. 块内归约 (Sum Reduction)
        stride = TPB // 2
        while stride > 0:
            if tid < stride:
                for k in range(6):
                    s_reduce[tid, k] += s_reduce[tid + stride, k]
            cuda.syncthreads()
            stride //= 2
            
        # D. 物理积分 (由线程0单线程执行)
        if tid == 0:
            # 1. 获取总的 Froude-Krylov 力
            fk_force_x = s_reduce[0, 0]
            fk_force_y = s_reduce[0, 1]
            fk_force_z = s_reduce[0, 2]
            fk_mom_x   = s_reduce[0, 3]
            fk_mom_y   = s_reduce[0, 4]
            fk_mom_z   = s_reduce[0, 5]
            
            # 2. 转回 Body Frame (因为阻尼和质量矩阵通常在Body系定义)
            # R_inv = R.T
            # F_body = R.T * F_world
            f_body_x = r00*fk_force_x + r10*fk_force_y + r20*fk_force_z
            f_body_y = r01*fk_force_x + r11*fk_force_y + r21*fk_force_z
            f_body_z = r02*fk_force_x + r12*fk_force_y + r22*fk_force_z
            
            m_body_x = r00*fk_mom_x + r10*fk_mom_y + r20*fk_mom_z
            m_body_y = r01*fk_mom_x + r11*fk_mom_y + r21*fk_mom_z
            m_body_z = r02*fk_mom_x + r12*fk_mom_y + r22*fk_mom_z
            
            # 3. 添加重力 (Body Frame)
            # Z轴向下，重力在世界系是 [0, 0, W]
            W = mass_matrix_inv[0,0] * 1.0 * g # 粗略获取质量，这里需要外部传入mass更好
            # 严谨做法：应传入 m，为了简化假设 M_inv[0,0] = 1/m
            m = 1.0 / mass_matrix_inv[0,0]
            W = m * g
            
            # 重力转换到 Body 系: R.T * [0,0,W]
            fg_body_x = r20 * W
            fg_body_y = r21 * W
            fg_body_z = r22 * W
            
            # 4. 添加阻尼 (Linear + Quadratic) & 控制力 & 恒定流
            u, v, w = s_state[6], s_state[7], s_state[8]
            p, q, r = s_state[9], s_state[10], s_state[11]
            
            # 总力 (Body Frame)
            # F_total = F_FK + F_Gravity + F_Current + F_Control - Damping
            # 注意：这里做极大简化，完整公式需矩阵运算
            
            # 示例：仅X轴方程
            # Fx_total = f_body_x + fg_body_x + tau_control[0] + current_force[0] ...
            #            - (damping_lin[0]*u + damping_quad[0]*abs(u)*u)
            
            forces = cuda.local.array(6, dtype=float32)
            velocities = cuda.local.array(6, dtype=float32)
            velocities[0]=u; velocities[1]=v; velocities[2]=w
            velocities[3]=p; velocities[4]=q; velocities[5]=r
            
            # 组装力向量
            forces[0] = f_body_x + fg_body_x + tau_control[0]
            forces[1] = f_body_y + fg_body_y + tau_control[1]
            forces[2] = f_body_z + fg_body_z + tau_control[2]
            forces[3] = m_body_x + tau_control[3] # 忽略重力对重心力矩
            forces[4] = m_body_y + tau_control[4]
            forces[5] = m_body_z + tau_control[5]
            
            # 减去阻尼和科里奥利 (简化)
            for k in range(6):
                forces[k] -= (damping_lin[k] * velocities[k] + damping_quad[k] * abs(velocities[k]) * velocities[k])
                
            # 5. 求解加速度 (acc = M_inv * forces)
            acc = cuda.local.array(6, dtype=float32)
            for r_i in range(6):
                acc_val = 0.0
                for c_i in range(6):
                    acc_val += mass_matrix_inv[r_i, c_i] * forces[c_i]
                acc[r_i] = acc_val
            
            # 6. 积分 (Euler) -> 更新 s_state
            # 更新速度
            for k in range(6):
                s_state[6+k] += acc[k] * dt
                
            # 更新位置 (需要 J(eta) 变换)
            # d_pos = R * vel (简化版，忽略角速度的变换矩阵差异)
            # 实际上 d_eta = J * nu
            # 这里简单处理位置：
            nu_u, nu_v, nu_w = s_state[6], s_state[7], s_state[8]
            dx = r00*nu_u + r01*nu_v + r02*nu_w
            dy = r10*nu_u + r11*nu_v + r12*nu_w
            dz = r20*nu_u + r21*nu_v + r22*nu_w
            
            s_state[0] += dx * dt
            s_state[1] += dy * dt
            s_state[2] += dz * dt
            
            # 简单处理角度 (小角度假设，大角度需要四元数或完整J矩阵)
            # d_euler = J2 * omega
            nu_p, nu_q, nu_r = s_state[9], s_state[10], s_state[11]
            # 简单映射，忽略万向节锁修正
            s_state[3] += (nu_p + nu_q * math.sin(phi)*math.tan(theta) + nu_r * math.cos(phi)*math.tan(theta)) * dt
            s_state[4] += (nu_q * math.cos(phi) - nu_r * math.sin(phi)) * dt
            s_state[5] += (nu_q * math.sin(phi)/math.cos(theta) + nu_r * math.cos(phi)/math.cos(theta)) * dt

            # 7. 写入轨迹输出 (Global Memory)
            if step < total_steps:
                out_trajectory[step, 0] = current_time
                for k in range(12):
                    out_trajectory[step, k+1] = s_state[k]

        cuda.syncthreads() # 等待线程0完成更新，所有线程进入下一步

@njit(parallel=True, fastmath=True)
def compute_directions_numba(vertices, faces, g_center):
    n_faces = len(faces)
    # 预分配结果数组（避免append的动态扩容）
    directions = np.empty((n_faces, 3), dtype=np.float64)
    
    # 并行循环遍历所有面（prange=并行range）
    for i in prange(n_faces):
        face = faces[i]
        # 计算面中心（手动求和替代np.mean，减少NumPy调用开销）
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        face_center = (v0 + v1 + v2) / 3.0
        
        # 计算方向向量并归一化
        dir_vec = g_center - face_center
        norm = np.linalg.norm(dir_vec)
        directions[i] = dir_vec / norm
    
    return directions
@cuda.jit
def wave_elevation_kernel(x, y, t, k, omega, wave_theta, amplitudes, phi_ij, 
                        g, eta_out):
    """
    CUDA核函数：并行计算波高
    每个线程计算一个点的波高
    """
    # 线程索引
    point_idx = cuda.grid(1)
    
    if point_idx < x.shape[0]:
        total_eta = 0.0
        
        # 获取该点的坐标
        x_point = x[point_idx]
        y_point = y[point_idx]
        
        # 遍历所有频率和方向
        for i in range(k.shape[0]):
            # 波数
            k_i = k[i]
            # 角频率
            omega_i = omega[i]
            
            # 计算波速
            c = math.sqrt(g / k_i)
            
            for j in range(wave_theta.shape[0]):
                # 方向角
                theta_j = wave_theta[j]
                
                # 计算相位
                phase = (k_i * (x_point * math.cos(theta_j) + 
                            y_point * math.sin(theta_j))
                        - omega_i * t 
                        + phi_ij[i, j])
                
                # 累加振幅贡献
                total_eta += amplitudes[i, j] * math.cos(phase)
        
        # 存储结果
        eta_out[point_idx] = total_eta






def compute_wave_frame_standalone(t, X, Y, wave_N, wave_M, omega, k, wave_theta, phi_ij, amplitudes):
    """
    不依赖 self 的独立波浪计算函数，专门用于多进程
    """
    # 这里复制原本 wave_elevation_vectorized 的逻辑
    # 注意：原本代码中 propagation 项是被注释掉的，这里保持一致
    phases = np.zeros((*X.shape, wave_N, wave_M))
    
    # 为了性能，尽量减少循环，但保持你原本的逻辑结构
    for i in range(wave_N):
        omega_i = omega[i]
        k_i = k[i]
        for j in range(wave_M):
            theta_j = wave_theta[j]
            phi_ij_val = phi_ij[i, j]
            
            # 计算相位
            phases[..., i, j] = (
                k_i * (X * np.cos(theta_j) + Y * np.sin(theta_j))
                - omega_i * t 
                + phi_ij_val
            )
            
    return np.sum(amplitudes * np.cos(phases), axis=(-1, -2))



class UnderwaterVehicle:
    def __init__(self,m,
                dt, t_max,
                length, width, height, g, rho, A_wp, nabla, GM_L, GM_T,
                xg, yg, zg, xb, yb, zb,
                Ix, Iy, Iz,
                X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot,
                X_u, Y_v, Z_w, K_p, M_q, N_r,
                X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr,
                
                d_uuv,Hs,T1,wave_N,wave_M,s,wave_type,main_theta,current_force_mag,current_angle,
                omega_min, omega_max, x_range, y_range, dx, dy,global_x_range,global_y_range,global_dx,global_dy,
                off_screen, cube_center, moviepath):
        
        self.length=length
        self.width=width
        self.height=height

        self.m = m
        self.u, self.v, self.w, self.p, self.q, self.r = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.x, self.y, self.z, self.phi, self.theta, self.psi = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.Tau_X, self.Tau_Y, self.Tau_Z, self.Tau_K, self.Tau_M, self.Tau_N = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.V = np.array([self.u, self.v, self.w, self.p, self.q, self.r])
        self.Eta = np.array([self.x, self.y, self.z, self.phi, self.theta, self.psi])
        self.Tau = np.array([self.Tau_X, self.Tau_Y, self.Tau_Z, self.Tau_K, self.Tau_M, self.Tau_N])
        self.V_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Eta_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 重力和浮力
        self.g = g
        self.rho=rho
        self.A_wp=A_wp
        self.nabla=nabla
        self.GM_L=GM_L
        self.GM_T=GM_T

        # 重心和浮心位置
        self.xg, self.yg, self.zg = xg, yg, zg
        self.xb, self.yb, self.zb = xb, yb, zb

        # 时间步长和总时间
        self.dt = dt
        self.t_max = t_max
        self.d_uuv=d_uuv

        # 质量矩阵
        self.Ix=Ix
        self.Iy=Iy
        self.Iz=Iz

        self.M_RB = np.array([
            [self.m, 0, 0, 0, 0, 0],
            [0, self.m, 0, 0, 0, 0],
            [0, 0, self.m, 0, 0, 0],
            [0, 0, 0, self.Ix, 0, 0],
            [0, 0, 0, 0, self.Iy, 0],
            [0, 0, 0, 0, 0, self.Iz]
        ])
        self.X_u_dot=X_u_dot
        self.Y_v_dot=Y_v_dot
        self.Z_w_dot=Z_w_dot
        self.K_p_dot=K_p_dot
        self.M_q_dot=M_q_dot
        self.N_r_dot=N_r_dot
        self.M_A = np.array([
            [self.X_u_dot, 0, 0, 0, 0, 0],
            [0, self.Y_v_dot, 0, 0, 0, 0],
            [0, 0, self.Z_w_dot, 0, 0, 0],
            [0, 0, 0, self.K_p_dot, 0, 0],
            [0, 0, 0, 0, self.M_q_dot, 0],
            [0, 0, 0, 0, 0, self.N_r_dot]
        ])
        self.M = self.M_RB + self.M_A

        # 科里奥利矩阵
        self.X_u=X_u
        self.Y_v=Y_v
        self.Z_w=Z_w
        self.K_p=K_p
        self.M_q=M_q
        self.N_r=N_r
        self.X_u_absu=X_u_absu
        self.Y_v_absv=Y_v_absv
        self.Z_w_absw=Z_w_absw
        self.K_p_absp=K_p_absp
        self.M_q_absq=M_q_absq
        self.N_r_absr=N_r_absr

        # 轨迹记录
        self.position_x = []
        self.position_y = []
        self.position_z = []
        self.position_phi_rad = []
        self.position_theta_rad = []
        self.position_psi_rad = []
        self.position_phi_degree = []
        self.position_theta_degree = []
        self.position_psi_degree = []


        self.sim_time = t_max
        self.fps = 1/dt
        self.total_frames = self.fps * self.sim_time

        # 生成波浪谱相关参数
        self.Hs = Hs      # 有效波高 (m)
        self.T1=T1

        self.wave_N = wave_N        # 频率离散点数
        self.wave_M = wave_M        # 方向离散点数
        self.main_theta=main_theta
        self.current_force_mag=current_force_mag
        self.current_angle=current_angle
        self.F_current_x = self.current_force_mag * np.cos(self.current_angle)
        self.F_current_y = self.current_force_mag * np.sin(self.current_angle)

        # 生成频率范围 (避免 ω=0)
        self.omega_min = omega_min
        self.omega_max = omega_max
        # self.omega = np.linspace(self.omega_min, self.omega_max, self.wave_N)
        # 生成对数间隔的频率数组
        self.omega = np.logspace(
            np.log10(self.omega_min), 
            np.log10(self.omega_max), 
            self.wave_N
        )
        # self.delta_omega =self.omega[1] - self.omega[0]
        self.delta_omega = np.diff(self.omega)
        self.delta_omega = np.append(self.delta_omega, self.delta_omega[-1])

        self.k = np.array([self.wave_number(omega_i) for omega_i in self.omega])
        self.s = s  # 方向集中度 (原2)

        # 生成方向范围 (0~2π)
        self.wave_theta = np.linspace(0, 2*pi, self.wave_M, endpoint=False)
        self.delta_theta = self.wave_theta[1] - self.wave_theta[0]

        self.phi_ij = np.random.uniform(0, 2*pi, (self.wave_N, self.wave_M))

        # 预计算波数
        self.k = self.omega**2 / self.g  # 波数 k = ω²/g (深水假设)

        # 预计算振幅
        self.wave_type=wave_type
        self.amplitudes = np.zeros((self.wave_N, self.wave_M))
        for i in range(self.wave_N):
            for j in range(self.wave_M):
                self.amplitudes[i, j] = sqrt(2 * self.S(self.omega[i],self.wave_type) * self.D(self.wave_theta[j]) * self.delta_omega[i] * self.delta_theta)
        
        # 方块参数
        self.cube_length = self.length
        self.cube_width = self.width
        self.cube_height = self.height
        self.cube_center = cube_center  # 初始中心位置
        # 创建方块
        self.initial_cube = pv.Cube(center=self.cube_center, 
                            x_length=self.cube_length,
                            y_length=self.cube_width,
                            z_length=self.cube_height)
        self.cube = pv.Cube(center=self.cube_center, 
                            x_length=self.cube_length, 
                            y_length=self.cube_width, 
                            z_length=self.cube_height)
    
        self.wave_cache_dir = "wave_cache"
        self.ensure_cache_dir_exists()

        self.local_wave_heights_cache = {}
        self.global_wave_heights_cache = {}

        # 2. 预处理波浪参数
        self.d_wave_k = cuda.to_device(self.k.astype(np.float32))
        self.d_wave_omega = cuda.to_device(self.omega.astype(np.float32))
        self.d_wave_theta = cuda.to_device(self.wave_theta.astype(np.float32))
        self.d_wave_amp = cuda.to_device(self.amplitudes.astype(np.float32))
        self.d_wave_phi = cuda.to_device(self.phi_ij.astype(np.float32))

        # 3. 预分配输出显存
        self.d_force = cuda.device_array(3, dtype=np.float32)
        self.d_moment = cuda.device_array(3, dtype=np.float32)

        # 定义六个面的顶点索引
        self.faces = [
            [0, 1, 2, 3],  # x-
            [0, 1, 7, 4],  # y-
            [0, 3, 5, 4],  # z-
            [4, 5, 6, 7],  # x+
            [2, 3, 5, 6],  # y+
            [1, 2, 6, 7]   # z+
        ]
        self.face_np=np.ascontiguousarray(self.faces, dtype=np.int64)

        self.unit_vertices = self.cube.points
        all_points=[]
        face_ids=[]
        for i, face in enumerate(self.faces):
            # 计算面的基向量
            v1 = self.unit_vertices[face[1]] - self.unit_vertices[face[0]]
            v2 = self.unit_vertices[face[3]] - self.unit_vertices[face[0]]
            v1_len = np.linalg.norm(v1)
            v2_len = np.linalg.norm(v2)
            
            # 计算采样点数量
            num_u = int(v1_len / self.d_uuv)
            num_v = int(v2_len / self.d_uuv)
            
            # 生成采样点坐标
            u_coords = np.linspace(0.5, v1_len - 0.5, num_u) / v1_len
            v_coords = np.linspace(0.5, v2_len - 0.5, num_v) / v2_len
            
            for u in u_coords:
                for v in v_coords:
                    point = self.unit_vertices[face[0]] + u*v1 + v*v2
                    all_points.append(point)
                    face_ids.append(i)
        # 转换为数组
        self.all_points_init = np.array(all_points)
        self.face_ids_init = np.array(face_ids)
        
        # 1. 预处理初始几何 (相对于局部原点)
        # 确保 all_points_init 的中心是 (0,0,0)，或者你知道它相对于重心的位置
        self.d_points_init = cuda.to_device(self.all_points_init.astype(np.float32))
        self.d_face_ids = cuda.to_device(self.face_ids_init.astype(np.int32))

        # 计算初始面法向量 (只做一次)
        # 假设 self.face_normals 是 (6, 3) 数组
        normals = np.array([
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [1, 0, 0],  [0, 1, 0],  [0, 0, 1]
        ], dtype=np.float32) # 简化示例，根据你的faces顺序对应
        self.d_normals = cuda.to_device(normals)

        # 2. 预处理波浪参数
        self.d_wave_k = cuda.to_device(self.k.astype(np.float32))
        self.d_wave_omega = cuda.to_device(self.omega.astype(np.float32))
        self.d_wave_theta = cuda.to_device(self.wave_theta.astype(np.float32))
        self.d_wave_amp = cuda.to_device(self.amplitudes.astype(np.float32))
        self.d_wave_phi = cuda.to_device(self.phi_ij.astype(np.float32))

        # 3. 预分配输出显存
        self.d_force = cuda.device_array(3, dtype=np.float32)
        self.d_moment = cuda.device_array(3, dtype=np.float32)

        # 预分配运动状态显存
        self.d_rot_matrix = cuda.device_array((3, 3), dtype=np.float32)
        self.d_trans_vec = cuda.device_array(3, dtype=np.float32)
        self.d_g_center = cuda.device_array(3, dtype=np.float32)

        # 计算 Grid 大小
        self.threads_per_block = 256
        self.blocks_per_grid = (self.all_points_init.shape[0] + 255) // 256

        # 计算 Grid 大小
        self.threads_per_block = 256
        self.blocks_per_grid = (self.all_points_init.shape[0] + 255) // 256

        # 创建分屏plotter (1行2列)
        self.off_screen=off_screen
        self.plotter = pv.Plotter(shape=(1, 2), off_screen=off_screen)
        self.plotter.enable_depth_peeling()  # 提高深度精度

        # # === 左侧子图 ===
        # self.plotter.subplot(0, 0)
        # self.plot_Z = self.local_wave_heights_cache[(0)]  # 这里 t 可以指定
        # self.points = np.column_stack((self.plot_X.ravel(), self.plot_Y.ravel(), self.plot_Z.ravel()))
        # self.point_cloud = pv.PolyData(self.points)
        # local_min = min(np.min(w[:,2]) for w in self.local_wave_heights_cache.values())
        # local_max = max(np.max(w[:,2]) for w in self.local_wave_heights_cache.values())

        # self.actor = self.plotter.add_mesh(
        #     self.point_cloud, 
        #     scalars=self.points[:, 2],         # 用z值着色
        #     cmap='viridis_r',               # 选择色带（如'viridis', 'jet', 'coolwarm'等）
        #     point_size=5,
        #     render_points_as_spheres=True,
        #     opacity=0.3,
        #     clim=(local_min, local_max)  # 固定颜色范围
        # )
        # self.plotter.add_axes(line_width=3, labels_off=False)


        # self.plotter.subplot(0, 1)
        # self.global_plot_Z = self.global_wave_heights_cache[(0)]
        # self.global_points = np.column_stack((self.global_plot_X.ravel(), self.global_plot_Y.ravel(), self.global_plot_Z.ravel()))
        # self.global_point_cloud = pv.PolyData(self.global_points)
        # global_min = min(np.min(w) for w in self.global_wave_heights_cache.values())
        # global_max = max(np.max(w) for w in self.global_wave_heights_cache.values())

        # self.actor = self.plotter.add_mesh(
        #     self.global_point_cloud, 
        #     scalars=self.global_points[:, 2],
        #     cmap='viridis_r',
        #     point_size=5,  # 点更小以提高性能
        #     render_points_as_spheres=True,
        #     opacity=0.3,
        #     clim=(global_min, global_max)  # 固定颜色范围
        # )
        # self.plotter.add_axes(line_width=3, labels_off=False)

        
        
        
        self.plotter.subplot(0, 0)
        self.cube_actor = self.plotter.add_mesh(self.cube, color='blue', opacity=1)

        self.plotter.subplot(0, 1)
        self.global_cube_actor = self.plotter.add_mesh(self.cube, color='red', opacity=1)
        self.global_cube_actor = self.plotter.add_mesh(
            self.cube, 
            color='crimson',  # 亮红色比普通red更鲜艳
            opacity=0.9,      # 降低透明度，增强存在感
            smooth_shading=True  # 平滑着色增加立体感
        )
        self.global_cube_wire = self.plotter.add_mesh(
            self.cube,
            color='black',    # 黑色边框与红色主体形成强对比
            style='wireframe',
            line_width=5,     # 加粗边框
            render_lines_as_tubes=True  # 线条渲染为管状，更醒目
        )
        self.moviepath=moviepath
        self.plotter.open_movie(moviepath)

        

        self.max_workers = multiprocessing.cpu_count()

    # def prepare_visualization_data(self):
    #     """
    #     [新方法] 在仿真结束后调用。
    #     1. 根据轨迹动态计算网格范围。
    #     2. 计算波浪数据。
    #     3. 初始化可视化窗口。
    #     """
    #     print("正在准备可视化数据...")
        
    #     # 1. 动态计算包围盒 (根据实际跑出来的轨迹)
    #     # 添加 10米的 buffer 缓冲，防止船贴着边缘
    #     buffer = 10.0 
    #     min_x, max_x = min(self.position_x) - buffer, max(self.position_x) + buffer
    #     min_y, max_y = min(self.position_y) - buffer, max(self.position_y) + buffer
        
    #     print(f"轨迹范围: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
        
    #     # 2. 生成全局网格 (Global Mesh) - 覆盖整个轨迹
    #     # 适当调整 global_dx (例如 0.5m 或 1.0m) 以平衡显存
    #     g_dx, g_dy = 0.5, 0.5
        
    #     self.global_plot_x = np.arange(min_x, max_x, g_dx)
    #     self.global_plot_y = np.arange(min_y, max_y, g_dy)
    #     self.global_plot_X, self.global_plot_Y = np.meshgrid(self.global_plot_x, self.global_plot_y)
        
    #     # 3. 计算波浪缓存 (Global)
    #     # 这里不需要 Local Cache 了，我们用全局网格覆盖所有区域即可
    #     # 左右两个窗口都用这一份数据，只是相机视角不同
    #     print(f"正在生成波浪网格数据 (网格点数: {self.global_plot_X.size})...")
    #     self.wave_init(self.global_plot_X, self.global_plot_Y, self.global_wave_heights_cache)
        
    #     # 4. 初始化 PyVista Plotter 和 Mesh
    #     self.plotter = pv.Plotter(shape=(1, 2), off_screen=self.off_screen)
    #     self.plotter.enable_depth_peeling(number_of_peels=4) # 开启深度剥离
        
    #     # 准备点云对象
    #     # 初始高度设为0
    #     self.global_points = np.column_stack((
    #         self.global_plot_X.ravel(), 
    #         self.global_plot_Y.ravel(), 
    #         np.zeros_like(self.global_plot_X.ravel())
    #     ))
    #     self.global_point_cloud = pv.PolyData(self.global_points)
    #     self.global_point_cloud.point_data["Elevation"] = self.global_points[:, 2]

    #     # 设置颜色范围
    #     z_lim = self.Hs * 0.8
        
    #     # === 左侧窗口 (跟随视角) ===
    #     self.plotter.subplot(0, 0)
    #     self.plotter.add_text("Local View (Follow)", font_size=10)
    #     self.plotter.add_mesh(
    #         self.global_point_cloud, # 复用同一个大网格
    #         scalars="Elevation",
    #         cmap='viridis_r',
    #         opacity=0.6,             # <--- [修复] 左侧透明度
    #         clim=[-z_lim, z_lim],
    #         show_scalar_bar=False
    #     )
    #     self.plotter.add_mesh(self.cube, color='orange', smooth_shading=True)

    #     # === 右侧窗口 (全局视角) ===
    #     self.plotter.subplot(0, 1)
    #     self.plotter.add_text("Global View", font_size=10)
    #     self.plotter.add_mesh(
    #         self.global_point_cloud, 
    #         scalars="Elevation",
    #         cmap='viridis_r',
    #         opacity=0.6,             # <--- [修复] 右侧透明度
    #         clim=[-z_lim, z_lim],
    #         show_scalar_bar=False
    #     )
    #     self.plotter.add_mesh(self.cube, color='orange', smooth_shading=True)
        
    #     self.plotter.open_movie(self.moviepath, framerate=self.fps)

    def prepare_visualization_data(self):
        """
        [优化版] 准备可视化数据
        1. 降低网格密度以提高FPS。
        2. 使用 StructuredGrid 创建连续水面以支持透明度。
        """
        print("正在准备可视化数据...")
        
        # 1. 动态计算包围盒
        buffer = 20.0 
        min_x, max_x = min(self.position_x) - buffer, max(self.position_x) + buffer
        min_y, max_y = min(self.position_y) - buffer, max(self.position_y) + buffer
        
        print(f"轨迹范围: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
        
        # 2. [关键优化] 降低可视化网格密度
        # 0.1 -> 0.5，点数减少25倍，渲染速度大幅提升
        # 如果依然卡顿，可以尝试改为 1.0
        g_dx, g_dy = 0.5, 0.5 
        
        # 生成网格坐标矩阵 (Meshgrid)
        x_range = np.arange(min_x, max_x, g_dx)
        y_range = np.arange(min_y, max_y, g_dy)
        self.global_plot_X, self.global_plot_Y = np.meshgrid(x_range, y_range)
        
        # 3. 计算波浪缓存
        print(f"正在生成波浪网格数据 (网格点数: {self.global_plot_X.size})...")
        # 注意：这里 wave_init 会把结果存入 cache，key是时间 t
        self.wave_init(self.global_plot_X, self.global_plot_Y, self.global_wave_heights_cache)
        
        # 4. 初始化 Plotter
        self.plotter = pv.Plotter(
            shape=(1, 2), 
            off_screen=self.off_screen,
            window_size=[1600, 900],  # [关键] 这里决定了视频分辨率
            # multi_samples=8            # [关键] 抗锯齿
        )
        # 开启深度剥离，确保半透明渲染正确
        self.plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
        
        # 5. [关键修改] 创建网格并转为 PolyData
        z_zeros = np.zeros_like(self.global_plot_X)
        
        # 先建立结构化网格
        temp_grid = pv.StructuredGrid(
            self.global_plot_X, 
            self.global_plot_Y, 
            z_zeros
        )
        # [重点] 提取表面转为 PolyData，它支持快速计算法线
        self.water_mesh = temp_grid.extract_surface()
        
        # 初始化标量 (取负号，让高度为正，深度为负)
        self.water_mesh.point_data["Elevation"] = -z_zeros.ravel(order='F')

        # 颜色范围 (根据波高动态调整)
        # 因为我们取了负号，z_lim 应该是 [-height, +height]
        z_lim = self.Hs * 0.6

        my_colormap = ["#001030", "#004080", "#0080FF", "#80C0FF"] 
        color_list = ["#001030", "#003060", "#0060A0", "#0080FF", "#80C0FF"]
        my_smooth_cmap = LinearSegmentedColormap.from_list("deep_ocean_smooth", color_list, N=256)
        
        # 定义材质
        water_props = dict(
            scalars="Elevation",   # 使用我们绑定的高度数据
            # cmap="ocean",          # 海洋色带 (低值深蓝，高值白色)
            cmap=my_smooth_cmap,
            opacity=0.75,
            clim=[-z_lim, z_lim],  # 范围
            show_scalar_bar=False,
            smooth_shading=True,
            specular=0.5,          # 1.0 -> 0.5 (柔和一点)
            specular_power=30.0,   # 128 -> 30 (光斑变大变散，像液态水)
            ambient=0.4,           # 提高环境光，让暗部别太死黑
            diffuse=0.8,
            lighting=True
        )
        # === 左侧窗口 ===
        self.plotter.subplot(0, 0)
        self.plotter.add_text("Local View (Follow)", font_size=18)
        self.plotter.add_mesh(self.water_mesh, **water_props)
        # 添加船体 (使用生成的AUV模型)
        self.plotter.add_mesh(self.cube, color='orange', smooth_shading=True)
        self.plotter.add_axes(line_width=5,labels_off=False)

        # === 右侧窗口 ===
        self.plotter.subplot(0, 1)
        self.plotter.add_text("Global View", font_size=18)
        self.plotter.add_mesh(self.water_mesh, **water_props)
        self.plotter.add_mesh(self.cube, color='orange', smooth_shading=True)
        self.plotter.add_axes(line_width=5,labels_off=False)


        # [可选] 添加一个灯光让波浪起伏更明显
        # light = pv.Light(position=(0, 0, -100), focal_point=(0, 0, 0), color='white', intensity=0.5)
        # self.plotter.add_light(light)
        self.plotter.remove_all_lights() # 清除默认灯光
        light_top = pv.Light(position=(0, 0, -100), focal_point=(0, 0, 0), color='white', intensity=0.7)
        self.plotter.add_light(light_top)
        
        self.plotter.open_movie(
            self.moviepath, 
            framerate=self.fps,  # 强制 60 帧，确保流畅
            quality=10     # [关键] 提升画质，减少模糊和伪影
        )

    def get_world_sample_points(self, vertices):
        """根据当前顶点位置计算世界坐标系中的采样点"""
        # 计算变换矩阵：局部到世界
        # 重心作为参考点
        g_center = np.mean(vertices, axis=0)
        
        # 构建变换矩阵（平移+旋转）
        # 这里简化处理：使用顶点0,1,3构建局部坐标系
        v0 = vertices[0]
        v1 = vertices[1]
        v3 = vertices[3]
        
        # 局部坐标轴
        local_x = (v1 - v0) / np.linalg.norm(v1 - v0)
        temp = (v3 - v0) / np.linalg.norm(v3 - v0)
        local_z = np.cross(local_x, temp)
        local_z /= np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        
        # 旋转矩阵
        R = np.column_stack([local_x, local_y, local_z])
        
        # 应用变换：world_points = R @ local_points.T + g_center
        world_points = (R @ self._local_sample_points.T).T + g_center
        
        return world_points, self._local_face_ids
    
    def update_J(self):
        """更新运动学矩阵 J_eta"""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.phi), np.sin(self.phi)],
            [0, -np.sin(self.phi), np.cos(self.phi)]
        ])
        Ry = np.array([
            [np.cos(self.theta), 0, np.sin(self.theta)],
            [0, 1, 0],
            [-np.sin(self.theta), 0, np.cos(self.theta)]
        ])
        Rz = np.array([
            [np.cos(self.psi), -np.sin(self.psi), 0],
            [np.sin(self.psi), np.cos(self.psi), 0],
            [0, 0, 1]
        ])
        # J1_eta1 = Rx @ Ry @ Rz
        J1_eta1 = Rz @ Ry @ Rx
        self.rotation_matrix=J1_eta1
        sf=np.sin(self.phi)
        sx=np.sin(self.theta)
        sp=np.sin(self.psi)
        cf=np.cos(self.phi)
        cx=np.cos(self.theta)
        cp=np.cos(self.psi)

        J1_eta2= np.array([
            [cx*cp,-cf*sp+sf*sx*cp,sf*sp+cf*sx*cp],
            [cx*sp,-cf*cp+sf*sx*sp,-sf*cp+cf*sx*sp],
            [-sx,cx*sp,cf*sx]
        ])
        if np.abs(np.abs(self.theta) - np.pi / 2) < 1e-6 or np.abs(np.abs(self.theta) + np.pi / 2) < 1e-6:
            J2_eta2 = np.eye(3)
        else:
            J2_eta2 = np.array([
                [1, np.sin(self.phi) * np.tan(self.theta), np.cos(self.phi) * np.tan(self.theta)],
                [0, np.cos(self.phi), -np.sin(self.phi)],
                [0, np.sin(self.phi) / np.cos(self.theta), np.cos(self.phi) / np.cos(self.theta)]
            ])
        # self.J_eta = np.block([
        #     [J1_eta1, np.zeros((3, 3))],
        #     [np.zeros((3, 3)), J2_eta2]
        # ])
        self.J_eta = np.block([
            [J1_eta1, np.zeros((3, 3))],
            [np.zeros((3, 3)), J2_eta2]
        ])

    def update_CV(self):
        """更新科里奥利矩阵 C_V"""
        a1, a2, a3, b1, b2, b3 = self.X_u_dot * self.u, self.Y_v_dot * self.v, self.Z_w_dot * self.w, self.K_p_dot * self.p, self.M_q_dot * self.q, self.N_r_dot * self.r
        CA_V = np.array([
            [0, 0, 0, 0, -a3, a2],
            [0, 0, 0, a3, 0, -a1],
            [0, 0, 0, -a2, a1, 0],
            [0, -a3, a2, 0, -b3, b2],
            [a3, 0, -a1, b3, 0, -b1],
            [-a2, a1, 0, -b2, b1, 0]
        ])
        CRB_V = np.array([
            [0, 0, 0, 0, self.m * self.w, self.m * self.v],
            [0, 0, 0, -self.m * self.w, 0, self.m * self.u],
            [0, 0, 0, self.m * self.v, -self.m * self.u, 0],
            [0, self.m * self.w, -self.m * self.v, 0, self.Iz * self.r, self.Iy * self.q],
            [-self.m * self.w, 0, self.m * self.u, -self.Iz * self.r, 0, self.Ix * self.p],
            [-self.m * self.v, -self.m * self.u, 0, -self.Iy * self.q, -self.Ix * self.p, 0]
        ])
        self.C_V = CA_V + CRB_V

    
    def update_DV(self):
        """更新阻尼矩阵 D_V"""
        self.D_V = np.array([
            [self.X_u + self.X_u_absu * np.abs(self.u), 0, 0, 0, 0, 0],
            [0, self.Y_v + self.Y_v_absv * np.abs(self.v), 0, 0, 0, 0],
            [0, 0, self.Z_w + self.Z_w_absw * np.abs(self.w), 0, 0, 0],
            [0, 0, 0, self.K_p + self.K_p_absp * np.abs(self.p), 0, 0],
            [0, 0, 0, 0, self.M_q + self.M_q_absq * np.abs(self.q), 0],
            [0, 0, 0, 0, 0, self.N_r + self.N_r_absr * np.abs(self.r)]
        ])



    
    def calc_hydro_forces(self, V):
        """
        根据 C++ 提供的 REMUS 系数计算水动力和力矩
        包含阻尼 (Damping) 和 耦合项 (Coupled Terms)
        输入 V: [u, v, w, p, q, r]
        输出 F_hydro: [Fx, Fy, Fz, K, M, N]
        """
        u, v, w, p, q, r = V
        
        # 辅助变量
        u_abs_u = u * abs(u)
        v_abs_v = v * abs(v)
        w_abs_w = w * abs(w)
        p_abs_p = p * abs(p)
        q_abs_q = q * abs(q)
        r_abs_r = r * abs(r)
        
        # 1. X 方向力 (Surge)
        # Fx = Xuu*u|u| + Xwq*w*q + Xqq*q*q + Xvr*v*r + Xrr*r*r + Xpr*p*r
        Fx = (self.Xuu * u_abs_u + 
              self.Xwq * w * q + 
              self.Xqq * q * q + 
              self.Xvr * v * r + 
              self.Xrr * r * r +
              self.Xpr * p * r)
              
        # 2. Y 方向力 (Sway)
        # Fy = Yv*v + Yr*r + Yv|v|*v|v| + Yr|r|*r|r| + Ypq*p*q
        Fy = (self.Yv * v + self.Yr * r + 
              self.Yv1v1 * v_abs_v + 
              self.Yr1r1 * r_abs_r + 
              self.Ypq * p * q)
              
        # 3. Z 方向力 (Heave)
        # Fz = Zw*w + Zq*q + Zvp*v*p + Zrr*r*r + Zw|w|*w|w| + Zq|q|*q|q|
        Fz = (self.Zw * w + self.Zq * q + 
              self.Zvp * v * p + 
              self.Zrr * r * r + # C++变量名是 Zrr, 对应系数 1.667e-3
              self.Zww * w_abs_w + 
              self.Zqq * q_abs_q)
              
        # 4. K 力矩 (Roll)
        # K = Kp*p + Kr*r + Kp|p|*p|p| + ...
        K_mom = (self.Kp * p + self.Kr * r + 
                 self.Kpp * p_abs_p + 
                 self.Kpq * p * q + 
                 self.Kqr * q * r + 
                 self.Kwp * w * p)
                 
        # 5. M 力矩 (Pitch)
        # M = Mw*w + Mq*q + Mww*w|w| + Mqq*q|q| + Muw*u*w + ...
        M_mom = (self.Mw * w + self.Mq * q + 
                 self.Mww * w_abs_w + 
                 self.Mqq * q_abs_q + 
                 self.Muw * u * w + 
                 self.Muq * u * q + 
                 self.Mvp * v * p + 
                 self.Mpr * p * r + 
                 self.Mrr * r * r)
                 
        # 6. N 力矩 (Yaw)
        # N = Nv*v + Nr*r + Nv|v|*v|v| + Nr|r|*r|r| + ...
        N_mom = (self.Nv * v + self.Nr * r + 
                 self.Nvv * v_abs_v + 
                 self.Nrr * r_abs_r + 
                 self.Npq * p * q + 
                 self.Nwp * w * p + 
                 self.Nvw * v * w)
                 
        return np.array([Fx, Fy, Fz, K_mom, M_mom, N_mom])
    

    def update_CRB(self):
        """计算刚体科里奥利力 C_RB * V"""
        # 仅保留刚体部分，因为附加质量部分已经在 calc_hydro_forces 中由系数包含了
        m = self.m
        u, v, w, p, q, r = self.V
        
        # 叉乘算子 S(v)
        # 简化的 C_RB 矩阵乘法结果
        # 假设重心在 (0,0,0) 的简化版，如果有重心偏移需完整公式
        # 这里用向量叉乘直接算： F_cor = m * (omega x v)
        
        omega = np.array([p, q, r])
        vel = np.array([u, v, w])
        
        F_cor_lin = m * np.cross(omega, vel) # m * (w x v)
        
        # 力矩 Coriolis: J * omega x omega (若 I 是对角)
        # 或者 M_cor = omega x (I * omega)
        I_vec = np.array([self.Ix, self.Iy, self.Iz])
        M_cor_ang = np.cross(omega, I_vec * omega)
        
        C_RB_V = np.zeros(6)
        C_RB_V[0:3] = F_cor_lin
        C_RB_V[3:6] = M_cor_ang
        
        self.C_RB_V = C_RB_V


    def update_gEta(self, t):
        # 1. 准备当前帧的变换矩阵
        # 旋转矩阵 (Python算好传进去，量很小)
        rotation = R.from_euler('xyz', [self.phi, self.theta, self.psi], degrees=False)
        rot_matrix = rotation.as_matrix().astype(np.float32)
        # 平移向量
        trans_vec = np.array([self.x, self.y, self.z], dtype=np.float32)

        total_force_global = np.zeros(3)
        total_moment_global = np.zeros(3)
        
        # 获取方块顶点和重心
        vertices = self.cube.points
        g_center = np.mean(vertices, axis=0,dtype=np.float32)

        # 2. 清零 GPU 结果数组
        self.d_force.copy_to_device(np.zeros(3, dtype=np.float32))
        self.d_moment.copy_to_device(np.zeros(3, dtype=np.float32))
    
        # 3. 启动加速核函数
        self.d_rot_matrix.copy_to_device(rot_matrix)
        self.d_trans_vec.copy_to_device(trans_vec)
        self.d_g_center.copy_to_device(g_center)
        fast_fsi_kernel[self.blocks_per_grid, self.threads_per_block](
            self.d_points_init, self.d_face_ids, self.d_normals, # 几何
            self.d_rot_matrix, self.d_trans_vec, self.d_g_center,
            self.d_wave_k, self.d_wave_omega, self.d_wave_theta, self.d_wave_amp, self.d_wave_phi, # 波浪
            float32(t), # 时间
            float32(self.rho), float32(self.g), float32(self.d_uuv), # 物理常数
            self.d_force, self.d_moment # 输出
        )
        total_force_global = self.d_force.copy_to_host()
        total_moment_global = self.d_moment.copy_to_host()
        
        R_inv = rot_matrix.T  # 旋转矩阵的转置 = 逆矩阵（全局到物体）
        total_force_body = R_inv @ total_force_global
        total_moment_body = R_inv @ total_moment_global
        
        # === 添加重力 ===
        W = self.m * self.g
        
        # 重力在物体坐标系中的分量
        F_gravity_body = np.array([
            W * np.sin(self.theta),
            -W * np.sin(self.phi) * np.cos(self.theta),
            -W * np.cos(self.phi) * np.cos(self.theta),
            -self.yg*W*np.cos(self.phi)*np.cos(self.theta)+self.zg*W*np.sin(self.phi)*np.cos(self.theta),
            self.zg*W*np.sin(self.theta)+self.xg*W*np.cos(self.phi)*np.cos(self.theta),
            -self.xg*W*np.cos(self.phi)*np.cos(self.theta)-self.yg*W*np.sin(self.theta)
        ])

        g_eta = np.zeros(6)
        g_eta[0:3] = total_force_body
        
        # 恢复力矩 = 浮力矩(物体坐标系)
        g_eta[3:6] = total_moment_body

        g_eta= g_eta + F_gravity_body

        # 数值稳定性检查
        if np.any(np.isnan(g_eta)) or np.any(np.abs(g_eta) > 1e10):
            print(f"警告：恢复力过大 {g_eta}，重置为0")
            g_eta = np.zeros(6)
        
        self.g_Eta = g_eta
    # def update_gEta(self, t):
    #     total_force_global = np.zeros(3)  # 全局坐标系中的总力
    #     total_moment_global = np.zeros(3)  # 全局坐标系中的总力矩（关于重心）


    #     # 获取此时此刻cube的顶点坐标   世界坐标系，z朝上
    #     vertices=self.cube.points
    #     face1=[vertices[0],vertices[1],vertices[2],vertices[3]]  # x-
    #     face2=[vertices[0],vertices[1],vertices[7],vertices[4]]  # y-
    #     face3=[vertices[0],vertices[3],vertices[5],vertices[4]]  # z-
    #     face4=[vertices[4],vertices[5],vertices[6],vertices[7]]  # x+
    #     face5=[vertices[2],vertices[3],vertices[5],vertices[6]]  # y+
    #     face6=[vertices[1],vertices[2],vertices[6],vertices[7]]  # z+

    #     faces=[face1,face2,face3,face4,face5,face6]

    #     g_center=(vertices[0]+vertices[1]+vertices[2]+vertices[3]+vertices[4]+vertices[5]+vertices[6]+vertices[7])/8

    #     # 压力方向是由面指向中心
    #     # 原点①（0，0，0）   指向点②（0，1，0），向量为（0，1，0）   即 ② - ①
    #     # ①面指向②中心   即 ② - ① 中心 - 面
    #     directions=[]
    #     for i in range(6):
    #         direction=g_center-(faces[i][0]+faces[i][1]+faces[i][2]+faces[i][3])/4
    #         direction=direction/np.linalg.norm(np.array(direction))
    #         directions.append(direction)

    #     # 现在每个面都有方向
    #     # 对每个面取面元，求面元上的压力
    #     for i in range(6):
    #         du=(faces[i][1]-faces[i][0])/np.linalg.norm(np.array(faces[i][1]-faces[i][0]))*self.d_uuv
    #         dv=(faces[i][3]-faces[i][0])/np.linalg.norm(np.array(faces[i][1]-faces[i][0]))*self.d_uuv

    #         for dx in range(int(np.linalg.norm(np.array(faces[i][1]-faces[i][0]))/self.d_uuv)):
    #             for dy in range(int(np.linalg.norm(np.array(faces[i][3]-faces[i][0]))/self.d_uuv)):
                    
    #                 d_point=faces[i][0]+du*(dx+0.5) + dv*(dy+0.5)

    #                 wave_height=self.wave_elevation_point(d_point[0],d_point[1],t)

    #                 if wave_height<d_point[2]:
    #                     pressure=self.rho*self.g*(d_point[2]-wave_height)*self.d_uuv**2
    #                     pressure_vector=-pressure*directions[i]

    #                     # 计算相对于重心的力矩
    #                     r = d_point - g_center
    #                     moment = np.cross(r, pressure_vector)

    #                     total_force_global += pressure_vector
    #                     total_moment_global += moment
    #     # === 坐标系转换：全局 -> 物体 ===
    #     # 计算旋转矩阵（局部到全局）
    #     rotation = R.from_euler('xyz', [self.phi, self.theta, self.psi], degrees=False)
    #     rot_matrix = rotation.as_matrix()  # 获取旋转矩阵
    #     R_inv = rot_matrix.T  # 旋转矩阵的转置 = 逆矩阵（全局到物体）
    #     total_force_body = R_inv @ total_force_global
    #     total_moment_body = R_inv @ total_moment_global
        
    #     # === 添加重力 ===
    #     W = self.m * self.g
        
    #     # 重力在物体坐标系中的分量
    #     F_gravity_body = np.array([
    #         W * np.sin(self.theta),
    #         -W * np.sin(self.phi) * np.cos(self.theta),
    #         -W * np.cos(self.phi) * np.cos(self.theta),
    #         -self.yg*W*np.cos(self.phi)*np.cos(self.theta)+self.zg*W*np.sin(self.phi)*np.cos(self.theta),
    #         self.zg*W*np.sin(self.theta)+self.xg*W*np.cos(self.phi)*np.cos(self.theta),
    #         -self.xg*W*np.cos(self.phi)*np.cos(self.theta)-self.yg*W*np.sin(self.theta)
    #     ])

    #     g_eta = np.zeros(6)
    #     g_eta[0:3] = total_force_body
        
    #     # 恢复力矩 = 浮力矩(物体坐标系)
    #     g_eta[3:6] = total_moment_body

    #     g_eta= g_eta + F_gravity_body

    #     # 数值稳定性检查
    #     if np.any(np.isnan(g_eta)) or np.any(np.abs(g_eta) > 1e10):
    #         print(f"警告：恢复力过大 {g_eta}，重置为0")
    #         g_eta = np.zeros(6)
        
    #     self.g_Eta = g_eta


   

    def update_V_u(self):
        """更新速度 u"""
        self.u = self.V[0]
        self.v = self.V[1]
        self.w = self.V[2]
        self.p = self.V[3]
        self.q = self.V[4]
        self.r = self.V[5]
    
    def update_Eta_x(self):
        """更新位置 x"""
        self.x = self.Eta[0]
        self.y = self.Eta[1]
        self.z = self.Eta[2]
        self.phi = self.Eta[3]
        self.theta = self.Eta[4]
        self.psi = self.Eta[5]

    def update_Tau_X(self):
        """更新控制力 X"""
        self.Tau_X = self.Tau[0]
        self.Tau_Y = self.Tau[1]
        self.Tau_Z = self.Tau[2]
        self.Tau_K = self.Tau[3]
        self.Tau_M = self.Tau[4]
        self.Tau_N = self.Tau[5]


    def update_params(self,t):
        """更新所有参数"""
        self.update_V_u()
        self.update_Eta_x()
        self.update_Tau_X()
        self.update_J()
        self.update_CV()
        self.update_DV()
        self.update_gEta(t)
        
        

    def simulate(self,tau_x0,tau_n0):
        """运行模拟"""
        # total_time = round(self.total_frames / self.fps, 2)  # 计算总时间并保留两位小数
    
        # with tqdm(total=total_time, desc="Simulation Progress", unit="s") as pbar:
        #     start_time=time.time()
        print("仿真时间：",self.sim_time)
        M_inv = np.linalg.inv(self.M)
        for frame in tqdm(range(int(self.total_frames)),dynamic_ncols=True,mininterval=0.5):
            t = frame / self.fps  # 当前时间
            # 1. 更新基本状态 (u, x, tau)
            self.update_params(t)

            # psi=self.psi
            # # --- 限制 Theta 防止奇点 ---
            # if self.theta > 1.4: self.theta = 1.4
            # if self.theta < -1.4: self.theta = -1.4
            # # -------------------------

            # # 2. 计算力
            # self.update_gEta(t) # 恢复力 + 波浪力 (Froude-Krylov)

            # # [新] 计算刚体科里奥利力
            # self.update_CRB()
            
            # # [新] 计算水动力 (阻尼 + 耦合项)
            # F_hydro = self.calc_hydro_forces(self.V)

            c = np.cos(self.psi)
            s = np.sin(self.psi)
            self.f_current_body_u =  self.F_current_x * c + self.F_current_y * s
            self.f_current_body_v = -self.F_current_x * s + self.F_current_y * c
            self.Tau_X = tau_x0 + self.f_current_body_u
            self.Tau_Y = 0 + self.f_current_body_v
            self.Tau_Z = 0
            self.Tau_K=0
            self.Tau_M=0
            self.Tau_N=tau_n0
            self.Tau = np.array([self.Tau_X, self.Tau_Y, self.Tau_Z, self.Tau_K,self.Tau_M,self.Tau_N])
            #self.Tau = np.linalg.inv(self.J_eta) @ self.Tau

            # self.V_dot = M_inv @ (self.Tau + F_hydro - self.C_RB_V - self.g_Eta)

            self.V_dot = M_inv @ (self.Tau - self.C_V @ self.V - self.D_V @ self.V - self.g_Eta)

            if np.linalg.norm(self.V_dot) > 500: # 异常大的加速度
                self.V_dot = self.V_dot / np.linalg.norm(self.V_dot) * 500

            self.V += self.V_dot * self.dt

            self.Eta_dot = self.J_eta @ self.V
            self.Eta += self.Eta_dot * self.dt

            self.position_x.append(self.Eta[0])
            self.position_y.append(self.Eta[1])
            self.position_z.append(self.Eta[2])
            self.position_phi_rad.append(self.Eta[3])
            self.position_theta_rad.append(self.Eta[4])
            self.position_psi_rad.append(self.Eta[5])
            self.position_phi_degree.append(self.Eta[3]/2/pi*360)
            self.position_theta_degree.append(self.Eta[4]/2/pi*360)
            self.position_psi_degree.append(self.Eta[5]/2/pi*360)

            # 更新方块位置 (两个可视化使用相同的方块位置)
            rotation = R.from_euler('xyz', [self.phi, self.theta, self.psi])
            rot_matrix = rotation.as_matrix()
            points = self.initial_cube.points - self.initial_cube.center
            rotated_points = (rot_matrix @ points.T).T
            cube_center = np.array([self.x, self.y, self.z])
            translated_points = rotated_points + cube_center
            
            # 更新方块
            self.cube.points = translated_points


    # def visualize(self):
    #     if not hasattr(self, 'plotter'):
    #         # 如果忘记调用 prepare，这里自动调用
    #         self.prepare_visualization_data()
    #     print(f"开始生成可视化动画 (FPS={self.fps}, 时长={self.sim_time}s)...")
        
    #     # 1. 修复视频时长：显式指定 framerate
    #     self.plotter.open_movie(self.moviepath, framerate=self.fps)
        
    #     # 预先获取引用
    #     self.plotter.subplot(0, 0)
    #     self.plotter.show_grid() # 可选：显示网格
    #     self.plotter.subplot(0, 1)
    #     self.plotter.show_grid()
        
    #     # 进度条
    #     pbar = tqdm(range(int(self.total_frames)), unit="frame")
        
    #     for frame in pbar:
    #         t = frame / self.fps
            
    #         # === A. 更新波浪数据 (GPU 预计算缓存) ===
    #         # 局部波浪
    #         if t in self.local_wave_heights_cache:
    #             # 只有当数据存在时才更新
    #             self.points[:, 2] = self.local_wave_heights_cache[t].ravel()
    #             self.point_cloud.points = self.points # 触发 PyVista 更新
    #             self.plotter.update_scalars(
    #                 scalars=self.points[:, 2], 
    #                     mesh=self.point_cloud, 
    #                     render=False
    #                 )
                
    #         # 全局波浪
    #         if t in self.global_wave_heights_cache:
    #             self.global_points[:, 2] = self.global_wave_heights_cache[t].ravel()
    #             self.global_point_cloud.points = self.global_points
    #             self.plotter.update_scalars(
    #                 scalars=self.global_points[:, 2], 
    #                     mesh=self.global_point_cloud, 
    #                     render=False
    #                 )

    #         # === B. 更新方块位置 (刚体变换) ===
    #         x, y, z = self.position_x[frame], self.position_y[frame], self.position_z[frame]
    #         phi, theta, psi = self.position_phi_rad[frame], self.position_theta_rad[frame], self.position_psi_rad[frame]
            
    #         # 计算旋转矩阵
    #         rotation = R.from_euler('xyz', [phi, theta, psi])
    #         rot_matrix = rotation.as_matrix()
            
    #         # 应用变换：先旋转后平移
    #         # 原始中心在(0,0,0)的点
    #         points_local = self.initial_cube.points - self.initial_cube.center 
    #         transformed_points = (rot_matrix @ points_local.T).T + np.array([x, y, z])
    #         self.cube.points = transformed_points
            
    #         # === C. 恢复特殊的相机视角逻辑 (原代码逻辑) ===
    #         # 仅在第一帧（或前几帧）设置一次，锁定视角
    #         if frame == 0:
    #             for i in [0, 1]:
    #                 self.plotter.subplot(0, i)
    #                 # 强制渲染一次以确保相机参数已初始化
    #                 self.plotter.render() 
                    
    #                 camera_position = self.plotter.camera.position
    #                 camera_focus = self.plotter.camera.focal_point
                    
    #                 # --- 你的原始逻辑：交换XY，Z取反 ---
    #                 # 这通常是为了配合 NED (北东地) 坐标系的可视化
    #                 new_camera_position = (camera_position[1], camera_position[0], -camera_position[2])
                    
    #                 self.plotter.camera.position = new_camera_position
    #                 self.plotter.camera.focal_point = camera_focus
    #                 self.plotter.camera.up = (0, 0, -1)  # 设置z轴向下（Z-down）
                    
    #                 # 重新对焦
    #                 self.plotter.reset_camera_clipping_range()

    #         # === D. 写入帧 ===
    #         self.plotter.write_frame()

    #     self.plotter.close()
    #     print(f"动画生成完成: {self.moviepath}")


    def visualize(self):
        if not hasattr(self, 'plotter'):
            # 如果忘记调用 prepare，这里自动调用
            self.prepare_visualization_data()

        grid_params = dict(
            color='black',          # 刻度线颜色
            font_size=10,           # 字体大小
            font_family='arial',    
            show_xaxis=True,        # 显示X轴
            show_yaxis=True,        # 显示Y轴
            show_zaxis=True,        # 显示Z轴
            xlabel='Distance X (m)',
            ylabel='Distance Y (m)',
            zlabel='Elevation Z (m)',
            grid='back',            # 网格画在背景墙上 ('back', 'front', 'all')
            location='outer',       # 刻度数值标在框外
            n_xlabels=5,            # X轴显示几个刻度
            n_ylabels=5,
            n_zlabels=3,
            use_2d=False,           # 3D 刻度
            fmt="%.1f"              # 数值格式 (保留1位小数)
        )

        self.plotter.subplot(0, 0)
        self.plotter.show_grid(**grid_params)
        self.plotter.subplot(0, 1)
        self.plotter.show_grid(**grid_params)

        self.plotter.open_movie(self.moviepath, framerate=self.fps)
            
        print("开始生成动画...")
        pbar = tqdm(range(int(self.total_frames)), unit="frame")

        distance = 40
        relative_offset = np.array([-40.0, -20.0, -32.0])
        # 设定死区阈值 (船偏离理想位置多少米后，相机才开始动)
        deadzone_radius = 5.0  
        # 设定平滑系数 (0.0 ~ 1.0)，越小越平滑但延迟越高，越大越紧跟
        # 0.05 表示每帧只修正 5% 的误差，非常顺滑
        smooth_factor = 0.05  

        # 初始化相机位置 (第一帧直接放在理想位置)
        current_cam_pos = np.array([
            self.position_x[0], self.position_y[0], self.position_z[0]
        ]) + relative_offset
        # 初始化关注点 (也是为了平滑，防止视线抖动)
        current_focal_point = np.array([
            self.position_x[0], self.position_y[0], self.position_z[0]
        ])

        
        for frame in pbar:
            t = frame / self.fps
            
            # 1. 更新波浪数据 (Global Cache)
            # if t in self.global_wave_heights_cache:
            #     new_z = self.global_wave_heights_cache[t].ravel()
                
            #     # 更新几何和颜色 (复用 mesh，所以只需要更新一次 global_point_cloud)
            #     self.global_points[:, 2] = new_z
            #     self.global_point_cloud.points = self.global_points
            #     self.global_point_cloud.point_data["Elevation"] = new_z

            if t in self.global_wave_heights_cache:
                # 获取 Z 值 (二维数组)
                z_values = self.global_wave_heights_cache[t] # shape: (ny, nx)
                
                # [关键] 更新 StructuredGrid 的坐标
                # 这种方式比 replace points 快
                # 注意：StructuredGrid 内部点的存储顺序，直接用 ravel('F') 是为了匹配 PyVista 内部结构
                # 但如果初始化时由 meshgrid(indexing='xy') 生成，通常直接 ravel() 即可。
                # 简单且稳健的方法：直接重新赋值 points
                # 为了性能，我们只更新 Z 轴，但这在 PyVista 里比较麻烦
                # 所以我们用最稳妥的方法：构建新点集
                
                new_points = np.column_stack((
                    self.global_plot_X.ravel(), 
                    self.global_plot_Y.ravel(), 
                    z_values.ravel()
                ))
                self.water_mesh.points = new_points
                # B. [关键] 更新颜色标量 (取反！)
                # 你的坐标系：Z正=深，Z负=高
                # 颜色逻辑：数值大=白(浪尖)，数值小=蓝(深渊)
                # 所以我们传 -z_values。
                # 例如：波峰 z=-2 -> 标量=+2 (白)；波谷 z=+2 -> 标量=-2 (蓝)
                self.water_mesh.point_data["Elevation"] = -z_values.ravel()

                # C. [关键] 重新计算法线 (PolyData 支持这个)
                # split_vertices=True 有助于处理边界的锐利光照，但可能会慢一点点，通常默认即可
                self.water_mesh.compute_normals(inplace=True) 


            # 2. 更新 AUV 位置姿态
            x, y, z = self.position_x[frame], self.position_y[frame], self.position_z[frame]
            phi, theta, psi = self.position_phi_rad[frame], self.position_theta_rad[frame], self.position_psi_rad[frame]
            
            # 计算变换
            rotation = R.from_euler('xyz', [phi, theta, psi])
            rot_matrix = rotation.as_matrix()
            points_local = self.initial_cube.points # 原始未动点
            # 旋转 + 平移
            self.cube.points = (rot_matrix @ points_local.T).T + np.array([x, y, z])
            
            # 3. 相机控制
            # 获取当前帧 船的真实位置
            ship_pos = np.array([
                self.position_x[frame], 
                self.position_y[frame], 
                self.position_z[frame]
            ])

            self.plotter.subplot(0, 0)
            ideal_cam_pos = ship_pos + relative_offset
            diff_vec = ideal_cam_pos - current_cam_pos
            distance = np.linalg.norm(diff_vec) # 计算距离

            if distance > deadzone_radius:
                # 如果超出了死区，相机开始移动
                # 但不是一步到位，而是每帧只移动一小部分 (Lerp)
                # 这样就消除了跳变，实现了连贯的运镜
                move_vec = diff_vec * smooth_factor
                current_cam_pos += move_vec
            
            # 4. 同样平滑关注点 (Focal Point)
            # 让视线中心也带有阻尼，避免船晃动时画面剧烈抖动
            focal_diff = ship_pos - current_focal_point
            current_focal_point += focal_diff * 0.1 # 关注点的跟随速度可以稍微快一点(0.1)
            
            # 5. 应用给 PyVista
            self.plotter.camera.position = tuple(current_cam_pos)
            self.plotter.camera.focal_point = tuple(current_focal_point)
            self.plotter.camera.up = (0, 0, -1) # 保持你要求的 Z轴向下 (如果是Z轴向上请改为 0,0,1)
            
            # # === 左侧窗口：相机跟随模式 ===
            # self.plotter.subplot(0, 0)
            # # 相机位置：在船的上方后方
            # # 简单跟随：相机始终在船的 (x-10, y, z+5) 位置
            # # 或者保持相对视角
            # if (abs(cam_x-(x - distance))>distance/2):
            #     cam_x = x +(cam_x-(x - distance))/abs(cam_x-(x - distance))
            
            #     self.plotter.camera.position = (cam_x, cam_y, cam_z)
            #     self.plotter.camera.focal_point = (x, y, z) # 盯着船看
            #     self.plotter.camera.up = (0, 0, -1) 
            # if (abs(cam_y-(y - distance * 0.5))>distance * 0.5/2):
            #     cam_y = y +(cam_y-(x - y - distance * 0.5))/abs(cam_y-(x - y - distance * 0.5))
            #     self.plotter.camera.position = (cam_x, cam_y, cam_z)
            #     self.plotter.camera.focal_point = (x, y, z) # 盯着船看
            #     self.plotter.camera.up = (0, 0, -1) 
            # === 右侧窗口：全局俯视模式 ===
            # if frame == 0: # 只在第一帧设置一次，之后不动
            #     self.plotter.subplot(0, 1)
            #     self.plotter.view_isometric()
            #     self.plotter.camera.position = (
            #         np.mean(self.position_x) - 30, 
            #         np.mean(self.position_y) - 30, 
            #         50
            #     )
            #     self.plotter.camera.focal_point = (np.mean(self.position_x), np.mean(self.position_y), 0)
            #     self.plotter.camera.up = (0, 0, 1)

            if frame == 0:
                # self.plotter.subplot(0, 0)
                # cam_x = x - distance  # 简单的后方跟随
                # cam_y = y - distance * 0.5
            
                # self.plotter.camera.position = (cam_x, cam_y, cam_z)
                # self.plotter.camera.focal_point = (x, y, z) # 盯着船看
                # self.plotter.camera.up = (0, 0, -1)

                self.plotter.subplot(0, 1)
                # 强制渲染一次以确保相机参数已初始化
                self.plotter.render() 
                
                camera_position = self.plotter.camera.position
                camera_focus = self.plotter.camera.focal_point
                
                # --- 你的原始逻辑：交换XY，Z取反 ---
                # 这通常是为了配合 NED (北东地) 坐标系的可视化
                new_camera_position = (camera_position[1], camera_position[0], -camera_position[2])
                
                self.plotter.camera.position = new_camera_position
                self.plotter.camera.focal_point = camera_focus
                self.plotter.camera.up = (0, 0, -1)  # 设置z轴向下（Z-down）
                
                # 重新对焦
                self.plotter.reset_camera_clipping_range()


            # 写入帧
            self.plotter.write_frame()

        self.plotter.close()
        print(f"动画生成完成: {self.moviepath}")



        # 左右拼接两个视频


    # def plot_trajectory(self):
    #     """绘制三维轨迹"""
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(self.position_x, self.position_y, self.position_z, label='Trajectory')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('3D Trajectory Plot')
    #     ax.legend()
    #     plt.show()
    #     # 保存图片至本地
    #     fig.savefig('./sim.png')
    def plot_trajectory(self,filename):
        """绘制多个参数随时间变化的曲线和三维轨迹"""
        # 创建一个包含7个子图的图形
        fig = plt.figure(figsize=(18, 10))
        
        # 设置子图布局：左侧3个子图，中间3个子图，右侧1个3D子图
        # 左侧：x、y、z随时间变化
        time= np.arange(0, self.sim_time, self.dt)
        ax1 = fig.add_subplot(331)
        ax1.plot(time, self.position_x)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('X')
        ax1.set_title('X vs Time')
        
        ax2 = fig.add_subplot(334)
        ax2.plot(time, self.position_y)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Y')
        ax2.set_title('Y vs Time')
        
        ax3 = fig.add_subplot(337)
        ax3.plot(time, self.position_z)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Z')
        ax3.set_title('Z vs Time')
        
        # 中间：phi、theta、psi随时间变化
        ax4 = fig.add_subplot(332)
        ax4.plot(time, self.position_phi_degree)  # 转换为角度
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Phi')
        ax4.set_title('Phi vs Time')
        
        ax5 = fig.add_subplot(335)
        ax5.plot(time, self.position_theta_degree)  # 转换为角度
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Theta')
        ax5.set_title('Theta vs Time')
        
        ax6 = fig.add_subplot(338)
        ax6.plot(time, self.position_psi_degree)  # 转换为角度
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Psi')
        ax6.set_title('Psi vs Time')
        
        # 右侧：3D轨迹
        ax7 = fig.add_subplot(133, projection='3d')
        ax7.plot(self.position_x, self.position_y, self.position_z, label='Trajectory')
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')
        ax7.set_zlabel('Z')
        ax7.set_title('3D Trajectory Plot')
        ax7.legend()
        
        # 自动调整布局
        plt.tight_layout()

        fig.savefig(filename, dpi=300, bbox_inches='tight') 
        
        # # 显示图形
        # plt.show()

           

    def write_data_to_file(self, filename):
        """将xyz数据写入文件"""
        try:
            with open(filename, 'w') as file:
                line = f"date,x(m),y(m),z(m),phi(°),theta(°),psi(°)\n"
                file.write(line)
                t=0
                for x, y, z,phi,theta,psi,  in zip(self.position_x, self.position_y, self.position_z,self.position_phi_rad,self.position_theta_rad,self.position_psi_rad):
                    t+=self.dt
                    line = f"{t:.3f},{x:.5f},{y:.5f},{z:.5f},{phi:.5f},{theta:.5f},{psi:.5f}\n"
                    file.write(line)
            print(f"数据已成功写入文件 {filename}")
        except Exception as e:
            print(f"写入文件时出现错误: {e}")

    def load_trajectory_from_file(self, filename):
        """
        从CSV文件加载轨迹数据，用于重现可视化。
        注意：文件中的角度是角度制(degree)，需要转回弧度制(radian)用于计算。
        """
        if not os.path.exists(filename):
            print(f"错误：文件 {filename} 不存在")
            return

        print(f"正在从 {filename} 加载轨迹数据...")
        
        # 使用 pandas 读取 CSV (自动处理表头)
        df = pd.read_csv(filename)
        
        # 1. 更新时间相关参数
        self.total_frames = len(df)
        # 获取最后一行的时间作为总时长
        self.sim_time = df['date'].iloc[-1] 
        # 尝试推断 dt (假设采样均匀)
        if len(df) > 1:
            self.dt = df['date'].iloc[1] - df['date'].iloc[0]
            self.fps = 1 / self.dt
        
        # 2. 填充位置数据
        self.position_x = df['x(m)'].values
        self.position_y = df['y(m)'].values
        self.position_z = df['z(m)'].values
        
        # 3. 填充角度数据 (注意：文件中是角度制，必须转回弧度制！)
        # visualize 函数中的 R.from_euler 需要弧度
        self.position_phi_rad = np.deg2rad(df['phi(°)'].values)
        self.position_theta_rad = np.deg2rad(df['theta(°)'].values)
        self.position_psi_rad = np.deg2rad(df['psi(°)'].values)

        # 同时也填充用于画图表的角度数组
        self.position_phi_degree = df['phi(°)'].values
        self.position_theta_degree = df['theta(°)'].values
        self.position_psi_degree = df['psi(°)'].values

        print(f"数据加载成功！总帧数: {self.total_frames}, 时长: {self.sim_time:.2f}s")

    # 定义方向分布函数 D(θ) (余弦型)
    def D(self, theta_val):
        # 步骤1：计算角度差，并归一化到 [-π, π] 区间（核心解决周期性）
        delta_theta = theta_val - self.main_theta
        # 使用 np.mod 实现周期归一：先转成 [0, 2π)，再转 [-π, π)
        delta_theta = np.mod(delta_theta + pi, 2 * pi) - pi
        
        # 步骤2：判断归一化后的角度差是否在 [-π/2, π/2] 范围内
        if -pi/2 <= delta_theta <= pi/2:
            return (2/pi) * np.cos(delta_theta)**2
        else:
            return 0  # 超出范围的方向返回0
        #return (2/pi) * np.cos(theta_val)**2  # 主浪向为θ=0

    # 定义频率谱 S(ω) (Pierson-Moskowitz)
    # def S(self,omega_val):
    #     B = 3.11 / self.Hs**2
    #     A = 8.1*10**(-3)*self.g**2
    #     return (A / omega_val**5) * np.exp(-B / omega_val**4)

    def S(self,omega,wave_type):
        # 'JONSWAP'  'pierson-Moskowitz'
        if wave_type=='JONSWAP':
            T1=self.T1
            gamma=3.3
            if(omega<=5.24/T1):
                sigma=0.07
            else:
                sigma=0.09
            Y=np.exp(-((0.191*omega*T1-1)/(2**0.5*sigma))**2)
            S= 155*self.Hs**2/T1**4*omega**(-5)*np.exp(-944/T1**4*omega**(-4))*gamma**Y
            return S
        if wave_type=='pierson-Moskowitz':
            B = 3.11 / self.Hs**2
            A = 8.1*10**(-3)*self.g**2
            return (A / omega**5) * np.exp(-B / omega**4)
        # 不然，退出程序
        print("生成波浪谱类型出错，检查'JONSWAP' or 'pierson-Moskowitz'!")
        exit()
    
    def wave_number(self, omega, depth=10.0):
        """计算色散关系 k(ω)"""
        k = omega**2 / self.g  # 初始近似
        for _ in range(5):  # 迭代求解
            k = omega**2 / (self.g * np.tanh(k * depth))
        return k
    
    def ensure_cache_dir_exists(self):
        """确保缓存目录存在"""
        if not os.path.exists(self.wave_cache_dir):
            os.makedirs(self.wave_cache_dir)
    def get_wave_data_filename(self,x_range,y_range,dx,dy):
        """根据波浪参数生成唯一的文件名"""
        # 提取关键波浪参数作为哈希
        wave_params = {
            "sim_time": self.sim_time,
            "fps": self.fps,
            "g":self.g,
            "Hs": self.Hs,
            "T1": self.T1,
            "wave_N": self.wave_N,
            "wave_M": self.wave_M,
            "omega_min": self.omega_min,
            "omega_max": self.omega_max,
            "omega":self.omega,
            "s":self.s,
            "wave_theta":self.wave_theta,
            "phi_ij":self.phi_ij,
            "wave_type":self.wave_type,
            "x_range": x_range,
            "y_range": y_range,
            "dx": dx,
            "dy": dy 
        }
        
        # 生成参数的哈希值作为文件名的一部分
        param_hash = hashlib.md5(str(wave_params).encode('utf-8')).hexdigest()[:8]
        return os.path.join(self.wave_cache_dir, f"wave_data_{param_hash}.npz")
    def load_wave_data_from_cache(self,file,x_range,y_range,dx,dy,cache):
        """从缓存加载波浪数据"""
        if os.path.exists(file):
            try:
                with np.load(file, allow_pickle=True) as data:
                    # 检查参数是否匹配
                    saved_params = data['params'].item()
                    current_params = {
                        "sim_time": self.sim_time,
                        "fps": self.fps,
                        "g":self.g,
                        "Hs": self.Hs,
                        "T1": self.T1,
                        "wave_N": self.wave_N,
                        "wave_M": self.wave_M,
                        "omega_min": self.omega_min,
                        "omega_max": self.omega_max,
                        "omega":self.omega,
                        "s":self.s,
                        "wave_theta":self.wave_theta,
                        "phi_ij":self.phi_ij,
                        "wave_type":self.wave_type,
                        "x_range": x_range,
                        "y_range": y_range,
                        "dx": dx,
                        "dy": dy 
                    }
                    
                    params_match = True
                    for k in current_params:
                        v1 = saved_params[k]
                        v2 = current_params[k]
                        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                            # 对于数组，使用allclose或array_equal
                            if not np.allclose(v1, v2):
                                params_match = False
                                break
                        else:
                            if v1 != v2:
                                params_match = False
                                break
                    
                    # 加载波浪高度数据
                    self.wave_heights_cache = {}
                    total_frames = int(self.sim_time * self.fps)
                    for frame in range(total_frames):
                        t = frame / self.fps
                        key = f'{t}'
                        if key in data:
                            cache[(t)] = data[key]
                    return True
            except Exception as e:
                print(f"加载波浪数据缓存失败: {e}")
                return False
        return False
    def save_wave_data_to_cache(self,file,x_range,y_range,dx,dy,cache):
        """将波浪数据保存到缓存"""
        try:
            # 准备要保存的参数
            params = {
                "sim_time": self.sim_time,
                "fps": self.fps,
                "g":self.g,
                "Hs": self.Hs,
                "T1": self.T1,
                "wave_N": self.wave_N,
                "wave_M": self.wave_M,
                "omega_min": self.omega_min,
                "omega_max": self.omega_max,
                "omega":self.omega,
                "s":self.s,
                "wave_theta":self.wave_theta,
                "phi_ij":self.phi_ij,
                "wave_type":self.wave_type,
                "x_range": x_range,
                "y_range": y_range,
                "dx": dx,
                "dy": dy 
            }
            # 准备要保存的数据
            save_data = {'params': params}
            total_frames = int(self.sim_time * self.fps)
            
            # 将波浪高度数据添加到保存字典中
            for frame in range(total_frames):
                t = frame / self.fps
                save_data[f'{t}'] = cache[(t)]
            
            # 保存到NPZ文件
            np.savez_compressed(file, **save_data)
            print(f"波浪数据已保存到 {file}")
        except Exception as e:
            print(f"保存波浪数据缓存失败: {e}")
    def compute_wave_heights_for_frame(self, plot_X, plot_Y, t):
        """计算单个时间步的局部和全局波浪高度"""
        if t%5==0:
            print(f"计算时间步 t={t:.2f} 的波浪高度...")
        wave_heights = self.wave_elevation_vectorized(plot_X, plot_Y, t=t)
        return wave_heights
    
    # def wave_init(self,plot_x,plot_y,cache):
    #     """预先计算整个仿真时间内的波浪高度"""
    #     print("预计算波浪高度...")
    #     total_frames = int(self.sim_time * self.fps)
    #     # 获取 CPU 核心数
    #     num_processes = multiprocessing.cpu_count()
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #         partial_compute_wave_heights = partial(self.compute_wave_heights_for_frame,plot_x, plot_y)
    #         # 生成时间步列表
    #         times = [frame / self.fps for frame in range(total_frames)]
    #         # 并行计算每个时间步的波浪高度
    #         results = list(executor.map(partial_compute_wave_heights, times))

    #     # 将结果存储到缓存中
    #     for t, wave_heights in zip(times, results):
    #         cache[(t)] = wave_heights

    #     print("波浪高度预计算完成。")
    # def wave_init(self, plot_x, plot_y, cache):
    #     """预先计算整个仿真时间内的波浪高度"""
    #     print("预计算波浪高度...")
    #     total_frames = int(self.sim_time * self.fps)
        
    #     # 获取 CPU 核心数
    #     num_processes = multiprocessing.cpu_count()
        
    #     # 准备好所有需要的参数（这些都是 numpy 数组或数值，可以被 pickle）
    #     # 使用 partial 固定那些不变的参数
    #     # 注意：partial 固定的参数在调用时会自动加在前面或通过关键字传入
    #     # 这里我们将 t 作为 executor.map 传入的变参，其余参数固定
    #     func_partial = partial(
    #         compute_wave_frame_standalone, 
    #         X=plot_x, 
    #         Y=plot_y, 
    #         wave_N=self.wave_N, 
    #         wave_M=self.wave_M, 
    #         omega=self.omega, 
    #         k=self.k, 
    #         wave_theta=self.wave_theta, 
    #         phi_ij=self.phi_ij, 
    #         amplitudes=self.amplitudes
    #     )

    #     with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #         # 生成时间步列表
    #         times = [frame / self.fps for frame in range(total_frames)]
            
    #         # 并行计算
    #         # executor.map 会将 times 中的每个元素作为第一个参数传给 func_partial
    #         # 因为我们在 partial 中指定了 X=..., Y=... 等关键字参数，
    #         # 所以 t 会自动作为第一个位置参数 compute_wave_frame_standalone(t, ...)
    #         results = list(tqdm(executor.map(func_partial, times), total=len(times), desc="Wave Pre-calc"))

    #     # 将结果存储到缓存中
    #     for t, wave_heights in zip(times, results):
    #         cache[(t)] = wave_heights

    #     print("波浪高度预计算完成。")

    def wave_init(self, plot_X, plot_Y, cache):
        """
        [GPU加速版] 预先计算整个仿真时间内的波浪高度
        """
        print("正在使用 GPU 加速预计算波浪高度...")
        
        # 1. 准备网格数据（扁平化并转为 float32）
        # 注意：这里传入的是 meshgrid 后的 X 和 Y
        x_flat = plot_X.ravel().astype(np.float32)
        y_flat = plot_Y.ravel().astype(np.float32)
        n_points = x_flat.shape[0]
        
        # 2. 准备波浪参数（转为 float32 并移至 GPU）
        # 这些参数在整个过程中是不变的，只拷贝一次到 GPU
        k_gpu = cuda.to_device(np.array(self.k, dtype=np.float32))
        omega_gpu = cuda.to_device(np.array(self.omega, dtype=np.float32))
        wave_theta_gpu = cuda.to_device(np.array(self.wave_theta, dtype=np.float32))
        amplitudes_gpu = cuda.to_device(np.array(self.amplitudes, dtype=np.float32))
        phi_ij_gpu = cuda.to_device(np.array(self.phi_ij, dtype=np.float32))
        g_val = np.float32(self.g)
        
        # 网格坐标移至 GPU
        x_gpu = cuda.to_device(x_flat)
        y_gpu = cuda.to_device(y_flat)
        
        # 结果容器（复用显存）
        eta_gpu = cuda.device_array(n_points, dtype=np.float32)
        
        # CUDA 线程配置
        threads_per_block = 256
        blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
        
        total_frames = int(self.sim_time * self.fps)
        
        # 3. 循环计算每一帧
        # 虽然这里有个 Python 循环，但繁重的计算都在 GPU 上完成
        for frame in tqdm(range(total_frames), desc="GPU Wave Calculation"):
            t = np.float32(frame / self.fps)
            
            # 调用核函数
            wave_elevation_kernel[blocks_per_grid, threads_per_block](
                x_gpu, y_gpu, t, k_gpu, omega_gpu, wave_theta_gpu, 
                amplitudes_gpu, phi_ij_gpu, g_val, eta_gpu
            )
            
            # 同步并拷贝回 CPU (这是必须的，因为我们要存入 cache 字典)
            # 如果显存足够大，可以先存如果不回传，但为了兼容后续 visualize 逻辑，先回传
            wave_heights = eta_gpu.copy_to_host()
            
            # 存入缓存 (reshape 回网格形状)
            cache[(t)] = wave_heights.reshape(plot_X.shape)

        print("GPU 波浪预计算完成。")

    def wave_elevation_vectorized(self, X, Y, t):
        phases = np.zeros((*X.shape, self.wave_N, self.wave_M))
        for i, omega_i in enumerate(self.omega):
            c = np.sqrt(self.g / self.k[i])  # 波速
            for j, theta_j in enumerate(self.wave_theta):
                # 增加传播项：c * t * cos/sin(theta)
                # propagation_x = c * t * np.cos(theta_j)
                # propagation_y = c * t * np.sin(theta_j)
                
                phases[..., i, j] = (
                    self.k[i] * (X * np.cos(theta_j) + 
                                Y * np.sin(theta_j))
                    - omega_i * t 
                    + self.phi_ij[i, j]
                )
        return np.sum(self.amplitudes * np.cos(phases), axis=(-1, -2))
    # def wave_elevation_point(self,X, Y, t):
    #     eta=0
    #     for i in range(self.wave_N):
    #         for j in range(self.wave_M):
    #             # 计算相位：k[i] * (X*cos(theta[j]) + Y*sin(theta[j])) - omega[i]*t + phi_ij[i,j]
    #             phase = self.k[i] * (X * cos(self.wave_theta[j]) + Y * sin(self.wave_theta[j])) - self.omega[i] * t + self.phi_ij[i, j]
    #             # 叠加波分量
    #             eta += self.amplitudes[i, j] * cos(phase)
    #     return eta

    def wave_elevation_point(self, X, Y, t):
        eta = 0
        for i in range(self.wave_N):
            c = np.sqrt(self.g / self.k[i])  # 波速
            for j in range(self.wave_M):
                # 增加传播项：c * t * cos/sin(theta)
                # propagation_x = c * t * cos(self.wave_theta[j])
                # propagation_y = c * t * sin(self.wave_theta[j])

                # 计算相位：k[i] * ((X - propagation_x) * cos(theta[j]) + (Y - propagation_y) * sin(theta[j])) - omega[i]*t + phi_ij[i,j]
                phase = self.k[i] * (X * cos(self.wave_theta[j]) + Y * sin(self.wave_theta[j])) - self.omega[i] * t + self.phi_ij[i, j]
                # 叠加波分量
                eta += self.amplitudes[i, j] * cos(phase)
        return eta
    # def wave_elevation_points(self, points, t):
    #     """一次性计算多个点的波高"""
    #     x = points[:, 0]
    #     y = points[:, 1]
    #     eta = np.zeros(len(points))
        
    #     # 预计算常数
    #     k = self.k
    #     omega = self.omega
    #     wave_theta = self.wave_theta
    #     amplitudes = self.amplitudes
    #     phi_ij = self.phi_ij
    #     g = self.g
        
    #     for i in range(len(omega)):
    #         c = np.sqrt(g / k[i])  # 波速
    #         for j in range(len(wave_theta)):
    #             # 传播项
    #             # propagation_x = c * t * np.cos(wave_theta[j])
    #             # propagation_y = c * t * np.sin(wave_theta[j])
                
    #             # 向量化计算相位
    #             phase = (
    #                 k[i] * (x  * np.cos(wave_theta[j]) + 
    #                         y  * np.sin(wave_theta[j]))
    #                 - omega[i] * t 
    #                 + phi_ij[i, j]
    #             )
    #             eta += amplitudes[i, j] * np.cos(phase)
        
    #     return eta
    

    
    def wave_elevation_points(self, points, t):
        """GPU加速的波高计算"""
        # 提取坐标
        x = points[:, 0].astype(np.float32)
        y = points[:, 1].astype(np.float32)
        
        # 波浪参数（转换为float32）
        k = np.array(self.k, dtype=np.float32)
        omega = np.array(self.omega, dtype=np.float32)
        wave_theta = np.array(self.wave_theta, dtype=np.float32)
        amplitudes = np.array(self.amplitudes, dtype=np.float32)
        phi_ij = np.array(self.phi_ij, dtype=np.float32)
        g = np.float32(self.g)
        
        # 点数
        n_points = len(points)
        
        # 分配GPU内存
        x_gpu = cuda.to_device(x)
        y_gpu = cuda.to_device(y)
        k_gpu = cuda.to_device(k)
        omega_gpu = cuda.to_device(omega)
        wave_theta_gpu = cuda.to_device(wave_theta)
        amplitudes_gpu = cuda.to_device(amplitudes)
        phi_ij_gpu = cuda.to_device(phi_ij)
        
        # 结果数组
        eta_gpu = cuda.device_array(n_points, dtype=np.float32)
        
        # 配置CUDA线程
        threads_per_block = 256
        blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
        
        # 启动核函数
        wave_elevation_kernel[blocks_per_grid, threads_per_block](
            x_gpu, y_gpu, t, k_gpu, omega_gpu, wave_theta_gpu, 
            amplitudes_gpu, phi_ij_gpu, g, eta_gpu
        )
        
        # # 同步并获取结果
        # cuda.synchronize()
        # eta = eta_gpu.copy_to_host()
        
        return eta_gpu



# 按照文献中，质量为177kg，重心位置为（0,0,0.1），浮心位置为（0,0,-0.05），重力加速度为9.8m/s^2
# 以螺旋桨分布内缩0.1为边界，长宽高分别为0.4m, 0.3m, 0.3m
# length=0.4m, width=0.3m, height=0.3m
# 重力加速度为9.8m/s^2，重力为1734.6N
# 当重力=浮力时，AUV在水中的体积为（rho*g*V_pai=mg -> V_pai=mg/g/rho） V_pai=177*9.8/1000/9.8=1.76m^3
# 当重力-浮力平衡时，没入水中的深度为h=(V_pai/width/length)=(1.76/0.3/0.4)=14.67cm=0.1467m
# 令AUV在水中漂浮平衡时为初始状态。此时水上部分高度为0.3-0.1467=0.1533m，水下部分高度为0.1467m

# m = 120.0
# length, width, height = 0.4, 0.3, 0.3
# dt, t_max, T = 0.0333, 300, 10
# g, rho, A_wp, nabla = 9.8, 1000, length * width, 0.1467
# GM_L, GM_T = 0.05, 0.05
# xg, yg, zg = 0.0, 0.0, 0.1
# xb, yb, zb = 0.0, 0.0, -0.05
# Ix, Iy, Iz = 10.7, 11.8, 13.4
# X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot = 58.4, 23.8, 23.8, 3.38, 1.18, 2.67
# X_u, Y_v, Z_w, K_p, M_q, N_r = 120, 90, 150, 50, 15, 18
# X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr = 90, 90, 120, 10, 12, 15



# 不按照文献，以浮木为例。浮木密度500kg/m3
# 质量为100斤，50kg。求得体积为0.1m3
# 长宽高4:3:3,分别为0.56, 0.42, 0.42
# 重心位置为（0,0,0.2），浮心位置为（0,0,0），重力加速度为9.8m/s^2
# 重力加速度为9.8m/s^2，重力为490
# 当重力=浮力时，浮木在水中的体积为（rho*g*V_pai=mg -> V_pai=mg/g/rho） V_pai=50/1000=0.05 m3
# 当重力-浮力平衡时，没入水中的深度为h=(V_pai/width/length)=(0.05/0.56/0.42)=0.2126m
# 令浮木在水中漂浮平衡时为初始状态。此时水上部分高度为0.42-0.2126=0.2074m，水下部分高度为0.2126m

# rho *g *V=1025*9.81* 4 *0.5 *0.5 =10055.25
# m<= 1025

# m = 1200
# # length, width, height = 0.56, 0.42, 0.42
# length, width, height = 4,0.5,0.5
# d_uuv=0.01

# # m = 500
# # # length, width, height = 0.56, 0.42, 0.42
# # length, width, height = 3, 2, 2
# # d_uuv=0.25

# fps=60
# dt, t_max, T = 1/fps,60, 10
# g, rho, A_wp  = 9.81, 1025, length * width
# nabla=  m / rho  # 排水体积 m³
# GM_L, GM_T = 0.05, 0.05
# xg, yg, zg = 0.0, 0.0, 0.2
# xb, yb, zb = 0.0, 0.0, 0

# Ix = 298
# Iy = 3.298e3
# Iz = 3.4e3
# X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot = 58.4, 23.8, 23.8, 3.38, 1.18, 2.67
# X_u, Y_v, Z_w, K_p, M_q, N_r = 120, 90, 150, 50, 15, 18
# X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr = 90, 90, 120, 10, 12, 15

# wave_N = 15        # 频率离散点数
# wave_M = 10        # 方向离散点数
# s= 5 # 方向集中度   0-10
# wave_type='JONSWAP'
# # 'JONSWAP'  'pierson-Moskowitz'

# # 生成频率范围 (避免 ω=0)
# omega_min = 0.1
# omega_max = 3.0

# range_max=10
# dd=0.1
# x_range=[-range_max,range_max]
# y_range=[-range_max,range_max]
# dx=dd
# dy=dd

# global_x_range = [-range_max*3, range_max*3]  # 更大的X范围
# global_y_range = [-range_max*3, range_max*3]  # 更大的Y范围
# global_dx = dd*10  # 更大的网格间距
# global_dy = dd*10

# off_screen=False

# cube_center = [0, 0, 0]  # 初始中心位置

# # 打开动画文件

# pi = np.pi

# Hs=1.5
# T1=5
# tau_x0=300
# tau_n0=0
# main_theta=0
# current_force_mag=0
# current_angle=0
# i=0
# # for i in range(0,180,20):
# main_theta=i/180*pi
# up_force=0

# if tau_x0==0 and up_force ==0:
#     folder=f"./仿真/无推力-质量{m}kg"
# else:
#     if tau_x0==0:
#         folder=f"./仿真/无推力-浮力增大{up_force}N-质量{m}kg"
#     elif up_force==0:
#         folder=f"./仿真/推力{tau_x0}N-质量{m}kg"
#     else:
#         folder=f"./仿真/推力{tau_x0}N-浮力增大{up_force}N-质量{m}kg"
# if not os.path.exists(folder):
#     # os.makedirs() 递归创建多级目录（若「仿真」目录也不存在，会一并创建）
#     # exist_ok=True 避免文件夹已存在时抛出异常（可选，增加鲁棒性）
#     os.makedirs(folder, exist_ok=True)
#     print(f"文件夹创建成功：{folder}")
# else:
#     print(f"文件夹已存在：{folder}")

# folder_name = os.path.basename(folder)

# csv_name=f"{folder}/{folder_name}.csv"
# jpg_name=f"{folder}/{folder_name}.jpg"
# moviepath=f"{folder}/{folder_name}.mp4"

# vehicle = UnderwaterVehicle(m,
#                             dt, t_max, T,
#                             length, width, height, g, rho, A_wp, nabla, GM_L, GM_T,
#                             xg, yg, zg, xb, yb, zb,
#                             Ix, Iy, Iz,
#                             X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot,
#                             X_u, Y_v, Z_w, K_p, M_q, N_r,
#                             X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr,
                            
#                             d_uuv,Hs,T1,wave_N,wave_M,s,wave_type,main_theta,current_force_mag,current_angle,
#                             omega_min, omega_max, x_range, y_range, dx, dy,global_x_range, global_y_range, global_dx, global_dy,
#                             off_screen, cube_center, moviepath)

# # try:
# vehicle.simulate(tau_x0=tau_x0,tau_n0=tau_n0,up_force=up_force)
# vehicle.write_data_to_file(csv_name)
# vehicle.plot_trajectory(jpg_name)
# vehicle.prepare_visualization_data()
# vehicle.visualize()


# # vehicle.load_trajectory_from_file(csv_name) 
# # vehicle.wave_init(vehicle.global_plot_X, vehicle.global_plot_Y, vehicle.global_wave_heights_cache)
# # # 如果需要局部波浪，也需要初始化 (如果 visualize 里用到了局部波浪)
# # vehicle.wave_init(vehicle.plot_X, vehicle.plot_Y, vehicle.local_wave_heights_cache)

# # # 3. 运行可视化
# # vehicle.visualize()






# rho *g *V=1000*9.8*0.5*0.4*0.4=787N
# m<= 787/9.8=80
m = 50
# length, width, height = 0.56, 0.42, 0.42
length, width, height = 1, 0.28, 0.28
d_uuv=0.01

# m = 500
# # length, width, height = 0.56, 0.42, 0.42
# length, width, height = 3, 2, 2
# d_uuv=0.25

fps=60
# dt, t_max= 1/fps,60
dt= 1/fps
g, rho, A_wp  = 9.8, 1000, length * width
nabla=  m / rho  # 排水体积 m³
GM_L, GM_T = 0.05, 0.05
xg, yg, zg = 0.0, 0.0, 0.2
xb, yb, zb = 0.0, 0.0, 0
# Ix, Iy, Iz = 10.7, 11.8, 13.4
# 长方体惯性矩计算（质量均匀分布）
Ix = (1/12) * m * (width**2 + height**2)
Iy = (1/12) * m * (length**2 + height**2)
Iz = (1/12) * m * (length**2 + width**2)
X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot = 58.4, 23.8, 23.8, 3.38, 1.18, 2.67
X_u, Y_v, Z_w, K_p, M_q, N_r = 120, 90, 150, 50, 15, 18
X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr = 90, 90, 120, 10, 12, 15



wave_N = 15        # 频率离散点数
wave_M = 10        # 方向离散点数
s= 5 # 方向集中度   0-10
wave_type='JONSWAP'
# 'JONSWAP'  'pierson-Moskowitz'

# 生成频率范围 (避免 ω=0)
omega_min = 0.1
omega_max = 3.0

range_max=10
dd=0.1
x_range=[-range_max,range_max]
y_range=[-range_max,range_max]
dx=dd
dy=dd

global_x_range = [-range_max*3, range_max*3]  # 更大的X范围
global_y_range = [-range_max*3, range_max*3]  # 更大的Y范围
global_dx = dd*10  # 更大的网格间距
global_dy = dd*10

off_screen=True

cube_center = [0, 0, 0]  # 初始中心位置

# 打开动画文件

pi = np.pi

# Hs=1.5
# T1=5
# tau_x0=0
tau_n0=0
# main_theta=0
current_force_mag=0
current_angle=0
i=0
# for i in range(0,180,20):
main_theta=i/180*pi


import argparse

def main():

    parser = argparse.ArgumentParser(description="接收参数")
    parser.add_argument("--Hs", type=float, default=1.5, help="Hs")
    parser.add_argument("--T1", type=float, default=5, help="T1")
    parser.add_argument("--main_theta", type=float, default=0, help="main_theta")

    parser.add_argument("--tau_x0", type=float, default=0, help="tau_x0")

    parser.add_argument("--t_max", type=float, default=30, help="t_max")

    parser.add_argument("--group", type=str, default="id",help="group")
    
    # 解析传入的参数
    args = parser.parse_args()


    Hs=args.Hs
    T1=args.T1
    main_theta=args.main_theta
    tau_x0=args.tau_x0
    t_max=args.t_max


    folder=f"./仿真1/"+args.group +"/"
    if not os.path.exists(folder):
        # os.makedirs() 递归创建多级目录（若「仿真」目录也不存在，会一并创建）
        # exist_ok=True 避免文件夹已存在时抛出异常（可选，增加鲁棒性）
        os.makedirs(folder, exist_ok=True)
        print(f"文件夹创建成功：{folder}")
    else:
        print(f"文件夹已存在：{folder}")

    csv_name=f"{folder}Hs-{Hs}-T1-{T1}-main_theta-{i}-tau_x0-{tau_x0}.csv"
    jpg_name=f"{folder}Hs-{Hs}-T1-{T1}-main_theta-{i}-tau_x0-{tau_x0}.jpg"
    moviepath=f"{folder}Hs-{Hs}-T1-{T1}-main_theta-{i}-tau_x0-{tau_x0}.mp4"

    vehicle = UnderwaterVehicle(m,
                                dt, t_max,
                                length, width, height, g, rho, A_wp, nabla, GM_L, GM_T,
                                xg, yg, zg, xb, yb, zb,
                                Ix, Iy, Iz,
                                X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot,
                                X_u, Y_v, Z_w, K_p, M_q, N_r,
                                X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr,
                                
                                d_uuv,Hs,T1,wave_N,wave_M,s,wave_type,main_theta,current_force_mag,current_angle,
                                omega_min, omega_max, x_range, y_range, dx, dy,global_x_range, global_y_range, global_dx, global_dy,
                                off_screen, cube_center, moviepath)

    # try:
    vehicle.simulate(tau_x0=tau_x0,tau_n0=tau_n0)
    vehicle.write_data_to_file(csv_name)
    vehicle.plot_trajectory(jpg_name)
    vehicle.prepare_visualization_data()
    vehicle.visualize()

if __name__ == "__main__":
    main()