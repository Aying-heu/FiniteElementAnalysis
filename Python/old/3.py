import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from math import pi, cos, sin, sqrt
import time
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial.transform import Rotation as R

# 预生成随机相位数组
np.random.seed(42)  # 固定随机种子确保结果可重现
class UnderwaterVehicle:
    def __init__(self,m,
                dt, t_max, T,
                length, width, height, g, rho, A_wp, nabla, GM_L, GM_T,
                xg, yg, zg, xb, yb, zb,
                Ix, Iy, Iz,
                X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot,
                X_u, Y_v, Z_w, K_p, M_q, N_r,
                X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr,
                
                d_uuv,Hs,wave_N,wave_M,
                omega_min, omega_max, x_range, y_range, dx, dy,
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
        self.T = T

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



        # 参数设置
        self.g = g       # 重力加速度 (m/s²)

        # 生成波浪谱相关参数
        self.Hs = Hs      # 有效波高 (m)


        self.wave_N = wave_N        # 频率离散点数
        self.wave_M = wave_M        # 方向离散点数

        # 生成频率范围 (避免 ω=0)
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.omega = np.linspace(self.omega_min, self.omega_max, self.wave_N)
        self.delta_omega =self.omega[1] - self.omega[0]

        # 生成方向范围 (0~2π)
        self.wave_theta = np.linspace(0, 2*pi, self.wave_M, endpoint=False)
        self.delta_theta = self.wave_theta[1] - self.wave_theta[0]

        self.phi_ij = np.random.uniform(0, 2*pi, (self.wave_N, self.wave_M))

        # 预计算波数
        self.k = self.omega**2 / g  # 波数 k = ω²/g (深水假设)

        # 预计算振幅
        self.amplitudes = np.zeros((self.wave_N, self.wave_M))
        for i in range(self.wave_N):
            for j in range(self.wave_M):
                self.amplitudes[i, j] = sqrt(2 * self.S(self.omega[i]) * self.D(self.wave_theta[j]) * self.delta_omega * self.delta_theta)

        self.x_range=x_range 
        self.y_range=y_range
        self.dx=dx
        self.dy=dy
        self.plot_x = np.arange(self.x_range[0], self.x_range[1] + self.dx, self.dx)
        self.plot_y = np.arange(self.y_range[0], self.y_range[1] + self.dy, self.dy)
        self.plot_X, self.plot_Y = np.meshgrid(self.plot_x, self.plot_y)


        self.sim_time = t_max
        self.fps = 1/dt
        self.total_frames = self.fps * self.sim_time
        self.points_num=((self.x_range[1]-self.x_range[0])/self.dx+1)*((self.y_range[1]-self.y_range[0])/self.dy+1)
    
        self.plot_Z = self.wave_elevation_vectorized(self.plot_X, self.plot_Y, t=0)  # 这里 t 可以指定
    
        self.points = np.column_stack((self.plot_X.ravel(), self.plot_Y.ravel(), self.plot_Z.ravel()))
        self.point_cloud = pv.PolyData(self.points)

        # 创建一个plotter对象
        self.plotter = pv.Plotter(off_screen=off_screen)  # off_screen适合生成动画文件
        # self.plotter.camera.up = (0, 0, -1)  # 设置z轴向下
        self.plotter.enable_depth_peeling()  # 提高深度精度
        # 添加点云到plotter
        self.actor = self.plotter.add_mesh(
            self.point_cloud, 
            scalars=self.points[:, 2],         # 用z值着色
            cmap='viridis_r',               # 选择色带（如'viridis', 'jet', 'coolwarm'等）
            point_size=5,
            render_points_as_spheres=True
        )
        self.plotter.add_axes(line_width=5, labels_off=False)

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
        self.cube_actor = self.plotter.add_mesh(self.cube, color='blue', opacity=0.7)

        # 打开动画文件
        self.plotter.open_movie(moviepath)


        self.d_uuv=d_uuv


    def update_J(self):
        """更新运动学矩阵 J_eta"""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.phi), np.sin(self.phi)],
            [0, -np.sin(self.phi), np.cos(self.phi)]
        ])
        Ry = np.array([
            [np.cos(self.theta), 0, -np.sin(self.theta)],
            [0, 1, 0],
            [np.sin(self.theta), 0, np.cos(self.theta)]
        ])
        Rz = np.array([
            [np.cos(self.psi), -np.sin(self.psi), 0],
            [np.sin(self.psi), np.cos(self.psi), 0],
            [0, 0, 1]
        ])
        # J1_eta1 = Rx @ Ry @ Rz
        J1_eta1 = Rz @ Ry @ Rx
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
        if np.abs(np.abs(self.theta) - np.pi / 2) < 1e-6:
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



    def update_gEta(self, t):
        total_force_global = np.zeros(3)  # 全局坐标系中的总力
        total_moment_global = np.zeros(3)  # 全局坐标系中的总力矩（关于重心）


        # 获取此时此刻cube的顶点坐标   世界坐标系，z朝上
        vertices=self.cube.points
        face1=[vertices[0],vertices[1],vertices[2],vertices[3]]  # x-
        face2=[vertices[0],vertices[1],vertices[7],vertices[4]]  # y-
        face3=[vertices[0],vertices[3],vertices[5],vertices[4]]  # z-
        face4=[vertices[4],vertices[5],vertices[6],vertices[7]]  # x+
        face5=[vertices[2],vertices[3],vertices[5],vertices[6]]  # y+
        face6=[vertices[1],vertices[2],vertices[6],vertices[7]]  # z+

        faces=[face1,face2,face3,face4,face5,face6]

        g_center=(vertices[0]+vertices[1]+vertices[2]+vertices[3]+vertices[4]+vertices[5]+vertices[6]+vertices[7])/8

        # 压力方向是由面指向中心
        # 原点①（0，0，0）   指向点②（0，1，0），向量为（0，1，0）   即 ② - ①
        # ①面指向②中心   即 ② - ① 中心 - 面
        directions=[]
        for i in range(6):
            direction=g_center-(faces[i][0]+faces[i][1]+faces[i][2]+faces[i][3])/4
            direction=direction/np.linalg.norm(np.array(direction))
            directions.append(direction)

        # 现在每个面都有方向
        # 对每个面取面元，求面元上的压力
        for i in range(6):
            du=(faces[i][1]-faces[i][0])/np.linalg.norm(np.array(faces[i][1]-faces[i][0]))*self.d_uuv
            dv=(faces[i][3]-faces[i][0])/np.linalg.norm(np.array(faces[i][1]-faces[i][0]))*self.d_uuv

            for dx in range(int(np.linalg.norm(np.array(faces[i][1]-faces[i][0]))/self.d_uuv)):
                for dy in range(int(np.linalg.norm(np.array(faces[i][3]-faces[i][0]))/self.d_uuv)):
                    
                    d_point=faces[i][0]+du*(dx+0.5) + dv*(dy+0.5)

                    wave_height=self.wave_elevation_point(d_point[0],d_point[1],t)

                    if wave_height<d_point[2]:
                        pressure=self.rho*self.g*(d_point[2]-wave_height)*self.d_uuv**2
                        pressure_vector = - pressure*directions[i]

                        # 计算相对于重心的力矩
                        r = d_point - g_center
                        moment = np.cross(r, pressure_vector)

                        total_force_global += pressure_vector
                        total_moment_global += moment
        # === 坐标系转换：全局 -> 物体 ===
        # 计算旋转矩阵（局部到全局）
        rotation = R.from_euler('xyz', [self.phi, self.theta, self.psi], degrees=False)
        rot_matrix = rotation.as_matrix()  # 获取旋转矩阵
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
        # while self.Eta[3] > pi:
        #     self.Eta[3] -= 2 * pi
        # while self.Eta[3] < -pi:
        #     self.Eta[3] += 2 * pi
        # while self.Eta[4] > pi:
        #     self.Eta[4] -= 2 * pi
        # while self.Eta[4] < -pi:
        #     self.Eta[4] += 2 * pi
        # while self.Eta[5] > pi:
        #     self.Eta[5] -= 2 * pi
        # while self.Eta[5] < -pi:
        #     self.Eta[5] += 2 * pi
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
        
        

    def simulate(self):
        """运行模拟"""
        for frame in range(int(self.total_frames)):
            t = frame / self.fps  # 当前时间
            print(f"Processing frame {frame + 1}/{self.total_frames}")
            
            self.update_params(t)
            # self.Tau_X = 100 * np.sin(2 * np.pi / self.T * t) + 500
            # self.Tau_Y = 50 * np.sin(2 * np.pi / self.T * t)
            # self.Tau_Z = 10 * np.sin(2 * np.pi / self.T * t)
            self.Tau_X = 0
            self.Tau_Y = 0
            self.Tau_Z = 0
            self.Tau_K=0
            self.Tau_M=0
            self.Tau_N=0
            self.Tau = np.array([self.Tau_X, self.Tau_Y, self.Tau_Z, self.Tau_K,self.Tau_M,self.Tau_N])
            #self.Tau = np.linalg.inv(self.J_eta) @ self.Tau

            self.V_dot = np.linalg.inv(self.M) @ (self.Tau - self.C_V @ self.V - self.D_V @ self.V - self.g_Eta)
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

        

            self.points[:, 2] = self.wave_elevation_vectorized(self.plot_X, self.plot_Y, t=t).ravel()
            self.point_cloud.points = self.points  # 更新点云数据
            self.plotter.update_scalars(
                scalars=self.points[:, 2], 
                mesh=self.point_cloud, 
                render=False
            )

            rotation = R.from_euler('xyz', [self.phi, self.theta, self.psi])
            rot_matrix = rotation.as_matrix()
            points = self.initial_cube.points - self.initial_cube.center
            rotated_points = (rot_matrix @ points.T).T
            cube_center = np.array([self.x, self.y, self.z])
            translated_points = rotated_points + cube_center
            
            # 更新方块
            self.cube.points = translated_points

            self.plotter.update_coordinates(self.cube.points, render=False)  # 更新方块坐标
    
            self.plotter.render()            # 渲染新帧
            self.plotter.write_frame()       # 写入帧

            if frame == 2:
                # 获取当前相机位置和关注点
                camera_position = self.plotter.camera.position
                camera_focus = self.plotter.camera.focal_point
                # 将相机的z坐标取负
                new_camera_position = (camera_position[1], camera_position[0], -camera_position[2])
                # 更新相机位置，保持关注点不变
                self.plotter.camera.position = new_camera_position
                self.plotter.camera.focal_point = camera_focus
                self.plotter.camera.up = (0, 0, -1)  # 设置z轴向下


        self.plotter.close()

        print("动画生成完成！")

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
    def plot_trajectory(self):
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
        
        # 显示图形
        plt.show()
        
        # 保存图片至本地
        fig.savefig('./sim_multi.png', dpi=300, bbox_inches='tight')    

    # def write_data_to_file(self, filename):
    #     """将xyz数据写入文件"""
    #     try:
    #         with open(filename, 'w') as file:
    #             for x, y, z,  in zip(self.position_x, self.position_y, self.position_z):
    #                 line = f"{x} {y} {z}\n"
    #                 file.write(line)
    #         print(f"数据已成功写入文件 {filename}")
    #     except Exception as e:
    #         print(f"写入文件时出现错误: {e}")

    # def read_and_plot_from_file(self, filename):
    #     """从文件中读取数据并绘制三维轨迹"""
    #     try:
    #         position_x = []
    #         position_y = []
    #         position_z = []
    #         with open(filename, 'r') as file:
    #             for line in file:
    #                 x, y, z = map(float, line.strip().split())
    #                 position_x.append(x)
    #                 position_y.append(y)
    #                 position_z.append(z)

    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.plot(position_x, position_y, position_z, label='Trajectory from file')
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')
    #         ax.set_zlabel('Z')
    #         ax.set_title('3D Trajectory Plot from File')
    #         ax.legend()
    #         plt.show()
    #     except Exception as e:
    #         print(f"读取文件时出现错误: {e}")

    # 定义方向分布函数 D(θ) (余弦型)
    def D(self,theta_val):
        return (2/pi) * np.cos(theta_val)**2  # 主浪向为θ=0

    # 定义频率谱 S(ω) (Pierson-Moskowitz)
    def S(self,omega_val):
        B = 3.11 / self.Hs**2
        A = 8.1*10**(-3)*self.g**2
        return (A / omega_val**5) * np.exp(-B / omega_val**4)


    # def wave_elevation_vectorized(self, X, Y, t):
    #     eta = np.zeros_like(X)
    #     for i in range(self.wave_N):
    #         for j in range(self.wave_M):
    #             # 计算相位：k[i] * (X*cos(theta[j]) + Y*sin(theta[j])) - omega[i]*t + phi_ij[i,j]
    #             phase = self.k[i] * (X * np.cos(self.wave_theta[j]) + Y * np.sin(self.wave_theta[j])) - self.omega[i] * t + self.phi_ij[i, j]
    #             # 叠加波分量
    #             eta += self.amplitudes[i, j] * np.cos(phase)
    #     return eta
    def wave_elevation_vectorized(self, X, Y, t):
        phases = np.zeros((*X.shape, self.wave_N, self.wave_M))
        for i, omega_i in enumerate(self.omega):
            for j, theta_j in enumerate(self.wave_theta):
                phases[..., i, j] = self.k[i]*(X*np.cos(theta_j) + Y*np.sin(theta_j)) - omega_i*t + self.phi_ij[i,j]
        return np.sum(self.amplitudes * np.cos(phases), axis=(-1,-2))
    
    def wave_elevation_point(self,X, Y, t):
        eta=0
        for i in range(self.wave_N):
            for j in range(self.wave_M):
                # 计算相位：k[i] * (X*cos(theta[j]) + Y*sin(theta[j])) - omega[i]*t + phi_ij[i,j]
                phase = self.k[i] * (X * cos(self.wave_theta[j]) + Y * sin(self.wave_theta[j])) - self.omega[i] * t + self.phi_ij[i, j]
                # 叠加波分量
                eta += self.amplitudes[i, j] * cos(phase)
        return eta



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
# g, rho, A_wp, nabla = 9.8, 1000, length * width, 1.76
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

m = 30
# length, width, height = 0.56, 0.42, 0.42
length, width, height = 0.5, 0.4, 0.4
d_uuv=0.05

fps=24
dt, t_max, T = 1/fps,30, 10
g, rho, A_wp = 9.8, 1000, length * width
nabla = m / (rho * length * width)
GM_L, GM_T = 0.05, 0.05
xg, yg, zg = 0.0, 0.0, 0.2
xb, yb, zb = 0.0, 0.0, 0
Ix, Iy, Iz = 10.7, 11.8, 13.4
X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot = 58.4, 23.8, 23.8, 3.38, 1.18, 2.67
X_u, Y_v, Z_w, K_p, M_q, N_r = 120, 90, 150, 50, 15, 18
X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr = 90, 90, 120, 10, 12, 15



# 生成波浪谱相关参数
Hs = 0.5      # 有效波高 (m)

wave_N = 5        # 频率离散点数
wave_M = 5        # 方向离散点数

# 生成频率范围 (避免 ω=0)
omega_min = 0.1
omega_max = 3.0

x_range=[-5,5]  
y_range=[-5,5]
dx=0.1
dy=0.1

off_screen=True

cube_center = [0, 0, 0]  # 初始中心位置

# 打开动画文件
moviepath='./point_cloud_animation.mp4'


vehicle = UnderwaterVehicle(m,
                            dt, t_max, T,
                            length, width, height, g, rho, A_wp, nabla, GM_L, GM_T,
                            xg, yg, zg, xb, yb, zb,
                            Ix, Iy, Iz,
                            X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot,
                            X_u, Y_v, Z_w, K_p, M_q, N_r,
                            X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr,
                            
                            d_uuv,Hs,wave_N,wave_M,
                            omega_min, omega_max, x_range, y_range, dx, dy,
                            off_screen, cube_center, moviepath)

try:
    vehicle.simulate()
    vehicle.plot_trajectory()
finally:
    vehicle.plotter.close()

