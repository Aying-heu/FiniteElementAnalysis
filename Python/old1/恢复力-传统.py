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
                 dt,t_max,T,
                 length,width,height,
                 g,rho,A_wp,nabla,GM_L,GM_T,
                 xg,yg,zg,xb,yb,zb,
                 Ix,Iy,Iz,
                 X_u_dot,Y_v_dot,Z_w_dot,K_p_dot,M_q_dot,N_r_dot,
                 X_u,Y_v,Z_w,K_p,M_q,N_r,
                 X_u_absu,Y_v_absv,Z_w_absw,K_p_absp,M_q_absq,N_r_absr
                 ):
        
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



        # 参数设置
        self.g = 9.8       # 重力加速度 (m/s²)

        # 生成波浪谱相关参数
        self.Hs = 0.5       # 有效波高 (m)


        self.wave_N = 5        # 频率离散点数
        self.wave_M = 5        # 方向离散点数

        # 生成频率范围 (避免 ω=0)
        self.omega_min = 0.1
        self.omega_max = 3.0
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

        self.x_range=[-5,5]  
        self.y_range=[-5,5]
        self.dx=0.1
        self.dy=0.1
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
        self.plotter = pv.Plotter(off_screen=False)  # off_screen适合生成动画文件
        self.plotter.enable_depth_peeling()  # 提高深度精度

        # 设置坐标系方向
        # self.plotter.camera_position[0][2] = -self.plotter.camera_position[0][2]
        
        #self.plotter.camera.up = (0, 0, -1)  # Z轴向下 

        # 添加坐标系指示器
        self.plotter.add_axes(line_width=5, labels_off=False)


        # 添加点云到plotter
        self.actor = self.plotter.add_mesh(
            self.point_cloud, 
            scalars=self.points[:, 2],         # 用z值着色
            cmap='viridis',               # 选择色带（如'viridis', 'jet', 'coolwarm'等）
            point_size=5,
            render_points_as_spheres=True
        )

        # 方块参数
        self.cube_length = self.length
        self.cube_width = self.width
        self.cube_height = self.height
        self.cube_center = [0, 0, 0]  # 初始中心位置
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
        self.plotter.open_movie('./point_cloud_animation.mp4')


        # 可选：重置相机位置以适应新方向
        #self.plotter.camera_position = 'xy'  # 从XY平面视角查看（Z轴垂直屏幕）

        #print(self.plotter.camera_position)
        # self.plotter.camera.focal_point = (0, 0, 0)


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

    # def update_gEta(self,t):
    #     """更新恢复力向量 g_Eta"""
    #     G = np.diag([
    #         0,
    #         0,
    #         self.rho*self.g*self.A_wp,
    #         self.rho*self.g*self.nabla*self.GM_T,
    #         self.rho*self.g*self.nabla*self.GM_L,
    #         0])
    #     self.g_Eta = G @ self.Eta

    # def update_gEta(self, t):
    #     """更新恢复力向量 g_Eta，考虑波浪引起的浮力变化"""
    #     # 计算当前AUV位置处的波高
    #     wave_height = self.wave_elevation_point(self.x, self.y, t)
        
    #     # 计算瞬时排水体积变化 (基于波高和垂荡位置)
    #     delta_V = self.A_wp * (wave_height - self.z)
        
    #     # z方向恢复力 = 静水恢复力 + 波浪引起的浮力变化
    #     Fz = self.rho * self.g * (self.A_wp * self.z + delta_V)
        
    #     # 横摇和纵摇恢复力 (保持原样)
    #     G = np.diag([
    #         0,
    #         0,
    #         self.rho*self.g*self.A_wp,
    #         self.rho*self.g*self.nabla*self.GM_T,
    #         self.rho*self.g*self.nabla*self.GM_L,
    #         0])
    #     self.g_Eta = G @ self.Eta
    #     self.g_Eta[2]=Fz
   
    # def update_gEta(self, t):
    #     """更新恢复力向量 g_Eta，考虑波浪引起的液面高度变化"""
    #     # 获取当前位置的波浪高度
    #     wave_height = self.wave_elevation_point(self.Eta[1], self.Eta[0], t)

        
    #     # 考虑波浪高度对排水体积和水线面面积的影响（这里简单示例，实际可能更复杂）
    #     # 假设排水体积和水线面面积与波浪高度线性相关
    #     distance_wave_self=wave_height-(-(self.Eta[2]+self.height/2))  # 波浪高度与AUVi底面的距离
    #     effective_nabla = distance_wave_self * self.A_wp   # nabla,表示排水体积
    #     effective_A_wp = self.A_wp # A_wp，表示水线面面积
        
    #     G = np.diag([
    #         0,
    #         0,
    #         self.rho * self.g * effective_A_wp,  # Z方向恢复力为负值
    #         self.rho * self.g * effective_nabla * self.GM_T,
    #         self.rho * self.g * effective_nabla * self.GM_L,
    #         0])
    #     # 相对波浪面的位移
    #     wave_displacement = np.array([0, 0, wave_height, 0, 0, 0])
    #     self.g_Eta = G @ (self.Eta + wave_displacement)

    def update_gEta(self,t):
        """计算重力和浮力引起的恢复力和力矩"""
        # 提取姿态角
        phi = self.Eta[3]  # 横摇角
        theta = self.Eta[4]  # 俯仰角
        psi= self.Eta[5]  # 偏航角
        
        # 计算重力和浮力
        W = self.m * self.g  # 重力
        wave_height = self.wave_elevation_point(self.Eta[0], self.Eta[1], t)
        distance_wave_self=self.Eta[2]+self.height/2-wave_height  # 波浪高度与AUV底面的距离
        if distance_wave_self > self.height:
            distance_wave_self = self.height
        if distance_wave_self < 0:
            distance_wave_self = 0
        volume= distance_wave_self * self.A_wp   
        B = self.rho * self.g * volume  # 浮力
        
        # 计算恢复力和力矩
        g_eta = np.zeros(6)
   
        g_eta[0] = (W - B) * np.sin(theta)
        g_eta[1] = -(W - B) * np.sin(phi) * np.cos(theta)
        g_eta[2] = -(W - B) * np.cos(phi) * np.cos(theta)
        g_eta[3] = (B * self.yb - W * self.yg) * np.cos(phi) * np.cos(theta) + (-B * self.xb + W * self.zg) * np.sin(phi) * np.cos(theta)
        g_eta[4] = (-B * self.zb + W * self.zg) * np.sin(theta) + (-B * self.xb + W * self.xg) * np.cos(theta) * np.cos(phi)
        g_eta[5] = (B * self.xb - W * self.xg) * np.sin(theta) * np.cos(phi) - (-B * self.yb + W * self.yg) * np.sin(theta)
        self.g_Eta= g_eta

  

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
        

    def simulate(self):
        """运行模拟"""
        for frame in range(int(self.total_frames)):
            t = frame / self.fps  # 当前时间
            print(f"Processing frame {frame + 1}/{self.total_frames}")
            
            self.update_params(t)
            # self.Tau_X = 100 * np.sin(2 * np.pi / self.T * t) + 500
            # self.Tau_Y = 50 * np.sin(2 * np.pi / self.T * t)
            # self.Tau_Z = 10 * np.sin(2 * np.pi / self.T * t)
            self.Tau_X = 200
            self.Tau_Y = 0
            self.Tau_Z = 0
            self.Tau_K=0
            self.Tau_M=0
            self.Tau_N=20
            self.Tau = np.array([self.Tau_X, self.Tau_Y, self.Tau_Z, self.Tau_K,self.Tau_M,self.Tau_N])
            #self.Tau = np.linalg.inv(self.J_eta) @ self.Tau

            self.V_dot = np.linalg.inv(self.M) @ (self.Tau - self.C_V @ self.V - self.D_V @ self.V - self.g_Eta)
            self.V += self.V_dot * self.dt

            self.Eta_dot = self.J_eta @ self.V
            self.Eta += self.Eta_dot * self.dt

            self.position_x.append(self.Eta[0])
            self.position_y.append(self.Eta[1])
            self.position_z.append(self.Eta[2])


            self.points[:, 2] = self.wave_elevation_vectorized(self.plot_X, self.plot_Y, t=t).ravel()
            self.point_cloud.points = self.points  # 更新点云数据
            self.plotter.update_scalars(
                scalars=self.points[:, 2], 
                mesh=self.point_cloud, 
                render=False
            )



            target_world_x=self.x
            target_world_y=self.y
            target_world_z=self.z

            rotation_world_x=self.phi
            rotation_world_y=self.theta
            rotation_world_z=self.psi

            current_cube = self.initial_cube.copy()
            # 旋转顺序：偏航 -> 俯仰 -> 横滚
            # 注意角度转换：

            current_cube.rotate_z(np.degrees(rotation_world_z), inplace=True)  # 偏航角取反
            current_cube.rotate_y(np.degrees(rotation_world_y), inplace=True)   # 横滚
            current_cube.rotate_x(np.degrees(rotation_world_x), inplace=True)  # 俯仰





            # current_cube.rotate_z(np.degrees(self.psi))   # 1. 偏航 (Z轴)
            # current_cube.rotate_y(np.degrees(-self.theta)) # 2. 俯仰 (Y轴，角度取反)
            # current_cube.rotate_x(np.degrees(self.phi))    # 3. 横滚 (X轴)
                        


            # 位置转换：NED -> 可视化坐标系
            # X_viz = 东向(Y_ned)
            # Y_viz = 北向(X_ned)
            # Z_viz = -深度(Z_ned)
            # cube_center = np.array([
            #     self.Eta[1],    # 东向 -> X_viz
            #     self.Eta[0],    # 北向 -> Y_viz
            #     -self.Eta[2]    # 深度取反 -> Z_viz
            # ])
            cube_center = np.array([
                target_world_x,    # 东向 -> X_viz
                target_world_y,    # 北向 -> Y_viz
                target_world_z    # 深度取反 -> Z_viz
            ])
            # 计算当前方块质心
            old_center = current_cube.points.mean(axis=0)
            # 平移所有点
            current_cube.points += (cube_center - old_center)

            self.cube.points = current_cube.points

            self.plotter.update_coordinates(self.cube.points, render=False)  # 更新方块坐标


    
            self.plotter.render()            # 渲染新帧
            self.plotter.write_frame()       # 写入帧

        self.plotter.close()

        print("动画生成完成！")

    def plot_trajectory(self):
        """绘制三维轨迹"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.position_x, self.position_y, self.position_z, label='Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory Plot')
        ax.legend()
        plt.show()
        fig.savefig('./sim.png')

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

m = 40
# length, width, height = 0.56, 0.42, 0.42
length, width, height = 0.5, 0.4, 0.4
dt, t_max, T = 1/24,30, 10
g, rho, A_wp, nabla = 9.8, 1000, length * width, 0.2126
GM_L, GM_T = 0.05, 0.05
xg, yg, zg = 0.0, 0.0, 0.2
xb, yb, zb = 0.0, 0.0, 0
Ix, Iy, Iz = 10.7, 11.8, 13.4
X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot = 58.4, 23.8, 23.8, 3.38, 1.18, 2.67
X_u, Y_v, Z_w, K_p, M_q, N_r = 120, 90, 150, 50, 15, 18
X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr = 90, 90, 120, 10, 12, 15


vehicle = UnderwaterVehicle(m,
                            dt, t_max, T,
                            length, width, height, g, rho, A_wp, nabla, GM_L, GM_T,
                            xg, yg, zg, xb, yb, zb,
                            Ix, Iy, Iz,
                            X_u_dot, Y_v_dot, Z_w_dot, K_p_dot, M_q_dot, N_r_dot,
                            X_u, Y_v, Z_w, K_p, M_q, N_r,
                            X_u_absu, Y_v_absv, Z_w_absw, K_p_absp, M_q_absq, N_r_absr)

try:
    vehicle.simulate()
    vehicle.plot_trajectory()
finally:
    vehicle.plotter.close()















