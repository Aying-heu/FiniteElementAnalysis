import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import signal
import os


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

        # 按照文献中，质量为177kg，重心位置为（0,0,0.1），浮心位置为（0,0,-0.05），重力加速度为9.8m/s^2
        # 以螺旋桨分布内缩0.1为边界，长宽高分别为0.4m, 0.3m, 0.3m
        # length=0.4m, width=0.3m, height=0.3m
        # 重力加速度为9.8m/s^2，重力为1734.6N
        # 当重力=浮力时，AUV在水中的体积为（rho*g*V_pai=mg -> V_pai=mg/g/rho） V_pai=177*9.8/1000/9.8=1.76m^3
        # 当重力-浮力平衡时，没入水中的深度为h=(V_pai/width/length)=(1.76/0.3/0.4)=14.67cm=0.1467m
        # 令AUV在水中漂浮平衡时为初始状态。此时水上部分高度为0.3-0.1467=0.1533m，水下部分高度为0.1467m

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
            [np.cos(self.psi), np.sin(self.psi), 0],
            [-np.sin(self.psi), np.cos(self.psi), 0],
            [0, 0, 1]
        ])
        J1_eta1 = Rx @ Ry @ Rz
        if np.abs(np.abs(self.theta) - np.pi / 2) < 1e-6:
            J2_eta2 = np.eye(3)
        else:
            J2_eta2 = np.array([
                [1, np.sin(self.phi) * np.tan(self.theta), np.cos(self.phi) * np.tan(self.theta)],
                [0, np.cos(self.phi), -np.sin(self.phi)],
                [0, np.sin(self.phi) / np.cos(self.theta), np.cos(self.phi) / np.cos(self.theta)]
            ])
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

    def update_gEta(self):
        """更新恢复力向量 g_Eta"""
        G = np.diag([
            0,
            0,
            self.rho*self.g*self.A_wp,
            self.rho*self.g*self.nabla*self.GM_T,
            self.rho*self.g*self.nabla*self.GM_L,
            0])
        self.g_Eta = G @ self.Eta

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

    def update_params(self):
        """更新所有参数"""
        self.update_V_u()
        self.update_Eta_x()
        self.update_Tau_X()
        self.update_J()
        self.update_CV()
        self.update_DV()
        self.update_gEta()
        

    def simulate(self):
        """运行模拟"""
        for t in np.arange(0, self.t_max, self.dt):
            self.update_params()
            # self.Tau_X = 100 * np.sin(2 * np.pi / self.T * t) + 500
            # self.Tau_Y = 50 * np.sin(2 * np.pi / self.T * t)
            # self.Tau_Z = 10 * np.sin(2 * np.pi / self.T * t)
            self.Tau_X = 100 
            self.Tau_Y = 0
            self.Tau_Z = 0
            self.Tau_K=0
            self.Tau_M=0
            self.Tau_N=10
            self.Tau = np.array([self.Tau_X, self.Tau_Y, self.Tau_Z, self.Tau_K,self.Tau_M,self.Tau_N])
            self.Tau = np.linalg.inv(self.J_eta) @ self.Tau

            self.V_dot = np.linalg.inv(self.M) @ (self.Tau - self.C_V @ self.V - self.D_V @ self.V - self.g_Eta)
            self.V += self.V_dot * self.dt

            self.Eta_dot = self.J_eta @ self.V
            self.Eta += self.Eta_dot * self.dt

            self.position_x.append(self.Eta[0])
            self.position_y.append(self.Eta[1])
            self.position_z.append(self.Eta[2])

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

    def write_data_to_file(self, filename):
        """将xyz数据写入文件"""
        try:
            with open(filename, 'w') as file:
                for x, y, z,  in zip(self.position_x, self.position_y, self.position_z):
                    line = f"{x} {y} {z}\n"
                    file.write(line)
            print(f"数据已成功写入文件 {filename}")
        except Exception as e:
            print(f"写入文件时出现错误: {e}")

    def read_and_plot_from_file(self, filename):
        """从文件中读取数据并绘制三维轨迹"""
        try:
            position_x = []
            position_y = []
            position_z = []
            with open(filename, 'r') as file:
                for line in file:
                    x, y, z = map(float, line.strip().split())
                    position_x.append(x)
                    position_y.append(y)
                    position_z.append(z)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(position_x, position_y, position_z, label='Trajectory from file')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Trajectory Plot from File')
            ax.legend()
            plt.show()
        except Exception as e:
            print(f"读取文件时出现错误: {e}")


# 主函数
if __name__ == "__main__":
    # ... 原有的参数设置 ...
    m = 177.0
    length, width, height = 0.4, 0.3, 0.3
    dt, t_max, T = 0.0333, 300, 10
    g, rho, A_wp, nabla = 9.8, 1000, length * width, 0.1467
    GM_L, GM_T = 0.05, 0.05
    xg, yg, zg = 0.0, 0.0, 0.1
    xb, yb, zb = 0.0, 0.0, -0.05
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


    vehicle.simulate()
    vehicle.plot_trajectory()
    # vehicle.write_data_to_file("trajectory_data.txt")
