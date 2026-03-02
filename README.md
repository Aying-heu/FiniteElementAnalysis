# UUV 6-DOF Hydrodynamic Simulator with GPU-Accelerated Panel Method
(基于GPU加速面元法与随机波浪谱的UUV六自由度水动力仿真系统)\

# 📝 项目简介
本项目是一个针对无人水下航行器（UUV）/ 浮标的高保真六自由度（6-DOF）水动力学仿真系统。系统不仅包含完整的刚体动力学与非线性水动力（REMUS模型）解算，更核心的创新在于：完全摒弃了传统的经验公式近似，采用 C++/CUDA 与 Python (Numba) 联合编程，实现了基于真实三维网格的 GPU 并行面元积分（Panel Method）。

系统能够在复杂的随机不规则波（JONSWAP/PM谱）环境下，实时且精确地计算 UUV 船体表面每一块面元的静水压力与动压力（Froude-Krylov 力），并支持高逼真度的 PyVista 3D 渲染，为水下机器人的抗浪控制、耐波性分析与运动规划提供了强大的数字孪生环境。

# ✨ 核心亮点 (Key Features)
- **🌊 高逼真度波浪谱建模**：实现了基于 JONSWAP 和 Pierson-Moskowitz 谱的不规则随机波浪场，支持多频率、多方向（余弦方向分布函数）的波浪叠加与色散关系迭代。
- **📐 动态三维网格生成**：纯 C++ 手写实现了基于正二十面体细分的 UUV 胶囊体（Capsule）网格生成算法，自适应划分面元并计算法向与面积，生成高质量 VTK 模型。
- **⚡ GPU/CUDA 极致加速**：将极度耗时的面元压力积分计算下放至 GPU。通过 Shared Memory 块内归约（Reduction）与原子操作（AtomicAdd），实现了成千上万个面元的波浪力与力矩的极速求解。
- **⚙️ 6-DOF 非线性水动力模型**：完整实现了包含附加质量矩阵、线/二次阻尼矩阵、科里奥利力以及基于 RAOs 参数的刚体运动学耦合解算。

# 🧠 核心算法解析 (Deep Dive)
## 1. 随机波浪谱生成 (Wave Spectrum Modeling)
现实中的海浪是不规则的。本项目采用长峰波/短峰波双重叠加模型，将海面波高$η(x,y,t)$ 离散为N个频率分量与M个方向分量的叠加：
- 频率谱$S(ω)$：实现了 JONSWAP 谱（适用于风浪成长的有限水域）与 Pierson-Moskowitz 谱（适用于充分发展的深水海域）。
- 方向谱$D(θ)$：采用能量守恒的余弦平方分布$D(θ)= π/2 * cos^2(\theta - \theta_{main} ) $，模拟风向对波浪传播的影响。
- 波幅计算：每一项微元波幅 $A_{i,j}=\sqrt{2*S(\omega_i)*D({\theta_j})*\Delta \omega * \Delta \theta} $ 。结合深水色散关系迭代求解波数 k，生成在时空中演化的三维随机波浪场。
## 2. 基于 CUDA 的面元积分法 (GPU-Accelerated Panel Integration)
传统仿真往往将浮力视为作用在固定浮心的常力，但在不规则波中，UUV 的湿表面积和压力分布是剧烈变化的。本项目采用了精确的面元法 (Panel Method)：
- 坐标系变换：将 UUV 表面的数千个三角面元从机体坐标系（Body Frame）通过旋转矩阵$R_{nb}$映射到世界坐标系（NED Frame）。
- 压力计算 (Froude-Krylov)：针对世界系下的每一个面元中心 $(x,y,z)$，判断其是否在瞬时波面 $η$ 之下。对于湿面元，计算：
  - 静水压力：$P_{static}=\rho g z $
  - 动压力 (Smith Effect)：考虑波浪随深度的指数衰减 $P_{dynamic}​=ρgAe^{−kz}cos(kx−ωt+ϕ)$
- 力与力矩求和：微元受力 $dF=−P * n * ds$，微元力矩 $dM= r * dF$。
- CUDA 加速：由于涉及 N_faces * N_waves * M_directions 次高频三角函数计算，纯 CPU 运算极慢。采用自定义 CUDA Kernel，每个 Thread 负责一个面元，利用 Shared Memory 树状归约 + atomicAdd 大幅消除了显存读写瓶颈，实现了物理引擎的实时运行
- 
## 3. 水动力系数与运动方程 (Hydrodynamics & RAOs)
除波浪力外，系统还建立了完备的 UUV 动力学模型（基于 REMUS 100 参数）：
- 将 $6×6$ 的刚体质量矩阵$M_{RB}$与附加质量矩阵 (Added Mass)$M_A$结合。
- 将复杂的线性阻尼与二次阻尼（如 $X_{u∣u∣}$,$Y _{v∣v∣}$等交叉耦合项）写入 ```update_DV()``` 及 ```calc_hydro_forces()```。
- 结合欧拉角运动学方程（解决万向节死锁前兆限制）与科里奥利力 $C(v)$，求解最终加速度 $\dot V = M^{-1}(\tau_{ctrl}+\tau_{wave}-C(V)V- C(V_r)V_r -D(V)V-g(\eta)) $
- 
# 🖥️ 渲染与可视化 (Visualization)
项目中内置了基于 PyVista 的双视角 3D 数字孪生渲染器：
- 全局视角 (Global View)：俯瞰大尺度波浪场与 UUV 运动轨迹，直观展示 JONSWAP 谱的波浪干涉与衍射效果。
- 相机跟随视角 (Local Follow View)：引入带死区（Deadzone）与 Lerp 平滑插值的无人机/追随者相机算法，完美跟踪 UUV 在汹涌海面上的升沉、横摇与纵摇动作。
- 引入了 StructuredGrid 与动态法线重计算（compute_normals），配合自定义海洋色带（Ocean Colormap），实现了具有高光反射质感的波浪渲染。

# 🛠️ 技术栈 (Tech Stack)
- 核心计算：C++ 17, CUDA 11/12 (并行计算, Shared Memory Reduction)
- 数值与矩阵：Eigen3 (C++), NumPy, Pandas
- 网格与几何：VTK 格式, C++ 动态面元生成算法
- 跨语言加速：Python Numba (@cuda.jit, @njit), Multiprocessing
- 3D 渲染：PyVista (OpenGL backend, Depth Peeling)

# 💡 
- 突出难度：为了解决不规则波浪下浮力与波浪力非线性时变的问题，我放弃了简单的集中力近似，手写了基于 CUDA 的微元法（Panel Method）积分引擎
- 在 CUDA 积分时，直接对全局内存做 atomicAdd 冲突太高，我使用了 Shared Memory（共享内存）进行了 Block 内的树状归约（Tree Reduction），最后只让 Thread 0 执行一次全局原子累加，极大地提升了显存带宽利用率
- 各种坐标系下的力矩转换（ry*fz - rz*fy 等）以及水动力参数的无量纲化还原（Scaling Factors）