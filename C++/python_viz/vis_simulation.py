import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LinearSegmentedColormap
import os
from numba import cuda, float32

# ================= 配置区域 =================
DATA_DIR = "/home/robot/AAA/UUV_Model/data/"  
MESH_FILE = os.path.join(DATA_DIR, "UUVHull.vtk")
# WAVE_FILE = os.path.join(DATA_DIR, "wave_parameter/wave_JONSWAP_Hs1.5_T5.0_N30_M20_s5_th0.00.csv")
WAVE_FILE = os.path.join(DATA_DIR, "wave_parameter/wave_JONSWAP_Hs0.0_T3.0_N30_M20_s5_th60.00_depth20.00.csv")
TRAJ_FILE = os.path.join(DATA_DIR, "record/motion/Model_State20260107_1059.csv")

# [优化] 设置目标视频帧率，避免渲染过高频率的仿真步长
TARGET_FPS = 30  

traj_filename = os.path.basename(TRAJ_FILE)
traj_core_name = os.path.splitext(traj_filename)[0]

OUTPUT_PNG = os.path.join(DATA_DIR, f"visualize/{traj_core_name}.png")
OUTPUT_MP4 = os.path.join(DATA_DIR, f"visualize/{traj_core_name}_opt.mp4") # 输出文件名加_opt区分


@cuda.jit
def wave_elevation_kernel(x, y, t, k, omega, wave_theta, amplitudes, phi_ij, 
                        g, eta_out):
    """
    CUDA核函数：并行计算波高
    [优化] 移除了未使用的变量计算 c
    """
    point_idx = cuda.grid(1)
    
    if point_idx < x.shape[0]:
        total_eta = 0.0
        x_point = x[point_idx]
        y_point = y[point_idx]
        
        # 遍历所有频率和方向
        for i in range(k.shape[0]):
            k_i = k[i]
            omega_i = omega[i]
            theta_j = wave_theta[i]
            
            # [优化] 移除了 c = math.sqrt(g / k_i) 因为公式里没用到，减少计算量
            
            phase = (k_i * (x_point * math.cos(theta_j) + 
                        y_point * math.sin(theta_j))
                    - omega_i * t 
                    + phi_ij[i])
            
            total_eta += amplitudes[i] * math.cos(phase)
        
        eta_out[point_idx] = total_eta

def wave_init(plot_X, plot_Y, wave_groups, cache, t_array):
    """
    [GPU加速版] 预先计算整个仿真时间内的波浪高度
    """
    print("正在使用 GPU 加速预计算波浪高度...")
    
    x_flat = plot_X.ravel().astype(np.float32)
    y_flat = plot_Y.ravel().astype(np.float32)
    n_points = x_flat.shape[0]
    
    # 拷贝波浪参数到 GPU
    k_gpu = cuda.to_device(np.array(wave_groups['k'], dtype=np.float32))
    omega_gpu = cuda.to_device(np.array(wave_groups['omega'], dtype=np.float32))
    wave_theta_gpu = cuda.to_device(np.array(wave_groups['theta'], dtype=np.float32))
    amplitudes_gpu = cuda.to_device(np.array(wave_groups['amplitude'], dtype=np.float32))
    phi_ij_gpu = cuda.to_device(np.array(wave_groups['phase'], dtype=np.float32))
    g_val = np.float32(9.81)
    
    x_gpu = cuda.to_device(x_flat)
    y_gpu = cuda.to_device(y_flat)
    eta_gpu = cuda.device_array(n_points, dtype=np.float32)
    
    threads_per_block = 256
    blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
    
    # [优化] 直接遍历需要渲染的时间点，而不是 fps * time
    for i, t_val in enumerate(tqdm(t_array, desc="GPU Wave Calculation")):
        t = np.float32(t_val)
        
        wave_elevation_kernel[blocks_per_grid, threads_per_block](
            x_gpu, y_gpu, t, k_gpu, omega_gpu, wave_theta_gpu, 
            amplitudes_gpu, phi_ij_gpu, g_val, eta_gpu
        )
        
        wave_heights = eta_gpu.copy_to_host()
        cache[i] = wave_heights.reshape(plot_X.shape) # 使用索引 i 存储

    print("GPU 波浪预计算完成。")

def plot_6dof_2x3(output_png=None, show_figure=False):
    """
    保持原绘图逻辑不变
    """
    try:
        df = pd.read_csv(TRAJ_FILE, sep=r'[,]+', engine='python')
        df = df.dropna(axis=1, how='all')
    except Exception as e:
        print(f"加载轨迹文件失败: {e}")
        return

    if output_png is None:
        output_png = OUTPUT_PNG
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle('UUV 6-DOF Motion Results (NED Frame)', fontsize=16, fontweight='bold')
    
    t = df['t']
    
    # Position
    axes[0, 0].plot(t, df['x'], linewidth=2, color='#1f77b4')
    axes[0, 0].set_title('X (Surge)')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(t, df['y'], linewidth=2, color='#ff7f0e')
    axes[0, 1].set_title('Y (Sway)')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(t, df['z'], linewidth=2, color='#2ca02c')
    axes[0, 2].set_title('Z (Heave/Depth)')
    axes[0, 2].grid(True)
    
    # Attitude
    roll_deg = np.degrees(df['roll'].fillna(0))
    pitch_deg = np.degrees(df['pitch'].fillna(0))
    heading_deg = np.degrees(df['heading'].fillna(0))
    heading_rad_unwrap = np.unwrap(np.radians(heading_deg), period=2*np.pi)
    heading_deg_unwrap = np.degrees(heading_rad_unwrap)
    
    axes[1, 0].plot(t, roll_deg, linewidth=2, color='#d62728')
    axes[1, 0].set_title('Roll')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(t, pitch_deg, linewidth=2, color='#9467bd')
    axes[1, 1].set_title('Pitch')
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(t, heading_deg_unwrap, linewidth=2, color='#8c564b')
    axes[1, 2].set_title('Yaw/Heading (Unwrapped)')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    save_dir = os.path.dirname(output_png)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"图表已保存至: {output_png}")
    if show_figure:
        plt.show()
    plt.close(fig)

def precompute_uuv_wave_height_vectorized(x_arr, y_arr, t_arr, wave_df):
    """
    [CPU加速] 利用Numpy广播机制，一次性计算轨迹上所有时间点的波高
    比逐帧计算快 100 倍以上
    """
    print("正在预计算 UUV 处波高 (Vectorized)...")
    
    # 1. 调整数据形状以利用广播
    # 轨迹数据形状: (N_frames, 1)
    X = x_arr[:, np.newaxis]
    Y = y_arr[:, np.newaxis]
    T = t_arr[:, np.newaxis]
    
    # 波浪参数形状: (1, N_components)
    amps = wave_df['amplitude'].values[np.newaxis, :]
    omegas = wave_df['omega'].values[np.newaxis, :]
    ks = wave_df['k'].values[np.newaxis, :]
    phases = wave_df['phase'].values[np.newaxis, :]
    thetas = wave_df['theta'].values[np.newaxis, :]
    
    # 2. 矩阵运算 (N_frames x N_components)
    # k * (x*cos(theta) + y*sin(theta))
    k_dot_pos = ks * (X * np.cos(thetas) + Y * np.sin(thetas))
    
    # Phase = k.x - w.t + phi
    phase_matrix = k_dot_pos - (omegas * T) + phases
    
    # 3. 计算 Cos 并求和 (沿 component 轴)
    # 结果形状: (N_frames, )
    elevation_array = np.sum(amps * np.cos(phase_matrix), axis=1)
    
    return elevation_array

def render_video():
    print("read data:")
    try:
        wave_params = pd.read_csv(WAVE_FILE, nrows=1)
        wave_groups = pd.read_csv(WAVE_FILE, skiprows=2)
        uuv_trace_raw = pd.read_csv(TRAJ_FILE, sep=r'[,]+', engine='python')
        uuv_trace_raw = uuv_trace_raw.dropna(axis=1, how='all')
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # ================= [关键优化1] 数据重采样 =================
    # 计算原始数据的FPS
    raw_time = uuv_trace_raw['t'].values
    if len(raw_time) > 1:
        raw_dt = np.mean(np.diff(raw_time))
        raw_fps = 1.0 / raw_dt
    else:
        raw_fps = TARGET_FPS

    # 如果原始FPS远大于目标FPS，进行降采样
    if raw_fps > TARGET_FPS * 1.5:
        step = int(raw_fps / TARGET_FPS)
        print(f"检测到高频数据 ({raw_fps:.1f} FPS)。正在降采样至约 {TARGET_FPS} FPS (步长: {step})...")
        uuv_trace = uuv_trace_raw.iloc[::step].reset_index(drop=True)
        actual_fps = raw_fps / step
    else:
        uuv_trace = uuv_trace_raw
        actual_fps = raw_fps
        print(f"数据频率正常 ({actual_fps:.1f} FPS)，无需降采样。")

    sim_time = uuv_trace['t'].values[-1]
    print(f"渲染总时长: {sim_time:.2f}s, 渲染帧数: {len(uuv_trace)}")

    # 1. 读取 UUV 模型
    try:
        uuv_mesh = pv.read(MESH_FILE)
    except:
        uuv_mesh = pv.Cylinder(radius=0.3, height=4.0, direction=(1,0,0))
    
    # 2. 创建波浪网格
    # ================= [关键优化2] 降低网格密度 =================
    buffer = 20.0 
    g_dx, g_dy = 1.5, 1.5  # [修改] 0.5 -> 1.5 (网格点数减少约9倍，大幅提升速度)
    
    min_x, max_x = min(uuv_trace['x'].values)-buffer, max(uuv_trace['x'].values)+buffer
    min_y, max_y = min(uuv_trace['y'].values)-buffer, max(uuv_trace['y'].values)+buffer
    x_range = np.arange(min_x, max_x, g_dx)
    y_range = np.arange(min_y, max_y, g_dy)
    global_plot_X, global_plot_Y = np.meshgrid(x_range, y_range)
    print(f"网格生成完毕 (Points: {global_plot_X.size})")

    # 预计算波浪 (传入重采样后的时间数组)
    global_wave_heights_cache = {}
    wave_init(global_plot_X, global_plot_Y, wave_groups, global_wave_heights_cache, uuv_trace['t'].values)

    # 4. 初始化 Plotter
    plotter = pv.Plotter(shape=(1, 2), off_screen=False, window_size=[1600, 900])
    plotter.set_background("lightblue")
    plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)

    z_zeros = np.zeros_like(global_plot_X)
    temp_grid = pv.StructuredGrid(global_plot_X, global_plot_Y, z_zeros)
    water_mesh = temp_grid.extract_surface()
    water_mesh.point_data["Elevation"] = -z_zeros.ravel(order='C')

    z_lim = wave_params['Hs'][0] * 0.6
    color_list = ["#001030", "#003060", "#0060A0", "#0080FF", "#80C0FF"]
    my_smooth_cmap = LinearSegmentedColormap.from_list("deep_ocean_smooth", color_list, N=256)

    water_props = dict(
        scalars="Elevation", cmap=my_smooth_cmap, opacity=0.75,
        clim=[-z_lim, z_lim], show_scalar_bar=False, smooth_shading=True,
        specular=0.5, specular_power=60.0, ambient=0.4, diffuse=0.8, lighting=True
    )

    # Subplot 0
    plotter.subplot(0, 0)
    plotter.add_mesh(water_mesh, **water_props)
    body_actor = plotter.add_mesh(uuv_mesh, color='orange', smooth_shading=True)
    plotter.add_axes(line_width=5, labels_off=False)
    plotter.show_grid(color='black', font_size=10, location='outer', ticks='outside')

    # Subplot 1
    plotter.subplot(0, 1)
    plotter.add_mesh(water_mesh, **water_props)
    global_body_actor = plotter.add_mesh(uuv_mesh, color='orange', smooth_shading=True)
    plotter.add_axes(line_width=5, labels_off=False)
    plotter.show_grid(color='black', location='outer')
    
    plotter.remove_all_lights()
    light_top = pv.Light(position=(0, 0, -100), focal_point=(0, 0, 0), color='white', intensity=0.7)
    plotter.add_light(light_top)
    plotter.enable_lightkit()

    plotter.open_movie(OUTPUT_MP4, framerate=actual_fps, quality=9) # 质量略微调低一点点以加速写入，一般看不出来
    
    # 提取轨迹数据 (已重采样)
    t_vals = uuv_trace['t'].values
    x_vals = uuv_trace['x'].values
    y_vals = uuv_trace['y'].values
    z_vals = uuv_trace['z'].values
    phi_vals = uuv_trace['roll'].values
    theta_vals = uuv_trace['pitch'].values
    psi_vals = uuv_trace['heading'].values
    
    relative_offset = np.array([-15.0, -15.0, -8.0])
    deadzone_radius = 5.0  
    smooth_factor = 0.1
    current_cam_pos = np.array([x_vals[0], y_vals[0], z_vals[0]]) + relative_offset
    current_focal_point = np.array([x_vals[0], y_vals[0], z_vals[0]])
    
    # ================= [新增] 预计算速度和波浪特征 =================
    # 1. 计算 UUV 速度 (差分法)
    # 既然已经重采样了，dt 就是恒定的
    dt = 1.0 / actual_fps # 使用实际的帧率计算 dt
    uuv_wave_heights_array = precompute_uuv_wave_height_vectorized(
        x_vals, y_vals, t_vals, wave_groups
    )
    
    # 计算速度向量 (m/s)
    vx_vals = np.gradient(x_vals, dt)
    vy_vals = np.gradient(y_vals, dt)
    vz_vals = np.gradient(z_vals, dt)
    u_vals = uuv_trace['u'].values
    v_vals = uuv_trace['v'].values
    w_vals = uuv_trace['w'].values
    p_vals = uuv_trace['p'].values
    q_vals = uuv_trace['q'].values
    r_vals = uuv_trace['r'].values

    wp_Hs = wave_params['Hs'].iloc[0]
    wp_T1 = wave_params['T1'].iloc[0]
    wp_theta = wave_params['main_theta'].iloc[0]
    # wave_type 可能是字符串，处理一下
    wp_type = wave_params['wave_type'].iloc[0] 
    
    g=9.81
    d=wave_params['depth'].iloc[0] 
    PI=math.pi
    L0 = (g/(2*PI))*wp_T1*wp_T1
    print("L0=",L0)
    if(d/L0>0.5):
        L=L0
    else:
        omega=2*PI/wp_T1
        L_est=L0
        tolerance = 1e-6
        for i in range(100):
            k = 2 * PI / L_est
            F = omega * omega - g * k * math.tanh(k * d)
            dF_dk = -g * (math.tanh(k * d) + k * d * (1 - math.tanh(k * d) * math.tanh(k * d)))
            dk = -F / dF_dk  # Newton-Raphson法迭代增量
            k += dk
            L_est = 2 * PI / k
            if (abs(dk / k) < tolerance): break
        L = L_est
    # 3. 波速 C = L / T
    C = L / wp_T1


    # 渲染循环
    # [优化] 使用 enumerate 直接获取索引，避免浮点数计算误差
    for frame_idx in tqdm(range(len(uuv_trace)), unit="frame", desc="Rendering Video"):
        x, y, z = x_vals[frame_idx], y_vals[frame_idx], z_vals[frame_idx]
        phi, theta, psi = phi_vals[frame_idx], theta_vals[frame_idx], psi_vals[frame_idx]
        
        # 更新波浪
        if frame_idx in global_wave_heights_cache:
            z_values = global_wave_heights_cache[frame_idx]
            water_mesh.points[:, 2] = -z_values.ravel()
            water_mesh.point_data["Elevation"] = z_values.ravel()
            
            # [关键优化3] 优化 compute_normals 参数
            # cell_normals=False: 不需要计算面的法线，只需顶点的
            # split_vertices=False: 不分割顶点，大幅减少计算量
            water_mesh.compute_normals(cell_normals=False, split_vertices=False, inplace=True) 
        
        # 更新位置
        rotation = R.from_euler('zyx', [psi, theta, phi])
        rot_matrix = rotation.as_matrix()
        transform_mat = np.eye(4)
        transform_mat[:3, :3] = rot_matrix
        transform_mat[:3, 3] = [x, y, z] 

        body_actor.user_matrix = transform_mat
        global_body_actor.user_matrix = transform_mat

        ship_pos = np.array([x,y,z])
        plotter.subplot(0, 0)
        ideal_cam_pos = ship_pos + relative_offset
        diff_vec = ideal_cam_pos - current_cam_pos
        distance = np.linalg.norm(diff_vec)

        if distance > deadzone_radius:
            move_vec = diff_vec * smooth_factor
            current_cam_pos += move_vec
        
        focal_diff = ship_pos - current_focal_point
        current_focal_point += focal_diff
            
        plotter.camera.position = tuple(current_cam_pos)
        plotter.camera.focal_point = tuple(current_focal_point)
        plotter.camera.up = (0, 0, -1)
            
        if frame_idx == 0:
            plotter.subplot(0, 1)
            plotter.render()
            cam_pos = plotter.camera.position
            cam_foc = plotter.camera.focal_point
            plotter.camera.position = (-cam_pos[0]*0.6, -cam_pos[1]*0.6, -cam_pos[2]*0.6)
            plotter.camera.focal_point = cam_foc
            plotter.camera.up = (0, 0, -1)
            plotter.reset_camera_clipping_range()

        # 1. 计算当前时刻船所在位置的实时波高 (Relative Wave Elevation)
        curr_t = t_vals[frame_idx]
        uuv_wave_h = uuv_wave_heights_array[frame_idx]
        
        vx=vx_vals[frame_idx]
        vy=vy_vals[frame_idx]
        vz=vz_vals[frame_idx]
        u=u_vals[frame_idx]
        v=v_vals[frame_idx]
        w=w_vals[frame_idx]
        p=p_vals[frame_idx]*57.3
        q=q_vals[frame_idx]*57.3
        r=r_vals[frame_idx]*57.3
        
        hud_text = (
            f"Local View (Follow)\n"
            f"TIME : {curr_t:06.2f} s\n"
            f"---------------------\n"
            f"World_Pos     : N {x:6.1f} | E {y:6.1f} | D {z:6.1f} m\n"
            f"World_Speed   : N {vx:6.1f} | E {vy:6.1f} | D {vz:6.1f} m\n"
            f"\n"
            f"Wave_Pos      : N {x:6.1f} | E {y:6.1f} | D {uuv_wave_h:6.1f} m\n"
            f"Self_Pos      : N {x:6.1f} | E {y:6.1f} | D {z:6.1f} m\n"
            f"\n"
            f"Self_Deg      : R {np.degrees(phi):6.1f} | P {np.degrees(theta):6.1f} | Y {np.degrees(psi):6.1f} deg\n"
            f"\n"
            f"Self_Speed    : u {u:6.1f} | v {v:6.1f} | w {w:6.1f} m/s\n"
            f"Self_Speed_Deg: p {p:6.1f} | q {q:6.1f} | r {r:6.1f} deg/s\n"
            f"---------------------\n"
        )
        
        # 3. 添加到左侧视图 (subplot 0)
        plotter.subplot(0, 0)
        plotter.add_text(
            hud_text, 
            position='upper_left', 
            font_size=12,       # 字体大小
            color='white',      # 字体颜色
            font='courier',     # 使用等宽字体，数字不跳动
            name='hud_status',  # [关键] 设置固定名字，实现更新而非覆盖
            shadow=True         # 添加阴影，在浅色背景下也能看清
        )
        
        # (可选) 在右侧全局视图显示简略信息
        plotter.subplot(0, 1)
        global_msg = (
            f"Global View\n"
            f"---------------------\n"
            f"Wave_Params:\n"
            f"  Hs              :{wp_Hs:6.1f} m\n"
            f"  T1              :{wp_T1:6.1f} s\n"
            f"  main_theta      :{wp_theta:6.1f} deg\n"
            f"  wave_type       :{str(wp_type)}\n"
            f"  wave length L   :{L:6.1f} m\n"
            f"  wave velocity C :{C:6.1f} m/s\n"
            f"\n"
            f"Target : N {x:6.1f} | E {y:6.1f} | D {z:6.1f} m\n"
            f"Wave   : N {x:6.1f} | E {y:6.1f} | D {uuv_wave_h:6.1f} m\n"
            )
        plotter.add_text(
            global_msg, 
            position='upper_left', 
            font_size=12,       # 字体大小
            color='white',      # 字体颜色
            font='courier',     # 使用等宽字体，数字不跳动
            name='hud_status',  # [关键] 设置固定名字，实现更新而非覆盖
            shadow=True         # 添加阴影，在浅色背景下也能看清
        )

        # --- HUD 更新逻辑结束 ---

        plotter.write_frame()

    plotter.close()
    print(f"视频已保存至: {OUTPUT_MP4}")

if __name__ == "__main__":
    plot_6dof_2x3()
    render_video()