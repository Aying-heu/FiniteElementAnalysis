# 固定随机种子（可选，保证结果可复现）
import numpy as np
np.random.seed(42)

pi = np.pi
data_setting_group = []

# 生成50组参数
for _ in range(50):
    # 1. 生成Hs：0.1~5，集中0.8~3（截断正态分布）
    while True:
        Hs = np.random.normal(loc=1.9, scale=3)  # 均值1.9，标准差0.7
        if 0.1 <= Hs <= 5:
            break
    
    # 2. 生成T1：随Hs调整（经验关系+扰动）
    T1_base = 2.5 * np.sqrt(Hs)  # 波高-周期经验公式
    T1 = T1_base + np.random.uniform(-0.3, 0.3)  # 小扰动
    T1 = max(0.1, T1)  # 保证T1≥0.1
    
    # 3. 生成tau_x0：0~50，集中0~10（Beta分布偏态采样）
    # Beta分布(α=1, β=5)偏向左，转换到0~50
    tau_x0_raw = np.random.beta(a=1, b=5)
    tau_x0 = tau_x0_raw * 50  # 映射到0~50
    tau_x0 = round(tau_x0, 1)  # 保留1位小数
    
    # 4. 生成tau_n0：-5~5，集中-1~1，且tau_x0越小tau_n0范围越窄
    if tau_x0 <= 10:
        # tau_x0小（0~10），tau_n0限制在-1~1
        tau_n0 = np.random.uniform(-1, 1)
    elif tau_x0 <= 30:
        # tau_x0中等（10~30），tau_n0限制在-3~3
        tau_n0 = np.random.uniform(-3, 3)
    else:
        # tau_x0大（30~50），tau_n0放宽到-5~5
        tau_n0 = np.random.uniform(-5, 5)
    tau_n0 = round(tau_n0, 1)  # 保留1位小数
    
    # 5. 生成main_theta：-π ~ π 均匀分布
    main_theta = np.random.uniform(-pi, pi)
    main_theta = round(main_theta, 2)  # 保留2位小数
    
    # 整理参数（Hs/T1保留1位小数，保证可读性）
    param = {
        'Hs': round(Hs, 1),
        'T1': round(T1, 1),
        'tau_x0': tau_x0,
        'tau_n0': tau_n0,
        'main_theta': main_theta
    }
    data_setting_group.append(param)



print("\n全部50组参数列表：")
for idx, param in enumerate(data_setting_group, 1):
    print(f"{idx}: {param}")