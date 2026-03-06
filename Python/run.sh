# #!/bin/bash
# # 批量运行四组 UUV 波浪仿真脚本
# # 赋予执行权限：chmod +x run_simulation.sh
# # 运行：./run_simulation.sh

# # ===================== 通用函数：计算并输出耗时 =====================
# # 传入参数：开始时间戳、任务名称
# print_elapsed_time() {
#     local start_time=$1
#     local task_name=$2
#     local end_time=$(date +%s.%N)
#     local elapsed=$(echo "$end_time - $start_time" | bc -l)  # bc计算浮点数
#     # 保留2位小数，输出友好格式
#     printf "\033[32m✅ %s 运行完成，耗时：%.2f 秒\033[0m\n\n" "$task_name" "$elapsed"
# }

# # ===================== 第一组 =====================
# echo -e "\033[34m=== 开始运行第一组仿真 ===\033[0m"
# Hs=0.3
# T1=3
# main_theta=0
# tau_x0=0
# t_max=600
# group="第一组"

# # 记录开始时间（新增）
# start_time=$(date +%s.%N)
# python3 小方块单个仿真_final.py \
#   --Hs $Hs \
#   --T1 $T1 \
#   --main_theta $main_theta \
#   --tau_x0 $tau_x0 \
#   --t_max $t_max \
#   --group "$group"
# # 输出耗时（新增）
# print_elapsed_time $start_time "第一组"

# # ===================== 第二组 =====================
# echo -e "\033[34m=== 开始运行第二组仿真 ===\033[0m"
# Hs=1.2
# T1=5
# main_theta=0
# tau_x0=0
# t_max=600
# group="第二组"

# # 记录开始时间（新增）
# start_time=$(date +%s.%N)
# python3 小方块单个仿真_final.py \
#   --Hs $Hs \
#   --T1 $T1 \
#   --main_theta $main_theta \
#   --tau_x0 $tau_x0 \
#   --t_max $t_max \
#   --group "$group"
# # 输出耗时（新增）
# print_elapsed_time $start_time "第二组"

# # ===================== 第三组（循环 main_theta 0~360°，步长20°） =====================
# echo -e "\033[34m=== 开始运行第三组仿真（循环波浪方向） ===\033[0m"
# Hs=0.8
# T1=4.5
# tau_x0=0
# t_max=180
# group="第三组"

# # 记录第三组总开始时间（新增）
# group3_total_start=$(date +%s.%N)
# # Shell 循环语法：0到360，步长20
# for main_theta in {0..360..20}; do
#   echo -e "\n  \033[36m运行第三组 - main_theta = $main_theta°...\033[0m"
#   # 记录单角度开始时间（新增）
#   single_start=$(date +%s.%N)
#   python3 小方块单个仿真_final.py \
#     --Hs $Hs \
#     --T1 $T1 \
#     --main_theta $main_theta \
#     --tau_x0 $tau_x0 \
#     --t_max $t_max \
#     --group "$group"
#   # 输出单角度耗时（新增）
#   print_elapsed_time $single_start "第三组-$main_theta°"
# done
# # 输出第三组总耗时（新增）
# print_elapsed_time $group3_total_start "第三组（所有角度）"

# # ===================== 第四组（循环 main_theta 0~360°，步长20°） =====================
# echo -e "\033[34m=== 开始运行第四组仿真（循环波浪方向） ===\033[0m"
# Hs=1.5
# T1=7
# tau_x0=0
# t_max=180
# group="第四组"

# # 记录第四组总开始时间（新增）
# group4_total_start=$(date +%s.%N)
# for main_theta in {0..360..20}; do
#   echo -e "\n  \033[36m运行第四组 - main_theta = $main_theta°...\033[0m"
#   # 记录单角度开始时间（新增）
#   single_start=$(date +%s.%N)
#   python3 小方块单个仿真_final.py \
#     --Hs $Hs \
#     --T1 $T1 \
#     --main_theta $main_theta \
#     --tau_x0 $tau_x0 \
#     --t_max $t_max \
#     --group "$group"
#   # 输出单角度耗时（新增）
#   print_elapsed_time $single_start "第四组-$main_theta°"
# done
# # 输出第四组总耗时（新增）
# print_elapsed_time $group4_total_start "第四组（所有角度）"

# # ===================== 全部完成 =====================
# echo -e "\033[32m🎉 所有仿真任务提交完成！\033[0m"














#!/bin/bash
# 批量多线程运行 UUV 波浪仿真脚本
# 赋予执行权限：chmod +x run_simulation.sh
# 运行：./run_simulation.sh

# ===================== 核心配置（关键！根据你的CPU/GPU调整） =====================
MAX_CONCURRENT=8  # 最大并发数（建议：CPU核心数/2 或 GPU数量，比如4/6/8）
LOG_DIR="./simulation_logs"  # 日志保存目录（避免多线程输出刷屏）
mkdir -p $LOG_DIR  # 自动创建日志目录

# ===================== 通用函数 =====================
# 1. 计算耗时函数
print_elapsed_time() {
    local start_time=$1
    local task_name=$2
    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc -l)
    printf "\033[32m✅ %s 运行完成，耗时：%.2f 秒\033[0m\n\n" "$task_name" "$elapsed"
}

# 2. 单任务运行函数（每个任务独立日志）
run_single_sim() {
    local Hs=$1
    local T1=$2
    local main_theta=$3
    local tau_x0=$4
    local t_max=$5
    local group=$6
    local task_name="${group}-main_theta=${main_theta}°"
    
    # 记录单任务开始时间
    local single_start=$(date +%s.%N)
    echo -e "\n  \033[36m开始运行 $task_name...\033[0m"
    
    # 运行Python脚本，输出重定向到独立日志（避免多线程输出混乱）
    python3 小方块单个仿真_final.py \
      --Hs $Hs \
      --T1 $T1 \
      --main_theta $main_theta \
      --tau_x0 $tau_x0 \
      --t_max $t_max \
      --group "$group" \
      > "${LOG_DIR}/${task_name}.log" 2>&1  # 日志保存到文件，2>&1表示错误也记录
    
    # 输出单任务耗时
    print_elapsed_time $single_start "$task_name"
}

# ===================== 第三组（多线程循环 main_theta 0~360°，步长20°） =====================
echo -e "\033[34m=== 开始运行第三组仿真（多线程，并发数：$MAX_CONCURRENT） ===\033[0m"
Hs=0.8
T1=4.5
tau_x0=0
t_max=180
group="第三组"
group3_total_start=$(date +%s.%N)

# 多线程循环：每次启动MAX_CONCURRENT个任务
count=0
for main_theta in {0..360..20}; do
  # 启动单任务（后台运行）
  run_single_sim $Hs $T1 $main_theta $tau_x0 $t_max $group &
  
  # 控制并发数：达到MAX_CONCURRENT时，等待一个任务完成再继续
  ((count++))
  if (( count % MAX_CONCURRENT == 0 )); then
    wait -n  # 等待任意一个后台任务完成，释放并发槽位
  fi
done
wait  # 等待第三组所有剩余任务完成
print_elapsed_time $group3_total_start "第三组（所有角度）"

# ===================== 第四组（多线程循环 main_theta 0~360°，步长20°） =====================
echo -e "\033[34m=== 开始运行第四组仿真（多线程，并发数：$MAX_CONCURRENT） ===\033[0m"
Hs=1.5
T1=7
tau_x0=0
t_max=180
group="第四组"
group4_total_start=$(date +%s.%N)

# 多线程循环
count=0
for main_theta in {0..360..20}; do
  run_single_sim $Hs $T1 $main_theta $tau_x0 $t_max $group &
  
  ((count++))
  if (( count % MAX_CONCURRENT == 0 )); then
    wait -n
  fi
done
wait  # 等待第四组所有剩余任务完成
print_elapsed_time $group4_total_start "第四组（所有角度）"

# ===================== 可选：恢复第一、二组（如需多线程运行） =====================
# echo -e "\033[34m=== 开始运行第一组仿真 ===\033[0m"
# Hs=0.3
# T1=3
# main_theta=0
# tau_x0=0
# t_max=600
# group="第一组"
# start_time=$(date +%s.%N)
# run_single_sim $Hs $T1 $main_theta $tau_x0 $t_max $group
# print_elapsed_time $start_time "第一组"

# echo -e "\033[34m=== 开始运行第二组仿真 ===\033[0m"
# Hs=1.2
# T1=5
# main_theta=0
# tau_x0=0
# t_max=600
# group="第二组"
# start_time=$(date +%s.%N)
# run_single_sim $Hs $T1 $main_theta $tau_x0 $t_max $group
# print_elapsed_time $start_time "第二组"

# ===================== 全部完成 =====================
echo -e "\033[32m🎉 所有仿真任务运行完成！日志保存在：$LOG_DIR\033[0m"
