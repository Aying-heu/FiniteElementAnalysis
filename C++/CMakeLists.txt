# cmake_minimum_required(VERSION 3.10)

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# set(CMAKE_CUDA_ARCHITECTURES 75)

# project(UUV_Model LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
# set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# include_directories(../include
# 		    ../include/Common
# 		    ../loadlog_utils/include)
			
# include_directories(/usr/include/eigen3)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp_solver)

# file(GLOB CPP_SOURCE_FILES "cpp_solver/*gpu.cpp" "cpp_solver/*ull.cpp")
# file(GLOB CUDA_SOURCE_FILES "cpp_solver/*.cu")

# aux_source_directory(. ROOT_SOURCE_FILES)
# aux_source_directory(../src REL_SOURCE_FILES)
# aux_source_directory(../loadlog_utils/src LOG_SOURCE_FILES)

# # 合并所有源文件
# set(ALL_SOURCE_FILES 
#     ${CPP_SOURCE_FILES} 
#     ${CUDA_SOURCE_FILES} 
#     ${ROOT_SOURCE_FILES} 
#     ${REL_SOURCE_FILES} 
#     ${LOG_SOURCE_FILES}
# )

# find_package(OpenMP REQUIRED)
# find_package(CUDA REQUIRED)

# MESSAGE("Source files found: ${ALL_SOURCE_FILES}")


# add_library(wave_cuda_lib STATIC
#     cpp_solver/wave_cuda.cu  # 你的 cu 文件
# )
# # 必须为 CUDA 库启用 Position Independent Code (如果它是静态库而被动态库链接)
# set_property(TARGET wave_cuda_lib PROPERTY POSITION_INDEPENDENT_CODE ON)


# # 5. 添加可执行文件，把 .cu 文件也放进去
# add_executable(${PROJECT_NAME} UUV_Model.cc ${ALL_SOURCE_FILES})

# # 6. 链接库
# # 现代 CMake 启用了 LANGUAGES CUDA 后，通常会自动链接 cudart
# # 但为了保险，可以查找并链接 CUDAToolkit (CMake 3.17+) 或者直接依赖编译器行为
# target_link_libraries(${PROJECT_NAME} PRIVATE 
#     pthread 
#     rt 
#     OpenMP::OpenMP_CXX
#     wave_cuda_lib
#     ${CUDA_LIBRARIES}
# )

# # 链接文件系统库
# find_library(STDCXXFS NAMES stdc++fs)
# if(STDCXXFS)
#     target_link_libraries(${PROJECT_NAME} ${STDCXXFS})
# endif()

# INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/../exe/)

cmake_minimum_required(VERSION 3.10)

# =============================================================
# 1. 强制指定编译器 (必须在 project 之前)
# =============================================================
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# 2. 架构设置
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(UUV_Model LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================
# 3. 关键修复：通过 Flag 指定使用静态 CUDA 运行时
# 这样 nvcc 就知道只链接静态库，不会重复链接动态库了
# =============================================================
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -cudart=static")
# 如果你是 Debug 模式，也要加上
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -cudart=static")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# 强制优先搜索 /usr/local/cuda 的头文件
include_directories(SYSTEM BEFORE "/usr/local/cuda/include")

include_directories(include
		    include/Common
		    loadlog_utils/include
            /usr/include/eigen3
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp_solver)

# 4. 源文件收集
file(GLOB CPP_SOURCE_FILES "cpp_solver/*gpu.cpp" "cpp_solver/*ull.cpp")

aux_source_directory(. ROOT_SOURCE_FILES)
aux_source_directory(src REL_SOURCE_FILES)
aux_source_directory(loadlog_utils/src LOG_SOURCE_FILES)

# =============================================================
# 5. 编译可执行文件
# =============================================================
add_executable(${PROJECT_NAME} 
    UUV_Model.cc 
    ${CPP_SOURCE_FILES}
    ${ROOT_SOURCE_FILES}
    ${REL_SOURCE_FILES}
    ${LOG_SOURCE_FILES}
    cpp_solver/wave_cuda.cu 
)

find_package(OpenMP REQUIRED)

# =============================================================
# 6. 链接
# =============================================================
target_link_libraries(${PROJECT_NAME} PRIVATE 
    pthread 
    rt 
    dl # 静态链接 cudart 必须依赖 dl
    OpenMP::OpenMP_CXX
    # 【注意】这里删除了 libcudart_static.a，交给编译器标志处理
)

# 强制使用 CUDA 链接器
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CUDA)

# 文件系统库兼容
find_library(STDCXXFS NAMES stdc++fs)
if(STDCXXFS)
    target_link_libraries(${PROJECT_NAME} ${STDCXXFS})
endif()

INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/../exe/)