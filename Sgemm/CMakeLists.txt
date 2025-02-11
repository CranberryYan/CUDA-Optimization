cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDATutorial LANGUAGES CXX CUDA)

find_package(CUDA 11.8 REQUIRED)
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

# set compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

#输出的可执行文件保存地址
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/app)

set(COMMON_HEADER_DIRS
    ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
    ${CUDA_PATH}/lib64
)
#指定头文件路径
include_directories(
    ${COMMON_HEADER_DIRS}
)
#指定待链接的lib路径
link_directories(
    ${COMMON_LIB_DIRS}
)

# baseline
add_executable(Sgemm_baseline
    Sgemm_baseline.cu
)

# v0
add_executable(Sgemm_v0
    Sgemm_v0.cu
)

# v1
add_executable(Sgemm_v1
    Sgemm_v1.cu
)

# v2
add_executable(Sgemm_v2
    Sgemm_v2.cu
)

# v2_2
add_executable(Sgemm_v2_2
    Sgemm_v2_2.cu
)

# v2_3
add_executable(Sgemm_v2_3
    Sgemm_v2_3.cu
)

# v3
add_executable(Sgemm_v3
    Sgemm_v3.cu
)

# v4
add_executable(Sgemm_v4
    Sgemm_v4.cu
)

# v4_2
add_executable(Sgemm_v4_2
    Sgemm_v4_2.cu
)

# v5
add_executable(Sgemm_v5
    Sgemm_v5.cu
)

# v6
add_executable(Sgemm_v6
    Sgemm_v6.cu
)

# v7
add_executable(Sgemm_v7
    Sgemm_v7.cu
)

# v8
add_executable(Sgemm_v8
    Sgemm_v8.cu
)