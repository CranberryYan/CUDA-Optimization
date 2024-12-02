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
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/15_gemv
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

# v0
add_executable(reduce_baseline
    reduce_baseline.cu
)

# v1
add_executable(reduce_v1
    reduce_v1_interleaved_addressing.cu
)

# v2
add_executable(reduce_v2
    reduce_v2_bank_conflict.cu
)

# v3
add_executable(reduce_v3
    reduce_v3_Idel_thread.cu
)

# v4
add_executable(reduce_v4
    reduce_v4_unrolling_the_last_warp.cu
)

# v5
add_executable(reduce_v5
    reduce_v5_complete_unrolling.cu
)

# v6.0
add_executable(reduce_v6.0
    reduce_v6.0_grid&block_size.cu
)

# v6.1
add_executable(reduce_v6.1
    reduce_v6.1_grid&block_size.cu
)

# v6.2
add_executable(reduce_v6.2
    reduce_v6.2_grid&block_size.cu
)

# v7
add_executable(reduce_v7
    reduce_v7_shfl_down_sync.cu
)

# v8
add_executable(reduce_v8
    reduce_v8_half.cu
)