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

# v0
add_executable(fastatomicadd_baseline
    fastatomicadd_baseline.cu
)

# v1
add_executable(fastatomicadd_v1
    fastatomicadd_v1.cu
)
