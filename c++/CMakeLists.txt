cmake_minimum_required(VERSION 3.13)
project(histogram_equalizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
set(CUDA_ARCHITECTURES 86)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(histogram_equalizer main.cu)

target_link_libraries(histogram_equalizer CUDA::cudart ${OpenCV_LIBS})
