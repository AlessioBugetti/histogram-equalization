cmake_minimum_required(VERSION 3.30)
project(histogram_equalizer)

set(CMAKE_CXX_STANDARD 20)

add_executable(histogram_equalizer main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(histogram_equalizer ${OpenCV_LIBS})