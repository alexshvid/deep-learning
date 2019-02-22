
cmake_minimum_required(VERSION 3.0)
project(deepnn
  VERSION 3.5.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(ArrayFire)

if(ArrayFire_CPU_FOUND)
  add_executable(deepnn_cpu main.cpp deepnet.cpp deepnet.h activation.cpp activation.h)
  target_link_libraries(deepnn_cpu ArrayFire::afcpu)
endif()


if(ArrayFire_CUDA_FOUND)
  add_executable(deepnn_cuda main.cpp deepnet.cpp activation.cpp activation.h)
  target_link_libraries(deepnn_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(deepnn_opencl main.cpp deepnet.cpp activation.cpp activation.h)
  target_link_libraries(deepnn_opencl ArrayFire::afopencl)
endif()
