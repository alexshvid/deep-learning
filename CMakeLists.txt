
cmake_minimum_required(VERSION 3.0)
project(deepnn
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

if(ArrayFire_CPU_FOUND)
  add_executable(deepnn_cpu main.cpp)
  target_link_libraries(deepnn_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(deepnn_cuda main.cpp)
  target_link_libraries(deepnn_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(deepnn_opencl main.cpp)
  target_link_libraries(deepnn_opencl ArrayFire::afopencl)
endif()
