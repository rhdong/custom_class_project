cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)

# Define our library target
# add_library(custom_class SHARED class.cpp)
# set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
# target_link_libraries(custom_class "${TORCH_LIBRARIES}")

# Select target CUDA binary architecture.
foreach(cuda_arch ${sm})
  list(APPEND cuda_arch_list ${cuda_arch})
  message(STATUS "Assign GPU architecture (sm=${cuda_arch})")
endforeach()
list(LENGTH cuda_arch_list cuda_arch_list_length)
if(cuda_arch_list_length EQUAL 0)
  list(APPEND cuda_arch_list "80")
  message(STATUS "Assign default GPU architecture sm=80")
endif()
foreach(cuda_arch ${cuda_arch_list})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${cuda_arch},code=sm_${cuda_arch}")
endforeach()

set(ENV{TORCH_CUDA_ARCH_LIST} "Ampere")

find_package(Torch REQUIRED)

# Define our library target
add_library(merlin_kv SHARED merlin-kv.cu)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(merlin_kv "${TORCH_LIBRARIES}")

set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
