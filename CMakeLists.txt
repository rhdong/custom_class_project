cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)

find_package(Torch REQUIRED)

# Define our library target
# add_library(custom_class SHARED class.cpp)
# set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
# target_link_libraries(custom_class "${TORCH_LIBRARIES}")

# Define our library target
add_library(merlin_kv SHARED merlin-kv.cu)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(merlin_kv "${TORCH_LIBRARIES}")

set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
