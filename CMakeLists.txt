cmake_minimum_required(VERSION 4.2)
project(llm_cpp CUDA CXX)

if (MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
    add_compile_options($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>)
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(llm_cpp main.cu
)

add_executable(dev_cuda_kernels dev/adamw.cu
        dev/utils.cuh)

set_target_properties(llm_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
