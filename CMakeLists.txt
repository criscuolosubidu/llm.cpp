cmake_minimum_required(VERSION 4.2)
project(llm_cpp CUDA)

if (MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
    add_compile_options($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>)
endif()

set(CMAKE_CUDA_STANDARD 23)

add_executable(llm_cpp main.cu
        main.cu)

add_executable(dev_cuda_kernels dev/adamw.cu)

set_target_properties(llm_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
