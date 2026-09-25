cmake_minimum_required(VERSION 3.24)
project(GPUCanonicalLabeling LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(gpu_canon_label
    main.cpp
    src/partition_refinement.cpp
    src/perm_group.cpp
    src/sequential.cpp
    src/CSRGraph.cpp
    src/nauty_parallel.cu
)

target_include_directories(gpu_canon_label PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(gpu_canon_label PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xptxas=--maxrregcount=32
    >
)