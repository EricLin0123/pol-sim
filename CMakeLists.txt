cmake_minimum_required(VERSION 3.24)
project(pol_sim LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
foreach(target thermal em)
    add_executable(${target} src/${target}.cu)
    target_link_libraries(${target} PRIVATE CUDA::cudart)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
endforeach()
