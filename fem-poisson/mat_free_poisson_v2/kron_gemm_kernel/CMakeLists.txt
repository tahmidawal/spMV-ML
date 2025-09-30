cmake_minimum_required(VERSION 3.15)
project(FFI-KRON-GEMM-KERNEL LANGUAGES CXX)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr") 

if(N STREQUAL "" OR NOT DEFINED N)
    message(FATAL_ERROR "Matrix dimension N is not provided. (N = p_order + 1)")
endif()


message("Using N=${N}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(FATAL_ERROR "Please set CMAKE_INSTALL_PREFIX")
endif()

if(NOT JAXLIB_INCLUDE_DIR)
    message(FATAL_ERROR "Please set JAXLIB_INCLUDE_DIR")
endif()

add_library(kron_gemm_kernel SHARED "kron_gemm_kernel.cu")
target_compile_definitions(kron_gemm_kernel PRIVATE N=${N})
set_target_properties(kron_gemm_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                CUDA_STANDARD 17)
set_target_properties(kron_gemm_kernel PROPERTIES CUDA_ARCHITECTURES "native")
target_include_directories(kron_gemm_kernel PUBLIC ${JAXLIB_INCLUDE_DIR})
target_link_libraries(kron_gemm_kernel PRIVATE CUDA::cudart)
install(TARGETS kron_gemm_kernel LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})

