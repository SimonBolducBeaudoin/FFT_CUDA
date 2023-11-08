# This won't work on cygwin because CUDA is not available on cygwin.
cmake_minimum_required(VERSION 3.21)

project(fftcuda LANGUAGES CXX CUDA VERSION 2.2.0)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)

file(GLOB SRC_FILES "src/*.cpp")

if(MSVC)
    # add_compile_options(/W4 /Ox /arch:AVX2 /showIncludes)
    add_compile_options(/W4 /Ox /arch:AVX2)
else()#(UNIX)
    add_compile_options(-Wall -O3 -march=native -MMD -MP)
endif()

add_library(fftcuda ${SRC_FILES})
#set_target_properties(fftcuda PROPERTIES CUDA_ARCHITECTURES 60)
set_target_properties(fftcuda PROPERTIES CUDA_ARCHITECTURES native)

include(CMakeConfigs/pybind11.cmake) # sets IS_MINGW,PY_INCL,SHARED_LIB_PREFIX,SHARED_LIB_SUFFIX and updates LINKS
include_directories(${PY_INCL}) # pybind11 includes
set_target_properties(fftcuda PROPERTIES PREFIX "${SHARED_LIB_PREFIX}")
set_target_properties(fftcuda PROPERTIES SUFFIX ${SHARED_LIB_SUFFIX})

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

if(MSVC)
    target_link_libraries(fftcuda ${LINKS})
    find_library(CUFFT NAMES cufft HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(fftcuda ${CUFFT})
    find_library(CUDART NAMES cudart HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(fftcuda ${CUDART})
    find_library(CUDA NAMES cuda HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(fftcuda ${CUDA})
else()
    target_link_libraries(fftcuda -L/opt/cuda/lib64 -lcufft -lcudart -lcuda)
endif()

if(MSVC)
    set(DIR ${PROJECT_SOURCE_DIR}/obj/Release) 
else()
    set(DIR ${PROJECT_SOURCE_DIR}/obj/) 
endif()

set(INSTALL_DIR ${PROJECT_SOURCE_DIR})

# Install does'nt work properly in visual studio. Why ? Idk ask them.
install(DIRECTORY ${DIR} DESTINATION ${INSTALL_DIR} FILES_MATCHING PATTERN "*.pyd" PATTERN "*.so" )
