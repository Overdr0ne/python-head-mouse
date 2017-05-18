cmake_minimum_required(VERSION 2.8.12)

PROJECT(head-pose)

SET(GCC_COVERAGE_LINK_FLAGS "-lX11 -mavx")
SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )

find_package( OpenCV REQUIRED )
find_package( dlib REQUIRED )

include_directories(${dlib_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
if(${OpenCV_VERSION} VERSION_GREATER 2.9.0)
    set(OPENCV3 TRUE)
    add_definitions(-DOPENCV3)
endif()

add_definitions(-mavx)

ADD_EXECUTABLE(webcam_head_mouse webcam_head_mouse.cpp)

TARGET_LINK_LIBRARIES(webcam_head_mouse ${dlib_LIBRARIES} ${OpenCV_LIBS})

# This is a CMake file meant to be included via include()
# It will trigger a compilation of dlib *in the project* 
# including it

set(DLIB_IN_PROJECT_BUILD true)

if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()


# Determine the path to dlib.
#set(dlib_path ${CMAKE_CURRENT_LIST_DIR})
#include(${dlib_path}/cmake_utils/add_global_compiler_switch.cmake)
#include(${dlib_path}/cmake_utils/use_cpp_11.cmake)

if (CMAKE_COMPILER_IS_GNUCXX)
    # By default, g++ won't warn or error if you forget to return a value in a
    # function which requires you to do so.  This option makes it give a warning
    # for doing this.
    add_global_compiler_switch(-Wreturn-type)
endif()

set(gcc_like_compilers GNU Clang  Intel)
set(intel_archs x86_64 i386 i686)


# Setup some options to allow a user to enable SSE and AVX instruction use.  
if ((";${gcc_like_compilers};" MATCHES ";${CMAKE_CXX_COMPILER_ID};")  AND
    (";${intel_archs};"        MATCHES ";${CMAKE_SYSTEM_PROCESSOR};") AND NOT USE_AUTO_VECTOR)
    option(USE_SSE2_INSTRUCTIONS "Compile your program with SSE2 instructions" OFF)
    option(USE_SSE4_INSTRUCTIONS "Compile your program with SSE4 instructions" OFF)
    option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX instructions"  ON)
    if(USE_AVX_INSTRUCTIONS)
        add_definitions(-mavx)
        message(STATUS "Enabling AVX instructions")
    elseif (USE_SSE4_INSTRUCTIONS)
        add_definitions(-msse4)
        message(STATUS "Enabling SSE4 instructions")
    elseif(USE_SSE2_INSTRUCTIONS)
        add_definitions(-msse2)
        message(STATUS "Enabling SSE2 instructions")
    endif()
endif()

macro (toggle_preprocessor_switch option_name)
   if (${option_name})
      add_global_define(${option_name})
   else()
      remove_global_define(${option_name})
   endif()
endmacro()

option(DLIB_USE_BLAS ${DLIB_USE_BLAS_STR} ON)
option(DLIB_USE_LAPACK ${DLIB_USE_LAPACK_STR} ON)

toggle_preprocessor_switch(DLIB_USE_BLAS)
toggle_preprocessor_switch(DLIB_USE_LAPACK)

if (DLIB_USE_BLAS OR DLIB_USE_LAPACK)
    # Try to find BLAS, LAPACK and MKL
    include(cmake_find_blas.txt)

    if (DLIB_USE_BLAS)
        if (blas_found)
            set (dlib_needed_libraries ${dlib_needed_libraries} ${blas_libraries})
        else()
            set(DLIB_USE_BLAS OFF CACHE STRING ${DLIB_USE_BLAS_STR} FORCE )
            toggle_preprocessor_switch(DLIB_USE_BLAS)
        endif()
    endif()

    if (DLIB_USE_LAPACK)
        if (lapack_found)
            set (dlib_needed_libraries ${dlib_needed_libraries} ${lapack_libraries})
        else()
            set(DLIB_USE_LAPACK OFF CACHE STRING ${DLIB_USE_LAPACK_STR} FORCE )
            toggle_preprocessor_switch(DLIB_USE_LAPACK)
        endif()
    endif()
endif()

set(CMAKE_BUILD_TYPE "Release")

# Don't add dlib if it's already been added to the cmake project
#if (NOT TARGET dlib)
    #add_subdirectory(${dlib_path} dlib_build)
#endif()
