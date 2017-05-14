cmake_minimum_required(VERSION 2.8.12)

PROJECT(head-pose)

SET(GCC_COVERAGE_LINK_FLAGS "-lX11")
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

ADD_EXECUTABLE(webcam_head_mouse webcam_head_mouse.cpp)

TARGET_LINK_LIBRARIES(webcam_head_mouse ${dlib_LIBRARIES} ${OpenCV_LIBS})
