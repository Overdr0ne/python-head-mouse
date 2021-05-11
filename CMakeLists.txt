cmake_minimum_required(VERSION 2.8.12)

PROJECT(head-pose)

add_subdirectory(/home/sam/src/dlib/dlib-19.16/dlib dlib_build)

if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the webcam_face_pose_ex example.")
else()
   find_package(VTK REQUIRED)
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})
      include(${VTK_USE_FILE})

      add_executable(webcam_head_mouse webcam_head_mouse.cpp)
      #target_link_libraries(webcam_head_mouse dlib::dlib ${OpenCV_LIBS})
      target_link_libraries(webcam_head_mouse dlib::dlib ${OpenCV_LIBS} ${VTK_LIBRARIES})
   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()

