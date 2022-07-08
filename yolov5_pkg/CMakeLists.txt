cmake_minimum_required(VERSION 3.0.2)
project(yolov5_pkg)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        cv_bridge
        )

find_package(OpenCV REQUIRED)

catkin_package(
        CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
        include
        lib
        ${catkin_INCLUDE_DIRS}
)

catkin_package(
        CATKIN_DEPENDS   roscpp rospy std_msgs
        DEPENDS EIGEN3  OpenCV
        INCLUDE_DIRS
)
#用于编译一个main程序
#file(GLOB SOURCE_FILES src/*.cpp)
#
#add_executable(${PROJECT_NAME}_yolo ${SOURCE_FILES})
#
#target_link_libraries (
#        ${PROJECT_NAME}_yolo
#        ${catkin_LIBRARIES}
#        ${OpenCV_LIBS}
#        pthread
#        MNN
#)

#用于编译多个main程序
#方法1：
file(GLOB SOURCE_FILES_1 src/main.cpp src/util.cpp src/Yolo.cpp)
add_executable(${PROJECT_NAME}_yolo ${SOURCE_FILES_1})
target_link_libraries (
        ${PROJECT_NAME}_yolo
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        pthread
        MNN
)


file(GLOB SOURCE_FILES_2 src/ros_main.cpp src/util.cpp src/Yolo.cpp)
add_executable(${PROJECT_NAME}_yolo_ros ${SOURCE_FILES_2})
target_link_libraries (
        ${PROJECT_NAME}_yolo_ros
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        pthread
        MNN
)



#方法2：
#file(GLOB_RECURSE mains RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/main*.cpp")
#foreach(mainfile IN LISTS mains)
#    # Get file name without directory
#    get_filename_component(mainname ${mainfile} NAME_WE)
#    add_executable(${mainname} ${mainfile})
#endforeach()
