cmake_minimum_required(VERSION 3.5)
project(robosense_voxel_odom)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# memory test mode
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread -fexceptions -Wall -g")
# running mode
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread -fexceptions")

set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-DPROJECT_PATH="${PROJECT_SOURCE_DIR}")

find_package(Boost REQUIRED COMPONENTS date_time)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV 3 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 4 REQUIRED)
endif ()
message(STATUS "PCL: " ${PCL_VERSION} " | OPENCV: " ${OpenCV_VERSION})

set(ROS_VERSION "")
if(DEFINED ENV{ROS_DISTRO})
  set(ROS_DISTRO $ENV{ROS_DISTRO})
  message(STATUS "Found ROS_DISTRO: ${ROS_DISTRO}")

  if(ROS_DISTRO MATCHES "^(humble|foxy|galactic|iron|rolling)$")
    set(ROS_VERSION "2")
  elseif(ROS_DISTRO MATCHES "^(noetic|melodic|kinetic)$")
    set(ROS_VERSION "1")
  else()
    message(WARNING "ROS version not detected, defaulting to ROS1")
    set(ROS_VERSION "1")
  endif()
else()
  message(WARNING "ROS_DISTRO environment variable not set")
  set(ROS_VERSION "1")
endif()

if(ROS_VERSION STREQUAL "1")
  message(STATUS "================= Building for ROS1 =================")
  add_definitions(-DROS_VERSION_1)

  find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    pcl_conversions
    pcl_ros
    cv_bridge
    image_transport
  )

  catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs pcl_conversions pcl_ros cv_bridge image_transport
  )
  include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
  )
elseif(ROS_VERSION STREQUAL "2")
  message(STATUS "================= Building for ROS2 =================")
  add_definitions(-DROS_VERSION_2)

  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclpy REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(pcl_ros REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(image_transport REQUIRED)

  include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
  )
endif()

add_library(ros_common SHARED src/ros_common.cpp)
if(ROS_VERSION STREQUAL "1")
  target_link_libraries(ros_common ${catkin_LIBRARIES})
elseif(ROS_VERSION STREQUAL "2")
  ament_target_dependencies(ros_common
    rclcpp
    rcl
    rcl_interfaces
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    pcl_conversions
    pcl_ros
    cv_bridge
    image_transport
    tf2_ros
  )
endif()

add_library(globals SHARED src/globals.cpp)
target_link_libraries(globals)

add_library(cam SHARED src/camera_model.cpp)
target_link_libraries(cam ${OpenCV_LIBRARIES})
add_library(tracking SHARED src/tracker.cpp src/tracking_manager.cpp)
target_link_libraries(tracking ${OpenCV_LIBRARIES} cam ros_common)

add_executable(voxelodom src/voxelslam.cpp src/voxel_map.cpp src/tools.cpp)
target_link_libraries(voxelodom ros_common ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp tracking cam globals)

if(ROS_VERSION STREQUAL "2")
  install(TARGETS voxelodom DESTINATION lib/${PROJECT_NAME})
  ament_package()
endif()