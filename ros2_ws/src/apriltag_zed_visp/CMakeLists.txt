cmake_minimum_required(VERSION 3.8)
project(apriltag_zed_visp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VISP REQUIRED)

# Find ZED SDK
find_package(ZED 4 QUIET)
if(NOT ZED_FOUND)
  find_package(ZED 3 QUIET)
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIRS}
)

# ROS2版本
add_executable(apriltag_detector
  src/apriltag_detector.cpp
  src/apriltag_36h11_detector.cpp
  src/pose_estimator.cpp
  src/error_compensation.cpp
)

ament_target_dependencies(apriltag_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)

target_link_libraries(apriltag_detector
  ${OpenCV_LIBS}
  ${VISP_LIBRARIES}
)

# 三标签基准系统（ID0+ID1 -> ID2）
if(ZED_FOUND)
  message(STATUS "ZED SDK found: ${ZED_VERSION}")

  find_package(CUDA QUIET)
  if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA found: ${CUDA_VERSION}")
  else()
    message(STATUS "CUDA not found, trying /usr/local/cuda")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
  endif()

  include_directories(${ZED_INCLUDE_DIRS})

  add_executable(apriltag_three_tags_ros
    src/apriltag_three_tags_ros.cpp
  )

  ament_target_dependencies(apriltag_three_tags_ros
    rclcpp
    std_msgs
    geometry_msgs
    tf2
  )

  target_link_libraries(apriltag_three_tags_ros
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
    cuda
  )

  install(TARGETS
    apriltag_three_tags_ros
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(TARGETS
  apriltag_detector
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
