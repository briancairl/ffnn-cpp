cmake_minimum_required(VERSION 2.8.3)
project(ffnn_app)

if(NOT CMAKE_CONFIGURATION_TYPES AND (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "None"))
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DEIGEN_NO_DEBUG -std=c++11 -Wall -Wextra -Werror -Wno-unused-parameter")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DEIGEN_NO_DEBUG -DFFNN_ALLOW_LOGGING -march=native -msse2")

find_package(Boost REQUIRED COMPONENTS thread random serialization)
find_package(Eigen3 REQUIRED)
find_package(ffnn REQUIRED)


catkin_package(
  CATKIN_DEPENDS
  	ffnn
  DEPENDS
    Boost
)

include_directories(
  ${ffnn_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

add_executable(train
  src/train.cpp
)
target_link_libraries(train
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

add_executable(test_trained
  src/test_trained.cpp
)
target_link_libraries(test_trained
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

