if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFFNN_NO_LOGGING -DFFNN_SUPRESS_ERROR_LOGGING")
endif()

############################################
# Tests:
# 	- Save
# 	- Load
#   - layer::Input
#   - layer::Output
#   - layer::FullyConnected
############################################

catkin_add_gtest(test_layer_fully_connected_io
  test_layer_fully_connected_io.cpp
)
target_link_libraries(test_layer_fully_connected_io
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

############################################
# Tests:
#    - layer::Input
#    - layer::Output
#    - layer::FullyConnected
#  	 - optimizer::GradientDescent
############################################

catkin_add_gtest(test_layer_fully_connected_training
  test_layer_fully_connected_training.cpp
)
target_link_libraries(test_layer_fully_connected_training
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

############################################
# Tests:
#    - layer::Input
#    - layer::Output
#    - layer::SparselyConnected
#  	 - optimizer::GradientDescent
############################################

catkin_add_gtest(test_layer_sparsely_connected_training
  test_layer_sparsely_connected_training.cpp
)
target_link_libraries(test_layer_sparsely_connected_training
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)