if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFFNN_NO_LOGGING -DFFNN_SUPRESS_ERROR_LOGGING")
endif()

#-------------------------------------------------------------
# IO
#-------------------------------------------------------------

##############################################################
# Tests:
# 	- Save
# 	- Load
#   - layer::Input
#   - layer::Output
#   - layer::FullyConnected
##############################################################

catkin_add_gtest(test_layer_fully_connected_io
  test_layer_fully_connected_io.cpp
)
target_link_libraries(test_layer_fully_connected_io
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

##############################################################
# Tests:
# 	- Save
# 	- Load
#   - layer::Input
#   - layer::Output
#   - layer::SparselyConnected
##############################################################

catkin_add_gtest(test_layer_sparsely_connected_io
  test_layer_sparsely_connected_io.cpp
)
target_link_libraries(test_layer_sparsely_connected_io
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

#-------------------------------------------------------------
# Training
#-------------------------------------------------------------

##############################################################
# Tests:
#    - layer::Input
#    - layer::Output
#    - layer::FullyConnected
#    - layer::Activation
#    - neuron::modifier::Dropout
#    - neuron::Linear
#  	 - optimizer::GradientDescent[FullyConnected]
#  	 - optimizer::GradientDescent[Activation]
##############################################################

catkin_add_gtest(test_layer_activation_dropout_training
  test_layer_activation_dropout_training.cpp
)
target_link_libraries(test_layer_activation_dropout_training
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

##############################################################
# Tests:
#    - layer::Input
#    - layer::Output
#    - layer::FullyConnected
#    - layer::Activation
#  	 - optimizer::GradientDescent[FullyConnected]
#  	 - optimizer::GradientDescent[Activation]
##############################################################

catkin_add_gtest(test_layer_fully_connected_training
  test_layer_fully_connected_training.cpp
)
target_link_libraries(test_layer_fully_connected_training
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

##############################################################
# Tests:
#    - layer::Input
#    - layer::Output
#    - layer::SparselyConnected
#  	 - optimizer::GradientDescent
##############################################################

catkin_add_gtest(test_layer_sparsely_connected_training
  test_layer_sparsely_connected_training.cpp
)
target_link_libraries(test_layer_sparsely_connected_training
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)