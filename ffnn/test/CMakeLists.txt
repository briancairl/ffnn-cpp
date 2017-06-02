# if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFFNN_NO_LOGGING -DFFNN_SUPRESS_ERROR_LOGGING")
# endif()

catkin_add_gtest(test_distribution
  test_distribution.cpp
)
target_link_libraries(test_distribution
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

catkin_add_gtest(test_layer_hidden
  test_layer_hidden.cpp
)
target_link_libraries(test_layer_hidden
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

catkin_add_gtest(test_layer_shape
  test_layer_shape.cpp
)
target_link_libraries(test_layer_shape
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

catkin_add_gtest(test_layer_convolution_filter
  test_layer_convolution_filter.cpp
)
target_link_libraries(test_layer_convolution_filter
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

catkin_add_gtest(test_layer_convolution
  test_layer_convolution.cpp
)
target_link_libraries(test_layer_convolution
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)

#-------------------------------------------------------------
# IO
#-------------------------------------------------------------

##############################################################
# Tests:
#   - Save
#   - Load
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
#    - layer::Convolution
#  	 - optimizer::GradientDescent[Convolution]
#  	 - optimizer::GradientDescent[Activation]
##############################################################

catkin_add_gtest(test_layer_convolution_training
  test_layer_convolution_training.cpp
)
target_link_libraries(test_layer_convolution_training
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
)
