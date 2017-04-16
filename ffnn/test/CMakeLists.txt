# ############################################
# # Layer
# ############################################

# catkin_add_gtest(test_layer
#   test_layer.cpp
# )
# target_link_libraries(test_layer
#   ${Boost_LIBRARIES}
#   ${catkin_LIBRARIES}
#   ${EIGEN_LIBRARIES}
#   ${GTEST_LIBRARIES}
# )

# ############################################
# # Simple Feed Forward Network
# # ------------------------------------------
# # Tests:
# # 	- layer/input.h
# #  	- layer/output.h
# # 	- layer/fully_connected.h
# ############################################

# catkin_add_gtest(test_simple_feed_forward
#   test_simple_feed_forward.cpp
# )
# target_link_libraries(test_simple_feed_forward
#   ${Boost_LIBRARIES}
#   ${catkin_LIBRARIES}
#   ${EIGEN_LIBRARIES}
#   ${GTEST_LIBRARIES}
# )

############################################
# Simple Feed Forward Network
# ------------------------------------------
# Tests:
#   - layer/input.h
#   - layer/output.h
#   - layer/fully_connected.h
#
# Trains a network to encode a sample sound
############################################

# Libsndfile
if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/include/ -lsndfile -lm")
endif()

catkin_add_gtest(test_simple_feed_forward_sound
  test_simple_feed_forward_sound.cpp
)
target_link_libraries(test_simple_feed_forward_sound
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
  ${GTEST_LIBRARIES}
  sndfile
)
