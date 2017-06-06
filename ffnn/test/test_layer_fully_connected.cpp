/**
 * @author Brian Cairl
 * @date 2017
 */
// C++ Standard Library
#include <exception>
#include <fstream>
#include <vector>

// Boost
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// GTest
#include <gtest/gtest.h>

// FFNN
#include <ffnn/logging.h>
#include <ffnn/layer/fully_connected.h>


TEST(TestLayerFullyConnected, Dynamic_SingleArg)
{
  // Layer-type alias
  using Hidden = ffnn::layer::FullyConnected<float>;

  Hidden fully_connected(30);

  EXPECT_EQ(fully_connected.getOutputShape().size(), 30);
}

TEST(TestLayerFullyConnected, Dynamic_Config_InputOutputSize)
{
  // Layer-type alias
  using Hidden = ffnn::layer::FullyConnected<float>;

  Hidden fully_connected(Hidden::Configuration()
                         .setInputShape(10, 1, 3)
                         .setOutputShape(30, 3));

  EXPECT_EQ(fully_connected.getInputShape().size(), 30);
  EXPECT_EQ(fully_connected.getOutputShape().size(), 90);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
