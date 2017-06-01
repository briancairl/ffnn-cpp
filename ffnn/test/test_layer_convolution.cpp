/**
 * @author Brian Cairl
 * @date 2017
 */
// C++ Standard Library
#include <fstream>
#include <type_traits>

// GTest
#include <gtest/gtest.h>

// FFNN
#include <ffnn/logging.h>
#include <ffnn/layer/convolution.h>

TEST(TestLayerConvolution, DefaultDynamic)
{
  typedef ffnn::layer::Convolution<float> ConvLayer;
  ConvLayer layer;
}


// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
