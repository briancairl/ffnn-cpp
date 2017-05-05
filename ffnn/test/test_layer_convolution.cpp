/**
 * @author Brian Cairl
 * @date 2017
 */
// C++ Standard Library
#include <exception>
#include <fstream>
#include <limits>
#include <vector>

// Boost
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// GTest
#include <gtest/gtest.h>

// FFNN
#include <ffnn/layer/convolution.h>

TEST(TestLayerConvolution, StaticInstanceColEmbedding)
{
  // Volume-type alias
  using Convolution = ffnn::layer::Convolution<float, 64, 64, 3, 4, 4, 4, 1, ffnn::layer::ColEmbedding>;

  // Dimensions inferred from template args
  Convolution convolution;

  FFNN_INFO(convolution.getReceptiveVolumes().size());
  convolution.initialize();
  FFNN_INFO(convolution.getReceptiveVolumes().size());

  FFNN_INFO(convolution.getReceptiveVolumes()[0][0]->inputDim());
  FFNN_INFO(convolution.getReceptiveVolumes()[0][0]->outputDim());
  FFNN_INFO("\n" << convolution.getReceptiveVolumes()[0][0]->getFilters()[0]);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
