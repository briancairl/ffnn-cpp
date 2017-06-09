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
#include <ffnn/logging.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/input.h>
#include <ffnn/layer/convolution.h>
#include <ffnn/layer/output.h>
#include <ffnn/optimizer/gradient_descent.h>
#include <ffnn/distribution/normal.h>

TEST(TestLayerConvolutionWithOptimizers, Static_GradientDescent)
{
  using namespace ffnn::layer;
  using namespace ffnn::optimizer;

  using convolution::ColEmbedding;
  using Options = convolution::options<16, 16, 3, 4, 4, 8, 1, 1, RowEmbedding>;

  // Layer-type alias
  using Layer  = Layer<float>;
  using Input  = Input<float>;
  using Hidden = Convolution<float, Options>;
  using Output = Output<float>;

  // Optimizer alias
  using Optimizer = GradientDescent<Hidden, CrossEntropy>;

  // Layer sizes
  static const ffnn::size_type DIM = 16 * 16 * 3;

  // Create layers
  auto input  = boost::make_shared<Input>(DIM);
  auto hidden = boost::make_shared<Hidden>(Hidden::Configuration()
                                           .setOptimizer(boost::make_shared<Optimizer>(1e-4)));
  auto output = boost::make_shared<Output>();

  // Create network
  std::vector<Layer::Ptr> layers({input, hidden, output});

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    EXPECT_TRUE(connect<Layer>(layers[idx-1UL], layers[idx]));
  }

  // Initialize and check all layers and 
  for(const auto& layer : layers)
  {
    layer->initialize();
    EXPECT_TRUE(layer->isInitialized());
  }

  // Create some data
  Hidden::InputBlockType input_data;
  input_data.setOnes();

  Hidden::OutputBlockType target_data;
  Hidden::OutputBlockType output_data;
  target_data.setOnes();
  output_data.setZero();

  // Check that error montonically decreases
  for (size_t idx = 0UL; idx < 1000; idx++)
  {
    // Forward activate
    (*input) << input_data;
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->forward());
    }
    (*output) >> output_data;

    // Set target
    (*output) << target_data;

    // Backward propogated error
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->backward());
    }

    // Trigget optimization
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->update());
    }
  }
  EXPECT_NEAR((target_data - output_data).norm(), 0.0, 1e-4);
}

TEST(TestLayerConvolutionWithOptimizers, Dynamic_GradientDescent)
{
  using namespace ffnn::layer;
  using namespace ffnn::optimizer;

  using convolution::ColEmbedding;

  // Layer-type alias
  using Layer  = Layer<float>;
  using Input  = Input<float>;
  using Hidden = Convolution<float>;
  using Output = Output<float>;

  // Optimizer alias
  using Optimizer = GradientDescent<Hidden, CrossEntropy>;

  // Layer sizes
  static const ffnn::size_type DIM = 16 * 16 * 3;

  // Create layers
  auto input  = boost::make_shared<Input>(DIM);
  auto hidden = boost::make_shared<Hidden>(Hidden::Configuration()
                                           .setInputShape(16, 16, 3)
                                           .setFilterShape(4, 4, 8)
                                           .setStride(1, 1)
                                           .setOptimizer(boost::make_shared<Optimizer>(1e-4)));
  auto output = boost::make_shared<Output>();

  // Create network
  std::vector<Layer::Ptr> layers({input, hidden, output});

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    EXPECT_TRUE(connect<Layer>(layers[idx-1UL], layers[idx]));
  }

  // Initialize and check all layers and 
  for(const auto& layer : layers)
  {
    layer->initialize();
    EXPECT_TRUE(layer->isInitialized());
  }

  // Create some data
  Hidden::InputBlockType input_data(hidden->getInputShape().height,
                                    hidden->getInputShape().width);
  input_data.setOnes();

  Hidden::OutputBlockType target_data(hidden->getOutputShape().height,
                                      hidden->getOutputShape().width);
  Hidden::OutputBlockType output_data(hidden->getOutputShape().height,
                                      hidden->getOutputShape().width);
  target_data.setOnes();
  output_data.setZero();

  // Check that error montonically decreases
  for (size_t idx = 0UL; idx < 1000; idx++)
  {
    // Forward activate
    (*input) << input_data;
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->forward());
    }
    (*output) >> output_data;

    // Set target
    (*output) << target_data;

    // Backward propogated error
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->backward());
    }

    // Trigget optimization
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->update());
    }
  }
  EXPECT_NEAR((target_data - output_data).norm(), 0.0, 1e-4);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
