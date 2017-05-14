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
#include <ffnn/layer/activation.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/input.h>
#include <ffnn/layer/convolution.h>
#include <ffnn/layer/output.h>
#include <ffnn/neuron/lecun_sigmoid.h>
#include <ffnn/optimizer/gradient_descent.h>
#include <ffnn/distribution/normal.h>

/***********************************************************/
// Creates network workflow with one Convolution hidden
// layer and updates using a GradientDescent optimizer
//
// Tests:
//    - layer::Input
//    - layer::Output
//    - layer::Convolution
//    - optimizer::GradientDescent
/***********************************************************/
TEST(TestLayerConvolutionWithOptimizers, GradientDescent)
{
  // Layer-type alias
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Hidden = ffnn::layer::Convolution<float, 64, 64, 3, 4, 4, 4, 1, ffnn::layer::ColEmbedding>;
  using Output = ffnn::layer::Output<float>;

  // Layer sizes
  static const Layer::SizeType DIM = 64 * 64 * 3;

  // Create layers
  auto input = boost::make_shared<Input>(DIM);
  auto hidden = boost::make_shared<Hidden>();
  auto output = boost::make_shared<Output>();

  // Set optimizer (gradient descent)
  {
    using Optimizer = ffnn::optimizer::GradientDescent<Hidden>;
    hidden->setOptimizer(boost::make_shared<Optimizer>(1e-7));
  }

  // Create network
  std::vector<Layer::Ptr> layers({input, hidden, output});

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[idx-1UL], layers[idx]));
  }

  // Intializer layers
  input->initialize();
  hidden->initialize(ffnn::distribution::Normal<float>(0, 1.0 / DIM),
                     ffnn::distribution::Normal<float>(0, 1.0 / DIM));
  output->initialize();

  // Initialize and check all layers and 
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->isInitialized());
  }

  // Create some data
  Hidden::InputBlockType input_data;
  input_data.setOnes();
  Hidden::OutputBlockType target_data;
  target_data.setOnes();
  target_data /= 1.0;
  Hidden::OutputBlockType output_data;
  output_data.setOnes();
  output_data /= 1.0;

  // Check that error montonically decreases
  //float prev_error = std::numeric_limits<float>::infinity();
  for (size_t idx = 0UL; idx < 1e4; idx++)
  {
    // Forward activate
    (*input) << input_data;
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->forward());
    }
    (*output) >> output_data;

    // Compute error and check
    //double error = (target_data - output_data).norm();
    //EXPECT_LE(error, prev_error);

    // Set target
    (*output) << target_data;

    //FFNN_INFO(output_data);

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

    // Store previous error
    //prev_error = error;
  }

  FFNN_INFO(output_data);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
