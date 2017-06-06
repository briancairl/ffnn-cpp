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
#include <ffnn/layer/fully_connected.h>
#include <ffnn/layer/output.h>
#include <ffnn/optimizer/gradient_descent.h>

TEST(TestLayerFullyConnectedTraining, Dynamic_GradientDescent)
{
  using namespace ffnn::layer;
  using namespace ffnn::optimizer;

  // Layer-type alias
  using Layer  = Layer<float>;
  using Input  = Input<float>;
  using Hidden = FullyConnected<float>;
  using Output = Output<float>;

  // Optimizer alias
  using Optimizer = GradientDescent<Hidden>;

  // Layer sizes
  static const ffnn::size_type DIM = 32;

  // Create layers
  auto input  = boost::make_shared<Input>(DIM);
  auto hidden = boost::make_shared<Hidden>(Hidden::Configuration()
                                           .setOutputShape(DIM)
                                           .setOptimizer(boost::make_shared<Optimizer>(1e-3)));
  auto output = boost::make_shared<Output>();

  // Create network
  std::vector<Layer::Ptr> layers({input, hidden, output});

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    EXPECT_TRUE(connect<Layer>(layers[idx-1UL], layers[idx]));
  }

  // Intializer layers
  input->initialize();
  hidden->initialize();
  output->initialize();

  // Initialize and check all layers and 
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->isInitialized());
  }

  // Create some data
  Hidden::InputBlockType target_data = Hidden::InputBlockType::Ones(DIM, 1);
  Hidden::InputBlockType output_data(DIM, 1);

  // Check that error montonically decreases
  for (size_t idx = 0UL; idx < 1000; idx++)
  {
    // Forward activate
    (*input) << target_data;
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


TEST(TestLayerFullyConnectedTraining, Static_GradientDescent)
{
  using namespace ffnn::layer;
  using namespace ffnn::optimizer;

  // Layer-type alias
  using Layer  = Layer<float>;
  using Input  = Input<float, input::options<32>>;
  using Hidden = FullyConnected<float, fully_connected::options<32, 32>>;
  using Output = Output<float, output::options<32>>;

  // Optimizer alias
  using Optimizer = GradientDescent<Hidden>;

  // Layer sizes
  static const ffnn::size_type DIM = 32;

  // Create layers
  auto input  = boost::make_shared<Input>(DIM);
  auto hidden = boost::make_shared<Hidden>(Hidden::Configuration()
                                           .setOutputShape(DIM)
                                           .setOptimizer(boost::make_shared<Optimizer>(1e-3)));
  auto output = boost::make_shared<Output>();

  // Create network
  std::vector<Layer::Ptr> layers({input, hidden, output});

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    EXPECT_TRUE(connect<Layer>(layers[idx-1UL], layers[idx]));
  }

  // Intializer layers
  input->initialize();
  hidden->initialize();
  output->initialize();

  // Initialize and check all layers and 
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->isInitialized());
  }

  // Create some data
  Hidden::InputBlockType target_data = Hidden::InputBlockType::Ones(DIM, 1);
  Hidden::InputBlockType output_data(DIM, 1);

  // Check that error montonically decreases
  for (size_t idx = 0UL; idx < 1000; idx++)
  {
    // Forward activate
    (*input) << target_data;
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
