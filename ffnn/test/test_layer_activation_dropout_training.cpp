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
#include <ffnn/neuron/linear.h>
#include <ffnn/neuron/modifier/dropout.h>
#include <ffnn/neuron/modifier/soft_dropout.h>
#include <ffnn/optimizer/gradient_descent.h>
#include <ffnn/distribution/standard_normal.h>

/// Create a neuron with dropout
template<typename ValueType>
class LinearStandardNormalDropout:
  public ffnn::neuron::modifier::Dropout<
    ValueType,
    ffnn::neuron::Linear,
    ffnn::distribution::StandardNormal, 5>
{};

/***********************************************************/
// Creates network workflow with one FullyConnected hidden
// layer and updates using a GradientDescent optimizer
//
// Tests:
//    - layer::Input
//    - layer::Output
//    - layer::FullyConnected
//    - layer::Activation
//    - neuron::modifier::Dropout
//    - neuron::Linear
//    - optimizer::GradientDescent[FullyConnected]
//    - optimizer::GradientDescent[Activation]
/***********************************************************/
TEST(TestLayerFullyConnectedDropoutActivationWithOptimizers, GradientDescent)
{
  // Layer-type alias
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Hidden = ffnn::layer::FullyConnected<float>;
  using Activation = ffnn::layer::Activation<float, LinearStandardNormalDropout>;
  using Output = ffnn::layer::Output<float>;

  // Layer sizes
  static const Layer::SizeType DIM = 32;

  // Create layers
  auto input = boost::make_shared<Input>(DIM);  
  auto hidden1 = boost::make_shared<Hidden>(DIM);
  auto hidden2 = boost::make_shared<Activation>();
  auto output = boost::make_shared<Output>();  

  // Set optimizer (gradient descent)
  {
    using Optimizer = ffnn::optimizer::GradientDescent<Hidden>;
    hidden1->setOptimizer(boost::make_shared<Optimizer>(1e-5));
  }

  // Create network
  std::vector<Layer::Ptr> layers({input, hidden1, hidden2, output});

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[idx-1UL], layers[idx]));
  }

  // Initialize and check all layers and 
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->initialize());
    EXPECT_TRUE(layer->isInitialized());
  }

  // Create some data
  Hidden::InputVector target_data = Hidden::InputVector::Ones(DIM);
  Hidden::InputVector output_data(DIM, 1);

  // Check that error montonically decreases
  float prev_error = std::numeric_limits<float>::infinity();
  size_t lt_count = 0;
  for (size_t idx = 0UL; idx < 50; idx++)
  {
    // Forward activate
    (*input) << target_data;
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->forward());
    }
    (*output) >> output_data;

    // Compute error and check
    double error = (target_data - output_data).norm();
    if (error < prev_error)
    {
      lt_count++;
    }

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

    // Store previous error
    prev_error = error;
  }
  EXPECT_TRUE(lt_count > 25);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
