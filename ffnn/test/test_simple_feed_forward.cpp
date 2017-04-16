/**
 * @author Brian Cairl
 * @date 2017
 */

// C++ Standard Library
#include <exception>
#include <iostream>

// GTest
#include <gtest/gtest.h>

// FFNN
#undef FFNN_NO_LOGGING
#include <ffnn/layer/fully_connected.h>
#include <ffnn/layer/input.h>
#include <ffnn/layer/output.h>

#include <ffnn/neuron/rectified_linear.h>
#include <ffnn/neuron/sigmoid.h>

TEST(TestFullyConnected, FullyConnectedForward)
{
  using Layer = ffnn::layer::Layer<float>;
  using LayerReLU = ffnn::layer::FullyConnected<float, ffnn::neuron::RectifiedLinear>;
  using LayerSigm = ffnn::layer::FullyConnected<float, ffnn::neuron::Sigmoid>;
  using Input = ffnn::layer::Input<float>;
  using Output = ffnn::layer::Output<float>;

  static const Layer::SizeType DIM_0 = 200;
  static const Layer::SizeType DIM_1 = 200;
  static const Layer::SizeType DIM_2 = 10;

  // Create two layers to connect
  auto input_layer = boost::make_shared<Input>(DIM_0);
  auto output_layer = boost::make_shared<Output>();

  std::vector<Layer::Ptr> layers(4UL);

  // Custom configuration for LayerReLU
  LayerReLU::Config config_;
  config_.learning_rate = 1e-3;

  // Allocate layers
  layers[0] = input_layer;
  layers[1] = boost::make_shared<LayerReLU>(DIM_1, config_);
  layers[2] = boost::make_shared<LayerSigm>(DIM_2);
  layers[3] = output_layer;

  // Connect layers
  EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[0], layers[1])); // input-->net
  {
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[1], layers[2]));
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[2], layers[1])); // recurrence
  } 
  EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[2], layers[3])); // net-->output

  // Initialize and check all layers
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->initialize());
    EXPECT_TRUE(layer->isInitialized());
  }

  // Pretend training on constant target
  Output::OutputVector output;
  const auto INPUT  = Input::InputVector::Random(DIM_0, 1);
  const auto TARGET = Output::OutputVector::Ones(DIM_2, 1) * 0.5;
  for (size_t itr = 0UL; itr < 5000; itr++)
  {
    // Forward-propagate on all layers
    (*input_layer) << INPUT;
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->forward());
    }
    (*output_layer) >> output;

    // Compute output error and backward-propagate on all layers
    (*output_layer) << TARGET;
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->backward());
    }

    // Apply weights to all layers
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->update());
    }
  }
  FFNN_DEBUG_NAMED("RMSE", (output - TARGET).norm());
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}