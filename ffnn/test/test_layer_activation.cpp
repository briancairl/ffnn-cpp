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
#include <ffnn/neuron/sigmoid.h>
#include <ffnn/layer/activation.h>

TEST(TestLayerInout, Static)
{
  using ffnn::layer::activation::options;
  using Neuron = ffnn::neuron::Sigmoid<float>;
  using Activation = ffnn::layer::Activation<float, Neuron, options<300>>;

  Activation layer;

  // Check sizes
  EXPECT_EQ(layer.getInputShape().size(), 300);
  EXPECT_EQ(layer.getOutputShape().size(), layer.getInputShape().size());
}

TEST(TestLayerInout, Dynamic_SingleArg)
{
  using Neuron = ffnn::neuron::Sigmoid<float>;
  using Activation = ffnn::layer::Activation<float, Neuron>;

  Activation layer(300);

  // Check sizes
  EXPECT_EQ(layer.getOutputShape().size(), 300);
  EXPECT_EQ(layer.getOutputShape().size(), layer.getInputShape().size());
}

TEST(TestLayerInout, Dynamic_InputShape)
{
  using Neuron = ffnn::neuron::Sigmoid<float>;
  using Activation = ffnn::layer::Activation<float, Neuron>;
  using Config = Activation::Configuration;

  Activation layer(Config().setInputShape(10, 10, 3));

  // Check sizes
  EXPECT_EQ(layer.getOutputShape().size(), 300);
  EXPECT_EQ(layer.getOutputShape().size(), layer.getInputShape().size());
}


TEST(TestLayerInout, Dynamic_OutputShape)
{
  using Neuron = ffnn::neuron::Sigmoid<float>;
  using Activation = ffnn::layer::Activation<float, Neuron>;
  using Config = Activation::Configuration;

  Activation layer(Config().setOutputShape(10, 10, 3));

  // Check sizes
  EXPECT_EQ(layer.getOutputShape().size(), 300);
  EXPECT_EQ(layer.getOutputShape().size(), layer.getInputShape().size());
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
