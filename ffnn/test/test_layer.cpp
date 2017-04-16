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
#include <ffnn/layer/layer.h>

/// Layer implementation
class Layer :
  public ffnn::layer::Layer<int>
{
public:
  Layer(const Layer::SizeType output_dim) :
    ffnn::layer::Layer<int>(output_dim)
  {}

  virtual bool load(std::istream& is) { return true; }
  virtual bool save(std::ostream& os) { return true; }
};

static const Layer::SizeType N_OUTPUTS = 10;

TEST(TestLayer, LayerNoInitialization)
{
  // Don't initialize layer
  Layer layer(N_OUTPUTS);
  EXPECT_FALSE(layer.isInitialized());
  EXPECT_THROW(layer.input(), std::logic_error);
  EXPECT_THROW(layer.output(), std::logic_error);
}

TEST(TestLayer, LayerInitialization)
{
  // Initialize layer
  Layer layer(N_OUTPUTS);
  EXPECT_TRUE(layer.initialize());

  // Should be initialized
  EXPECT_TRUE(layer.isInitialized());

  // Already initialized
  EXPECT_FALSE(layer.initialize());
}

TEST(TestLayer, LayerConnectForward)
{
  static const Layer::SizeType DIM_0 = 10;
  static const Layer::SizeType DIM_1 = 20;
  static const Layer::SizeType DIM_2 = 30;

  // Create two layers to connect
  std::vector<boost::shared_ptr<Layer>> layers(3UL);
  layers[0] = boost::make_shared<Layer>(DIM_0);
  layers[1] = boost::make_shared<Layer>(DIM_1);
  layers[2] = boost::make_shared<Layer>(DIM_2);

  // Check that all layers are uninitialized
  for(auto& layer : layers)
  {
    EXPECT_FALSE(layer->isInitialized());
  }

  // Connect layers
  EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[0], layers[1]));
  EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[1], layers[2]));
  EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[2], layers[0]));

  // Check that all layers are initialized
  for(const auto& layer : layers)
  {
    layer->initialize();
    EXPECT_TRUE(layer->isInitialized());
  }

  // // Check output dimensions
  // EXPECT_EQ(layers[0]->output().rows(), DIM_0);
  // EXPECT_EQ(layers[1]->output().rows(), DIM_1);
  // EXPECT_EQ(layers[2]->output().rows(), DIM_2);
  
  // // Check input-output consistency
  // EXPECT_EQ(layers[1]->input().rows(), layers[0]->output().rows());
  // EXPECT_EQ(layers[2]->input().rows(), layers[1]->output().rows());
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}