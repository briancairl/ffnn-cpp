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
#include <ffnn/layer/input.h>

TEST(TestLayerInout, Static)
{
  using ffnn::layer::input::options;
  using Input = ffnn::layer::Input<float, options<300>>;

  Input layer;

  // Check output size (same as effective network input size)
  EXPECT_EQ(layer.getOutputShape().size(), 300);
}

TEST(TestLayerInout, Dynamic_SingleArg)
{
  using Input = ffnn::layer::Input<float>;

  Input layer(300);

  // Check output size (same as effective network input size)
  EXPECT_EQ(layer.getOutputShape().size(), 300);
}

TEST(TestLayerInout, Dynamic_InputShape)
{
  using Input = ffnn::layer::Input<float>;
  using Config = Input::Configuration; 

  Input layer(Config().setInputShape(10, 10, 3));

  // Check output size (same as effective network input size)
  EXPECT_EQ(layer.getOutputShape().size(), 300);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
