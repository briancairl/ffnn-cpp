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
#include <ffnn/layer/output/output.h>

TEST(TestLayerOutput, Static)
{
  using ffnn::layer::output::options;
  using Output = ffnn::layer::Output<float, options<300>>;

  Output layer;

  // Check output size (same as effective network output size)
  EXPECT_EQ(layer.getInputShape().size(), 300);
}

TEST(TestLayerOutput, Dynamic_SingleArg)
{
  using Output = ffnn::layer::Output<float>;

  Output layer(300);

  // Check input size (same as effective network output size)
  EXPECT_EQ(layer.getInputShape().size(), 300);
}

TEST(TestLayerOutput, Dynamic_OutputShape)
{
  using Output = ffnn::layer::Output<float>;
  using Config = Output::Configuration; 

  Output layer(Config().setOutputShape(10, 10, 3));

  // Check input size (same as effective network output size)
  EXPECT_EQ(layer.getInputShape().size(), 300);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
