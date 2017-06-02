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
#include <ffnn/layer/convolution.h>

TEST(TestLayerConvolution, Dynamic_SizingWithConfigStruct)
{
  using Conv = ffnn::layer::Convolution<float>;
  using Config = Conv::Configuration;

  Conv layer(Config()
             .setInputShape(64, 64, 3)
             .setFilterShape(4, 4, 5)
             .setStride(3, 3));

  // Check ColEmbedding input sizing
  EXPECT_EQ(layer.getInputShape().height, 64 * 3);
  EXPECT_EQ(layer.getInputShape().width,  64);

  // Check ColEmbedding output sizing
  EXPECT_EQ(layer.getOutputShape().height, 21 * 5);
  EXPECT_EQ(layer.getOutputShape().width,  21);
}

TEST(TestLayerConvolution, Static_FilterSizing_ColEmbedding)
{
  using ffnn::layer::convolution::ColEmbedding;
  using Options = ffnn::layer::convolution::options<4, 4, 2, 4, 4, 5, 3, 3, ColEmbedding>;
  using Conv = ffnn::layer::Convolution<float, Options>;

  Conv layer;

  // Check ColEmbedding input sizing
  EXPECT_EQ(layer.getInputShape().height, 4 * 2);
  EXPECT_EQ(layer.getInputShape().width,  4);

  // Check ColEmbedding output sizing
  EXPECT_EQ(layer.getOutputShape().height, 1 * 5);
  EXPECT_EQ(layer.getOutputShape().width,  1);
}

TEST(TestLayerConvolution, Static_FilterSizing_RowEmbedding)
{
  using ffnn::layer::convolution::RowEmbedding;
  using Options = ffnn::layer::convolution::options<4, 4, 2, 4, 4, 5, 3, 3, RowEmbedding>;
  using Conv = ffnn::layer::Convolution<float, Options>;

  Conv layer;

  // Check ColEmbedding input sizing
  EXPECT_EQ(layer.getInputShape().height, 4);
  EXPECT_EQ(layer.getInputShape().width,  4 * 2);

  // Check ColEmbedding output sizing
  EXPECT_EQ(layer.getOutputShape().height, 1);
  EXPECT_EQ(layer.getOutputShape().width,  1 * 5);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
