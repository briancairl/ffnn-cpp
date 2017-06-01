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
#include <ffnn/layer/convolution/filter.h>

TEST(TestLayerConvolutionFilter, DefaultDynamic)
{
  typedef ffnn::layer::convolution::Filter<float> Filter;
  Filter filter;

  EXPECT_EQ(filter.size(), 0);

  filter.resize(5);
  for (const auto& kernel : filter)
  {
    EXPECT_EQ(kernel.rows(), 0);
    EXPECT_EQ(kernel.cols(), 0);
  }
}

TEST(TestLayerConvolutionFilter, SetupDynamic)
{
  typedef ffnn::layer::convolution::Filter<float> Filter;
  Filter filter;

  filter.setZero(4, 5, 6, 7);

  EXPECT_EQ(filter.size(), 7);
  for (const auto& kernel : filter)
  {
    EXPECT_EQ(kernel.rows(), 24);
    EXPECT_EQ(kernel.cols(), 5);
    EXPECT_NEAR(kernel.sum(), 0, 1e-9);
  }
  EXPECT_NEAR(filter.bias, 0, 1e-9);
}

TEST(TestLayerConvolutionFilter, StaticColEmbedding)
{
  using ffnn::layer::convolution::ColEmbedding;
  using ffnn::layer::convolution::filter_traits;
  typedef ffnn::layer::convolution::Filter<float, filter_traits<float, 4, 4, 4, 10, ColEmbedding>> Filter;
  Filter filter;

  EXPECT_NO_THROW(filter.setZero(4, 4, 4, 10));
  EXPECT_EQ(filter.size(), 10);
  for (const auto& kernel : filter)
  {
    EXPECT_EQ(kernel.rows(), 16);
    EXPECT_EQ(kernel.cols(), 4);
    EXPECT_NEAR(kernel.sum(), 0, 1e-9);
  }
  EXPECT_NEAR(filter.bias, 0, 1e-9);
}

TEST(TestLayerConvolutionFilter, StaticRowEmbedding)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter_traits;
  typedef ffnn::layer::convolution::Filter<float, filter_traits<float, 3, 3, 5, 12, RowEmbedding>> Filter;
  Filter filter;

  EXPECT_NO_THROW(filter.setZero(3, 3, 5, 12));
  EXPECT_EQ(filter.size(), 12);
  for (const auto& kernel : filter)
  {
    EXPECT_EQ(kernel.rows(), 3);
    EXPECT_EQ(kernel.cols(), 15);
  }
}

TEST(TestLayerConvolutionFilter, SerializationDynamic)
{
  typedef ffnn::layer::convolution::Filter<float> Filter;
  Filter filter;
  filter.setZero(4, 5, 6, 7);
  {
    std::ofstream ofs("filter.bin", std::ios::binary);
    FFNN_SERIALIZATION_OUTPUT_ARCHIVE_TYPE oar(ofs);
    oar << filter;
  }

  Filter filter_loaded;
  {
    std::ifstream ifs("filter.bin", std::ios::binary);
    FFNN_SERIALIZATION_INPUT_ARCHIVE_TYPE iar(ifs);
    iar >> filter_loaded;
  }

  EXPECT_EQ(filter.size(), filter_loaded.size());
  for (size_t idx = 0UL; idx < filter.size(); idx++)
  {
    EXPECT_EQ(filter[idx].rows(), filter_loaded[idx].rows());
    EXPECT_EQ(filter[idx].cols(), filter_loaded[idx].cols());
  }
}

TEST(TestLayerConvolutionFilter, SerializationStaticColEmbedding)
{
  using ffnn::layer::convolution::ColEmbedding;
  using ffnn::layer::convolution::filter_traits;
  typedef ffnn::layer::convolution::Filter<float, filter_traits<float, 3, 3, 5, 12, ColEmbedding>> Filter;

  Filter filter;
  filter.setZero(3, 3, 5, 12);
  {
    std::ofstream ofs("filter.bin", std::ios::binary);
    FFNN_SERIALIZATION_OUTPUT_ARCHIVE_TYPE oar(ofs);
    oar << filter;
  }

  Filter filter_loaded;
  {
    std::ifstream ifs("filter.bin", std::ios::binary);
    FFNN_SERIALIZATION_INPUT_ARCHIVE_TYPE iar(ifs);
    iar >> filter_loaded;
  }

  EXPECT_EQ(filter.size(), filter_loaded.size());
  for (size_t idx = 0UL; idx < filter.size(); idx++)
  {
    EXPECT_EQ(filter[idx].rows(), filter_loaded[idx].rows());
    EXPECT_EQ(filter[idx].cols(), filter_loaded[idx].cols());
  }
}


TEST(TestLayerConvolutionFilter, SerializationStaticRowEmbedding)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter_traits;
  typedef ffnn::layer::convolution::Filter<float, filter_traits<float, 3, 3, 5, 12, RowEmbedding>> Filter;

  Filter filter;
  filter.setZero(3, 3, 5, 12);
  {
    std::ofstream ofs("filter.bin", std::ios::binary);
    FFNN_SERIALIZATION_OUTPUT_ARCHIVE_TYPE oar(ofs);
    oar << filter;
  }

  Filter filter_loaded;
  {
    std::ifstream ifs("filter.bin", std::ios::binary);
    FFNN_SERIALIZATION_INPUT_ARCHIVE_TYPE iar(ifs);
    iar >> filter_loaded;
  }

  EXPECT_EQ(filter.size(), filter_loaded.size());
  for (size_t idx = 0UL; idx < filter.size(); idx++)
  {
    EXPECT_EQ(filter[idx].rows(), filter_loaded[idx].rows());
    EXPECT_EQ(filter[idx].cols(), filter_loaded[idx].cols());
  }
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
