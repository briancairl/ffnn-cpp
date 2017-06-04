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

TEST(TestLayerConvolutionFilter, Static_ColEmbedding_KernelSizing)
{
  using ffnn::layer::convolution::ColEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<4, 4, 4, 10, ColEmbedding>> Filter;
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

TEST(TestLayerConvolutionFilter, Static_RowEmbedding_KernelSizing)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<3, 3, 5, 12, RowEmbedding>> Filter;
  Filter filter;

  EXPECT_NO_THROW(filter.setZero(3, 3, 5, 12));
  EXPECT_EQ(filter.size(), 12);
  for (const auto& kernel : filter)
  {
    EXPECT_EQ(kernel.rows(), 3);
    EXPECT_EQ(kernel.cols(), 15);
  }
}

TEST(TestLayerConvolutionFilter, Static_Scaling)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<3, 3, 5, 12, RowEmbedding>> Filter;
  Filter filterA, filterB;

  filterA.bias = 1;
  for (auto& kernel : filterA)
  {
    kernel.setOnes();
  }

  filterA *= 2;

  EXPECT_NEAR(filterA.bias, 2.0, 1e-9);
  for (const auto& kernel : filterA)
  {
    EXPECT_NEAR(kernel(0, 0), 2.0, 1e-9);
  }
}

TEST(TestLayerConvolutionFilter, Static_ElementWiseDivision)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<3, 3, 5, 12, RowEmbedding>> Filter;
  Filter filterA, filterB;

  filterA.bias = 1;
  for (auto& kernel : filterA)
  {
    kernel.setOnes();
  }

  filterB.bias = 1;
  for (auto& kernel : filterB)
  {
    kernel.setOnes();
  }

  filterB *= 2;
  filterA /= filterB;

  EXPECT_NEAR(filterA.bias, 0.5, 1e-9);
  for (const auto& kernel : filterA)
  {
    EXPECT_NEAR(kernel(0, 0), 0.5, 1e-9);
  }
}

TEST(TestLayerConvolutionFilter, Static_Assignment)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<3, 3, 5, 12, RowEmbedding>> Filter;
  Filter filterA, filterB;

  filterA.bias = 1;
  for (auto& kernel : filterA)
  {
    kernel.setOnes();
  }

  filterB.bias = 1;
  for (auto& kernel : filterB)
  {
    kernel.setOnes();
  }

  filterB *= 2;
  filterA *= filterB;

  EXPECT_NEAR(filterA.bias, 2.0, 1e-9);
  for (const auto& kernel : filterA)
  {
    EXPECT_NEAR(kernel(0, 0), 2.0, 1e-9);
  }
}

TEST(TestLayerConvolutionFilter, Static_ElementWiseMultiplication)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<3, 3, 5, 12, RowEmbedding>> Filter;
  Filter filterA, filterB;

  filterA.bias = 1;
  for (auto& kernel : filterA)
  {
    kernel.setOnes();
  }

  filterB.bias = 1;
  for (auto& kernel : filterB)
  {
    kernel.setOnes();
  }

  filterA *= 3;
  filterB *= 2;
  filterA *= filterB;

  EXPECT_NEAR(filterA.bias, 6.0, 1e-9);
  for (const auto& kernel : filterA)
  {
    EXPECT_NEAR(kernel(0, 0), 6.0, 1e-9);
  }
}

TEST(TestLayerConvolutionFilter, Static_ElementWiseAddition)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<3, 3, 5, 12, RowEmbedding>> Filter;
  Filter filterA, filterB;

  filterA.bias = 1;
  for (auto& kernel : filterA)
  {
    kernel.setOnes();
  }

  filterB.bias = 1;
  for (auto& kernel : filterB)
  {
    kernel.setOnes();
  }

  filterA *= 3;
  filterB *= 2;
  filterA += filterB;

  EXPECT_NEAR(filterA.bias, 5.0, 1e-9);
  for (const auto& kernel : filterA)
  {
    EXPECT_NEAR(kernel(0, 0), 5.0, 1e-9);
  }
}

TEST(TestLayerConvolutionFilter, Static_ElementWiseSubtraction)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<3, 3, 5, 12, RowEmbedding>> Filter;
  Filter filterA, filterB;

  filterA.bias = 1;
  for (auto& kernel : filterA)
  {
    kernel.setOnes();
  }

  filterB.bias = 1;
  for (auto& kernel : filterB)
  {
    kernel.setOnes();
  }

  filterA *= 3;
  filterB *= 2;
  filterA -= filterB;

  EXPECT_NEAR(filterA.bias, 1.0, 1e-9);
  for (const auto& kernel : filterA)
  {
    EXPECT_NEAR(kernel(0, 0), 1.0, 1e-9);
  }
}

TEST(TestLayerConvolutionFilter, Static_Serialization_ColEmbedding)
{
  using ffnn::layer::convolution::ColEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<3, 3, 5, 12, ColEmbedding>> Filter;

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


TEST(TestLayerConvolutionFilter, Static_Serialization_RowEmbedding)
{
  using ffnn::layer::convolution::RowEmbedding;
  using ffnn::layer::convolution::filter::options;
  typedef ffnn::layer::convolution::Filter<float, options<3, 3, 5, 12, RowEmbedding>> Filter;

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

TEST(TestLayerConvolutionFilter, Dynamic_Serialization)
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

TEST(TestLayerConvolutionFilter, Dynamic_Default)
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

TEST(TestLayerConvolutionFilter, Dynamic_Setup)
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


// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
