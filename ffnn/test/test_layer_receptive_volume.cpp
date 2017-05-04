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
#include <ffnn/layer/receptive_volume.h>

TEST(TestLayerReceptiveVolume, DynamicInstanceColEmbedding)
{
  // Volume-type alias
  using Volume = ffnn::layer::ReceptiveVolume<float>;

  // Dimensions supplied in constructor
  Volume volume(Volume::DimType(4, 6, 8), 12);

  EXPECT_TRUE(volume.getFilters().empty());
  EXPECT_TRUE(volume.initialize());
  EXPECT_EQ(volume.getFilters().size(), 12);

  for (const auto& filter : volume.getFilters())
  {
    // Columns should contain embedded depth (default)
    EXPECT_EQ(filter.rows(), 4 * 8);
    EXPECT_EQ(filter.cols(), 6);
    FFNN_DEBUG('\n' << filter);
  }
}

TEST(TestLayerReceptiveVolume, StaticInstanceRowEmbedding)
{
  // Volume-type alias
  using Volume = ffnn::layer::ReceptiveVolume<float, 4, 6, 8, 12, ffnn::layer::RowEmbedding>;

  // Dimensions inferred from template args
  Volume volume;

  EXPECT_TRUE(volume.getFilters().empty());
  EXPECT_TRUE(volume.initialize());

  for (const auto& filter : volume.getFilters())
  {
    // Rows should contain embedded depth (default)
    EXPECT_EQ(filter.rows(), 4);
    EXPECT_EQ(filter.cols(), 6 * 8);
    FFNN_DEBUG('\n' << filter);
  }
}

TEST(TestLayerReceptiveVolume, StaticInstanceRowEmbedding_Forward)
{
  // Volume-type alias
  using Volume = ffnn::layer::ReceptiveVolume<float, 4, 6, 8, 12>;

  // Dimensions inferred from template args
  Volume volume;

  EXPECT_TRUE(volume.getFilters().empty());
  EXPECT_TRUE(volume.initialize());

  using OutputBlock = Volume::BiasVectorType;
  OutputBlock output;
  output.setZero();

  using InputBlock = Volume::KernelMatrixType;
  InputBlock input;
  input.setOnes();

  volume.forward(input, output);
  FFNN_DEBUG('\n' << output);
}

TEST(TestLayerReceptiveVolume, StaticInstanceRowEmbedding_ForwardBlockInput)
{
  // Volume-type alias
  using Volume = ffnn::layer::ReceptiveVolume<float, 4, 6, 8, 12>;

  // Dimensions inferred from template args
  Volume volume;

  EXPECT_TRUE(volume.getFilters().empty());
  EXPECT_TRUE(volume.initialize());

  typedef Eigen::Matrix<float, 24, 2> OutputBlock;
  OutputBlock output;
  output.setZero();

  typedef Eigen::Matrix<float, 64, 12> InputBlock;
  InputBlock input;
  input.setOnes();

  EXPECT_NO_THROW(volume.forward(input.block<32, 6>(0, 0),  output.block<12, 1>(0, 0)));
  FFNN_DEBUG('\n' << output);
  EXPECT_NO_THROW(volume.forward(input.block<32, 6>(32, 0), output.block<12, 1>(12, 1)));
  FFNN_DEBUG('\n' << output);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
