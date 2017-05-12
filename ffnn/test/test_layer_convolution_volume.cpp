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
#include <ffnn/layer/convolution_volume.h>
#include <ffnn/distribution/normal.h>

TEST(TestLayerConvolutionVolume, DynamicInstanceColEmbedding)
{
  // Volume-type alias
  using Volume = ffnn::layer::ConvolutionVolume<float>;

  // Shape supplied in constructor
  Volume volume(Volume::ShapeType(4, 6, 8), 12);

  EXPECT_EQ(volume.getFilters().size(), 12);
  EXPECT_TRUE(volume.initialize(ffnn::distribution::StandardNormal<float>(),
                                ffnn::distribution::StandardNormal<float>()));

  for (const auto& filter : volume.getFilters())
  {
    // Columns should contain embedded depth (default)
    EXPECT_EQ(filter.kernel.rows(), 4 * 8);
    EXPECT_EQ(filter.kernel.cols(), 6);
    FFNN_DEBUG('\n' << filter);
  }
}

TEST(TestLayerConvolutionVolume, StaticInstanceRowEmbedding)
{
  // Volume-type alias
  using Volume = ffnn::layer::ConvolutionVolume<float, 4, 6, 8, 12, ffnn::layer::RowEmbedding>;

  // Shape inferred from template args
  Volume volume;

  EXPECT_EQ(volume.getFilters().size(), 12);
  EXPECT_TRUE(volume.initialize(ffnn::distribution::StandardNormal<float>(),
                                ffnn::distribution::StandardNormal<float>()));

  for (const auto& filter : volume.getFilters())
  {
    // Rows should contain embedded depth (default)
    EXPECT_EQ(filter.kernel.rows(), 4);
    EXPECT_EQ(filter.kernel.cols(), 6 * 8);
    FFNN_DEBUG('\n' << filter);
  }
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
