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

TEST(TestLayerReceptiveVolume, InstanceDynamicColEmbedding)
{
  // Volume-type alias
  using Volume = ffnn::layer::ReceptiveVolume<int>;

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
    FFNN_INFO('\n' << filter);
  }
}

TEST(TestLayerReceptiveVolume, StatisDynamicRowEmbedding)
{
  // Volume-type alias
  using Volume = ffnn::layer::ReceptiveVolume<int, 4, 6, 8, 12, ffnn::layer::RowEmbedding>;

  // Dimensions inferred from template args
  Volume volume;

  EXPECT_TRUE(volume.getFilters().empty());
  EXPECT_TRUE(volume.initialize());

  for (const auto& filter : volume.getFilters())
  {
    // Rows should contain embedded depth (default)
    EXPECT_EQ(filter.rows(), 4);
    EXPECT_EQ(filter.cols(), 6 * 8);
    FFNN_INFO('\n' << filter);
  }
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
