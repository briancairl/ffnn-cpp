/**
 * @author Brian Cairl
 * @date 2017
 */
// C++ Standard Library
#include <exception>
#include <fstream>
#include <vector>

// Boost
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// GTest
#include <gtest/gtest.h>

// FFNN
#include <ffnn/logging.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/input.h>
#include <ffnn/layer/fully_connected.h>
#include <ffnn/layer/output.h>
#include <ffnn/io.h>

/***********************************************************/
// Creates network with one FullyConnected hidden layer 
// (dyanmic-sized) and saves it
//
// Tests:
//    - Save
//    - layer::Input
//    - layer::Output
//    - layer::FullyConnected
/***********************************************************/
TEST(TestFullyConnectedLayerIO, SaveDynamicSize)
{
  // Layer-type alias
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Hidden = ffnn::layer::FullyConnected<float>;
  using Output = ffnn::layer::Output<float>;

  // Layer sizes
  static const Layer::SizeType DIMS[2] = {32, 64};

  // Create layers
  std::vector<Layer::Ptr> layers({
    boost::make_shared<Input>(DIMS[0]),
    boost::make_shared<Hidden>(DIMS[1]),
    boost::make_shared<Output>()
  });

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[idx-1UL], layers[idx]));
  }

  // Initialize and check all layers and 
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->initialize());
    EXPECT_TRUE(layer->isInitialized());
  }

  // Forward activate
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->forward());
  }

  // Save all layer data to same file
  std::ofstream ofs("full_io_test.nnl", std::ios::binary);
  for(const auto& layer : layers)
  {
    EXPECT_NO_THROW(ffnn::save(ofs, *layer));
  }
  ofs.close();
}

/***********************************************************/
// Reconstructs network with one FullyConnected hidden layer 
// from saved data
//
// Tests:
//    - Load
//    - layer::Input
//    - layer::Output
//    - layer::FullyConnected
/***********************************************************/
TEST(TestFullyConnectedLayerIO, LoadDynamicSize)
{
  // Layer-type alias
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Hidden = ffnn::layer::FullyConnected<float>;
  using Output = ffnn::layer::Output<float>;

  // Layer sizes
  static const Layer::SizeType DIMS[2] = {32, 64};

  // Create layers
  std::vector<Layer::Ptr> layers({
    boost::make_shared<Input>(),
    boost::make_shared<Hidden>(),
    boost::make_shared<Output>()
  });

  // Load all layer data from same file
  std::ifstream ifs("full_io_test.nnl", std::ios::binary);
  for(const auto& layer : layers)
  {
    EXPECT_NO_THROW(ffnn::load(ifs, *layer));
  }
  ifs.close();

  // Check input/output sizes and connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    EXPECT_EQ(layers[idx-1]->outputShape().size(), DIMS[idx-1]);
    EXPECT_EQ(layers[idx]->inputShape().size(), DIMS[idx-1]);
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[idx-1UL], layers[idx]));
  }

  // Ensure we can't reinitialize a save layer
  // - This would overwrite data like weights, etc
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->initialize());
    EXPECT_TRUE(layer->isInitialized());
  }

  // Forward activate
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->forward());
  }
}

/***********************************************************/
// Attempt to reconstructs network with one FullyConnected
// hidden layer with a mismatched signature from saved data
//
// Tests:
//    - Load
//    - layer::Input
//    - layer::Output
//    - layer::FullyConnected
/***********************************************************/
TEST(TestFullyConnectedLayerIO, LoadSignatureMismatch)
{
  // Layer-type alias
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Hidden = ffnn::layer::FullyConnected<float, 4>;
  using Output = ffnn::layer::Output<float>;

  // Create layers
  std::vector<Layer::Ptr> layers({
    boost::make_shared<Input>(),
    boost::make_shared<Hidden>(),
    boost::make_shared<Output>()
  });

  // Load all layer data from same file
  std::ifstream ifs("full_io_test.nnl", std::ios::binary);
  EXPECT_NO_THROW(ffnn::load(ifs, *layers[0]));
  EXPECT_THROW(ffnn::load(ifs, *layers[1]), std::runtime_error);
  ifs.close();
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
