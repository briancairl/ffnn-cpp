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
#include <ffnn/layer/input.h>
#include <ffnn/layer/output.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/hidden/hidden.h>

class Hidden :
  public ffnn::layer::Hidden<float>
{
public:
  Hidden() :
    ffnn::layer::Hidden<float>(Hidden::ShapeType(5, 2), Hidden::ShapeType(2, 5))
  {
    FFNN_INFO("Input  : " << input_shape_);
    FFNN_INFO("Output : " << output_shape_);
  }

  bool forward()
  {
    FFNN_INFO("Input  : " << input_.rows()  << " by " << input_.cols());
    FFNN_INFO("Output : " << output_.rows() << " by " << output_.cols());

    std::memcpy(output_.data(), input_.data(), input_.size() * sizeof(float));
    FFNN_INFO("\n" << input_);
    FFNN_INFO("\n" << output_);
    return true;
  }

  bool backward()
  {
    return true;
  }

  bool update()
  {
    return true;
  }

  inline SizeType inputHeight() const { return input_.rows(); }
  inline SizeType inputWidth() const { return input_.cols(); }
  inline SizeType outputHeight() const { return output_.rows(); }
  inline SizeType outputWidth() const { return output_.cols(); }
};

TEST(TestLayerHiddenBasic, IOSizing)
{

  // Layer-type alias
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Output = ffnn::layer::Output<float>;

  // Layer sizes
  static const Layer::SizeType DIM = 10;

  // Create layers
  auto input = boost::make_shared<Input>(DIM);
  auto hidden = boost::make_shared<Hidden>();
  auto output = boost::make_shared<Output>();

  // Create network
  std::vector<Layer::Ptr> layers({input, hidden, output});

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

  // Create dummy input values
  Hidden::InputBlockType input_block(5, 2);
  for (size_t idx = 0; idx < DIM; idx++)
  {
    input_block(idx) = idx;
  }

  // Forward propagate
  (*input) << input_block;
  input->forward();
  hidden->forward();
  output->forward();
  (*output) >> input_block;

  // Check effective hidden input/output sizes
  EXPECT_EQ(hidden->inputHeight(), 5);
  EXPECT_EQ(hidden->inputWidth(), 2);
  EXPECT_EQ(hidden->outputHeight(), 2);
  EXPECT_EQ(hidden->outputWidth(), 5);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
